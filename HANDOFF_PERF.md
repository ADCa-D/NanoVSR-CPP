# 性能改造：把逐 tile 的串行改成流水（GPU 利用率 60% → 90%）

## 现象

任务管理器里 GPU 利用率在 100% → 60% → 100% 之间来回跳，端到端 1080p 约 2.2 s/帧，
而单 tile 的纯 GPU 时间加起来只占 60%。

## 根因：不是"上下文切换"，是 CPU/GPU 在 tile 粒度上串行

```cpp
for (auto& tile : tiles) {
    打包 tile 像素到 staging;      // CPU，GPU 空转
    上传 + 推理（45 层）;           // GPU，CPU 空转
    下载 + 合成到输出帧;            // CPU，GPU 空转
}
```

1080p 下 21 个 tile = 21 次 CPU/GPU 交替。两次 `submit_wait` 把边界固定死，
所以谁也不能在对方工作时推进。

## 三个阶段的工作量（估算）

| 阶段 | CPU | GPU |
|---|---|---|
| 打包 u8→uint（每 tile 每帧 ~14 万像素） | 有 | 空 |
| 推理 45 层 | 空 | 有 |
| 合成 uint→RGB + 写管道 | 有 | 空 |

合计 CPU 侧约占端到端的 35–40%，这段时间 GPU 完全闲着。

## 方案 A（先做）：双缓冲流水

```
        CPU: [打包 tile0][打包 tile1][打包 tile2]...
        GPU:             [tile0 推理][tile1 推理][tile2 推理]...
        合成: 在 tile N 的 GPU 结果回来后立刻做（或再交给 GPU）
```

实现要点：

1. 两个 `VkCommandBuffer` + 两个 `VkFence`，对应两套 staging buffer
2. 提交 tile N 后**不等待**，直接开始打包 tile N+1
3. 在打包 N+1 之前 `vkWaitForFences(fence[N-1])`，确保那套 staging 可复用
4. 合成放在 fence 等待之后，与下一个 tile 的打包交错

伪代码：

```cpp
int slot = 0;
for (size_t i = 0; i < tiles.size(); ++i) {
    if (i >= 2) { vkWaitForFences(fence[slot]); vkResetFences(...); }
    pack_into(stage[slot], tiles[i]);          // CPU：本块
    record_and_submit(tiles[i], stage[slot]);  // 不等待
    if (i >= 1) composite(prev_result_slot);   // CPU：上一块的结果
    slot ^= 1;
}
```

风险：`bout`/`bstage_out` 目前是单份，双缓冲后需要两份（大 tile 下各 8-16 MB，可接受）。

## 方案 B（之后）：打包/合成多线程

打包是纯内存循环，按行切给 `std::thread::hardware_concurrency()` 个线程即可。
实测这类循环在 8 核上通常 6–7× 加速。

## 方案 C（可选）：多 queue 重叠拷贝

Vulkan 的 transfer queue 与 compute queue 分离时，可以让 `copy → VRAM` 与
上一块的 compute 重叠。收益小（ReBAR 下拷贝量本就很小），优先级最低。

## 预期

| 阶段 | 利用率 | 1080p 端到端 |
|---|---|---|
| 现在 | 60% | 2.2 s/帧 |
| A | 85% | ~1.55 s/帧 |
| A+B | 92% | ~1.45 s/帧 |

注意这是端到端提速，**单 tile 的 GPU 时间不变**。

## 验收方法

1. 在 `pack` / `submit` / `composite` 三处打时间戳，统计各自的占比
2. 跑同一段素材，比较改前改后的 fps（同 chunk、同模型）
3. 用 GPUView / 任务管理器确认利用率曲线变成平顶而不是锯齿

## 边界

- 单 tile 的场合（≤480p 输入）本来就没有交替，收益接近 0 —— 这是 1080p 及以上的优化
- 双缓冲会让显存/内存各多一份 staging，大 tile 时需要重新核账本（`S_PX` 里的 staging 项 ×2）

---

## 实施结果（2026-11-17，已完成代码，待 NV 上验收）

方案 A + B 都已落地，全部在 `src/nanovsr.cpp`：

- **双槽命令缓冲**：`Gpu` 里 `NSLOT=2`，`cbs[2] / fences[2] / dpools[2] / slot_pending[2]`，
  `cb` 始终是 `cbs[cur]` 的别名（fill/copy/dispatch/barrier 内部无需改）。
  新增 `begin(slot)` / `submit(slot)`（不等待）/ `wait(slot)` / `reset_pool(slot)` / `submit_wait()`。
  `begin(slot)` 内必须 `wait(slot)` + `vkResetCommandBuffer`，否则重录同一 cb 会 device lost（已踩过）。
- **CPU 侧三份数据各两套**：`bstage_in / bstage_in_b`、`bout / bout_b`、`bstage_out / bstage_out_b`。
  GPU 推理缓冲**只有一份** —— 同一 queue 的提交顺序保证后一个 cmd 在前一个之后执行，不需要双份。
- **交错关系**：提交 tile N 后立刻打包 tile N+1；合成 tile N 放在下一次用到该 slot 之前（`hold[slot]`）。
- **打包/合成多线程**：`parallel_rows` 按行切给 `hardware_concurrency()` 个线程；`pack_tile` / `composite_tile`。
- **开关**：`--serial` 恢复老的逐 tile 串行（同一份二进制做 A/B）；`--gpu N` 指定设备索引（跳过 DISCRETE_GPU 选择）。
- **时间分解**：运行结束打印 打包 / 录制 / 合成 / 等GPU(fence) / 其余 的秒数与占**总墙钟**的百分比。
  注意 `acc_rec` 已扣掉 `begin()/reset_pool()` 内部的 fence 等待，否则和"等 GPU"重复计数、百分比之和 >100%。

### 实测结果（2026-11-17，RTX 5060 Laptop，重启驱动后）

素材 clip_1080.mp4（内部降到 1024x576 → 输出 4096x2304），--tile 200x144（24 块），--limit 10，crf 1：

| | 总墙钟 | fps | 等 GPU | 打包 | 录制 | 合成 | 其余 |
|---|---|---|---|---|---|---|---|
| 流水（默认） | **9.29 s** | 1.08 | 88.1% | 3.2% | 1.3% | 3.7% | 3.6% |
| --serial | 10.13 s | 0.99 | 89.6% | 3.0% | 1.2% | 3.6% | 3.4% |

- **提速 8.3%**，输出**逐位一致**（抽第 7 帧用 psnr.py 对拍得 maxabs=0、MSE=0）。
- 真正的验收标准是“对得上 torch”，改前改后一字不差：kn_in.mp4 第 5 帧 crf 0
  644k **45.0817 dB**、5.4m **45.0842 dB**（与改造前的基线完全相同）。
- **重要修正**：原判断“CPU 侧占 35–40%、GPU 因此只有 60% 利用率”**是错的**。
  实测打包+录制+合成合计只有 **8% 左右**（打包/合成已多线程），GPU 本来就有 88% 在忙。
  所以方案 A 的天花板就是这 8%，本次已经基本吃满；再往上只能减少 GPU 工作量或块数，不是靠重叠。
- 时间分解里的“等 GPU”就是 GPU 忙的比例，1 − 等GPU% 才是在等 CPU。

### 实施中踩到的两个坑（都已修，别再犯）

1. begin(slot) 里只调 vkBeginCommandBuffer 不调 vkResetCommandBuffer → 第二次重录同一个 cb 是 UB，
   表现为 vkQueueSubmit failed: -4（DEVICE_LOST），并且会把 NVIDIA 驱动弄进坏状态
   （之后任何 vkCreateDevice 都崩 0xC0000409，只能重启机器）。
2. 老代码在 tile 中间有一对 g.reset_pool(); g.begin();（原本用来分隔两次提交）。改成单 cb 后忘了删，
   它会把已录制的**前半段全部丢掉**，输出变成几乎全黑但程序不报错 —— 而且“流水 vs 串行”两边都黑，
   对拍反而显示逐位一致，很容易骗过验收。教训：单 cb 化之后必须 grep 一遍所有 begin(/submit( 边界，
   并且一定要用 torch 参考（psnr 应 ~45 dB）而不是只做自己跟自己的 A/B。

---

## 第二轮（2026-11-17）：按问题清单修 12 条

清单分三组：并发/显存安全、GPU 调度、CPU-IO 流水。逐条处理结果：

| # | 问题 | 处理 |
|---|---|---|
| 1 | 双缓冲竞态 | `Gpu::begin(slot)` 里改成 `for (s in NSLOT) wait(s)`：两槽共用 bu8/bfeat/bfwd/bp0/bp1 等中间缓冲，录制前必须等**所有**在飞提交落地。队列只保证执行顺序，不保证内存可见性。 |
| 2 | 纯 GPU 缓冲用了 HOST_VISIBLE | 新增 `Gpu::alloc_device()`（`mem_type(bits, 0)` = 只要 DEVICE_LOCAL，不映射）。bx/bu8/bfeat/bfwd/bbwd/bp0/bp1/bcatus/bcat/bfused/bconv1/bup1/bconv2/bup2/blast/bout/bout_b 全部改用它；只有 4 个 staging 还走 `alloc_host`。 |
| 3 | 死代码 | `ffmpeg -filters` 探测改成只在“超采样且 downscale != lanczos”时执行；`enc_scale.clear()` 那个分支与整段 `if (false) {…}` 删除，重写成 `if (pre > 1 && supersample) … else if (pre > 1) …` + 一行统一输出分辨率。 |
| 4 | 热循环调试打印 | `if (name == "forward_0") printf("[diag]")` 删除（CM 路径已验证，不再需要）。 |
| 5 | 越界/溢出 | `next()` 加 `i + 1 < argc`；`jint` 找不到 key 时返回默认值（原来把 npos 当偏移）；层解析加 `pe/q1/q2 == npos` 保护；`w_off/b_off` 改用 `strtoll` 走 64 位（5.4m 的偏移超 2GB）；prelu 解析同样加保护。 |
| 6 | run_dir 冗余拷贝 | 最后一层（`b + 1 == nb`）直接写 `dst_all` 的 `tt*px*F*4` 偏移，prop 改成 `(Buf*, off)` 二元组。省掉每个方向每帧一次整帧 copy + barrier。 |
| 7 | 帧循环过度 barrier | pF32 整 chunk 一次 dispatch（`H = t*px`）；feat_extract 逐帧 dispatch 时 `do_barrier=false`，循环后补一次 `g.barrier()`；`Gpu::dispatch` 增加 `do_barrier` 参数。bfwd/bbwd 共用 bp0/bp1，且单 queue 上本来就是顺序执行，没有 host 等待可省，未动。 |
| 8 | 描述符静态缓存 | **未做**：时间分解里“录制命令”（含每次 vkAllocateDescriptorSets + vkUpdateDescriptorSets + 字符串查表）现在只占 **1.0–1.3%**，这就是收益上限，改动的复杂度和风险不成比例。 |
| 9 | 边缘 tile 按 px 调度 | pF32 按 `t*px` 合并后仍是整缓冲；pAdd 改成 `W*H*F`、pCat 改成 `W*H`、pFinal 改成 `(H*4)*(W*4)`，都用当前 tile 的真实规模，不再按最大 px 空转。 |
| 10 | 编解码与 GPU 串行 | 新增写线程：`std::thread writer` 持 `std::deque<WriteItem>` 写 fout，主线程只把整 chunk 的 `out_frames` 指针入队就继续跑下一个 chunk。输出**双缓冲轮转**（`obuf[2]`），复用前用 `written >= pushed - 1` 等它写完，最多一个 chunk 在飞，不会内存膨胀。 |
| 11 | 每 chunk 重复分配清零 | `frames` 与 `obuf[2]` 提到 while 循环外，按 `chunk` 一次分配；`fread` 不再逐帧 resize，`out_frames.assign(…, 0)` 的每轮清零删除（composite 会写满每个像素）。 |
| 12 | 每 tile 建/join 线程 | 新增常驻 `RowPool`（`n = hardware_concurrency()`，worker 用 gen_ 世代号唤醒，主线程做第 0 片），`parallel_rows` 只剩 `pool.run(nrows, fn)`。 |

### A/B 实测（同一台机器连续跑，RTX 5060 Laptop）

素材 clip_1080.mp4，`--model-size 1.7m --tile 200x144`（90 块）`--limit 10 --crf 1`，改前二进制保留为 `nanovsr_before.exe`：

| 版本 | 调度 | 总墙钟 | 等 GPU | CPU 侧 |
|---|---|---|---|---|
| 改前 | --serial | 15.93 s | 91.2% | 5.0% |
| 改后 | --serial | **15.02 s** | 96.5% | 1.0% |
| 改前 | 流水 | 15.06 s | 91.4% | 5.2% |
| 改后 | 流水 | **14.79 s** | 96.3% | 1.1% |

- 串行段（纯 CPU 开销）**-5.7%**，流水整体 **-1.8%**。GPU 自己的计算时间没变（≈14.5 s，两版几乎相同）。
- **绝对值和上一节的 9.29 s 不可比**：那次是 24 块 tile、机器时钟状态也不同。只有同一批里改前/改后的对比有意义。
- 结论没变：这个 work 是 **GPU 饱和型**（等 GPU 96%），CPU 侧的优化空间已经只有 1% 左右。

### 正确性（改完必须重新对拍）

- `kn_in.mp4` 第 5 帧 `--crf 0 --codec h264`：644k **45.0817 dB**、5.4m **45.0842 dB**，与改前**完全相同**。
- 改前输出 `base_old.mp4` 与改后 `after2.mp4` **逐位一致**（文件大小同为 6124037 B，抽帧 maxabs=0）。
- 图片输入（`--in *.png`）与 `--serial` 路径都复测通过。

---

## 第三轮（2026-11-17 晚）：双缓冲回归 + 描述符缓存 + 解码预读

第二轮把 `begin(slot)` 写成「等所有槽」，等于每录一块 tile 前都死等 GPU，双缓冲实际失效；
同时热循环里还在拼字符串、查 map、现分配描述符，fread 也还是同步的。本轮补齐：

| 项 | 改法 |
|---|---|
| begin 全槽等待 | 改回 `wait(slot)` 只等本槽；在 `vkBeginCommandBuffer` 之后立刻 `barrier()`。同队列提交顺序已保证执行先后，缺的只是读写可见性，cmd 开头一条 barrier 就等于给本块与所有前序提交建立依赖，CPU 不必等另一槽的 fence。 |
| 描述符分配 | 新增 `Gpu::set_for()` + `std::map<DsKey, VkDescriptorSet> dscache`，key = 4 个绑定的 (buffer, offset, range)。缓冲整个运行期只分配一次、偏移只由 px/帧号决定，所以首次之后全命中。配套把每槽描述符池 `dpools[NSLOT]` 换成单个永久池 `dpool`（永不 reset），`reset_pool()` 删除。 |
| 层查找 | 新增 `struct ConvPlan` + `make_plan()`，初始化时把 `forward_0..nb-1` / `backward_0..nb-1` 的层下标、O、k、grid.z 预解析成 `fwdPlan`/`bwdPlan`，热循环里只做一次数组下标，不再 `std::string` 拼接 + `idx[name]` 查 map（`run_plan()`）。 |
| 解码 | 新增预读线程 + 输入双缓冲（`inbuf[2]`，每份 `chunk` 帧）；状态机 0 空闲 / 1 填充中 / 2 已就绪或使用中，读线程只碰状态 0 的那份，主线程消费完再还回去。最多提前一个 chunk。 |
| 显存账本 | 4 份 staging（`bstage_in`/`bstage_out`/`bstage_in_b`/`bstage_out_b` = 4+64 各两份）都走 `alloc_host`，是系统内存，不占显存堆，从 `S_PX_F` 里去掉 136 B/px/帧（只留 `bout` 与 `bout_b`）。 |

### 实测（同机同素材、同命令，与上一节直接可比）

| 场景 | 上一轮 | 本轮 |
|---|---|---|
| clip_1080 / 1.7m / `--tile 200x144` 90 块 / 10 帧 / 流水 | 14.79 s（录制 1.3%） | 14.84 s（录制 **0.5%**） |
| 同上 `--serial` | 15.02 s | 15.01 s |
| kn_in / 1.7m / `--chunk 8` / 24 帧 | 5.98 s | 5.97 s |

- **墙钟没变**：这个 work 等 GPU 96%，CPU 侧总量只有 1%，天花板本来就这么低。
- 但 CPU 侧内部时间确实砍下来了：录制命令 0.19 s → **0.07 s（-63%）**；多 chunk 场景的
  「其余(读写编)」0.37 s → **0.15 s（-60%）**（解码与 GPU 重叠生效）。这两项在 CPU 受限的场合
  （小模型、大 tile、慢机器）才有意义，5060 上属于「顺手做对」。
- `begin` 的全槽等待在 GPU 饱和时本来就测不出来（这也是第二轮没发现的原因），但结构上是错的：
  GPU 一变快或 tile 一变少，它就会立刻变成串行瓶颈。

### 正确性

- 644k **45.0817 dB**、5.4m **45.0842 dB**，与改前完全相同；644k `--limit 6` 输出仍 6124037 B（逐位一致）。
- 预读线程没有错帧：`--chunk 8 --limit 16`（2 chunk）与 `--limit 24`（3 chunk）的第 15 帧 maxabs=0。
- 图片输入正常；5.4m + 1080p（30 帧、auto tile 336x336、8 块）无 OOM，账本余 321 MB。

### 账本实测（去掉 staging 后）

| 场景 | 时间维 | 合计 / 预算 |
|---|---|---|
| 644k / kn_in / chunk 30 | 0.70 KB/px/帧 | 5.11 / 5.98 GB（余 868 MB） |
| 5.4m / clip_1080 / chunk 30 | 1.27 KB/px/帧 | 5.66 / 5.98 GB（余 321 MB） |

去掉 136 B/px/帧 后理论值更准，最紧的一档还剩 321 MB 余量，没有把 tile 撑爆。

## 死代码清理（2026-11-17 深夜）

用 `cl /W4`（屏蔽 4244/4267/4245/4100/4996）扫三个 host，剩下的“定义了没人用”全部清掉：

| 位置 | 内容 | 处理 |
|---|---|---|
| `src/nanovsr.cpp` | `const size_t max_px = ...`（算完只有 `(void)max_px;`） | 删 |
| `src/nanovsr.cpp` | `Gpu::submit_wait()`（只剩定义） | 删 |
| `src/nanovsr.cpp` | `const int src_w = W, src_h = H;` | 删 |
| `src/nanovsr.cpp` | `const VkPipeline pShuffle = g.pipe("shaders/shuffle_prelu.spv")`（nanovsr 只用 `pShuffle16`，非 f16 变体只有 model_test/bench_model 用） | 删 |
| `src/nanovsr.cpp` | `const double t1 = ...`（计时残留） | 删 |
| `SPIR-V` | 14 字节，内容是 `[1/2] GLSL -`，误提交的重定向产物 | `git rm` |
| `shaders/conv3x3.spv` | 25 KB，没有对应 `.comp`（CM 之前的老 SIMT 内核遗留） | 删 |
| `shaders/z.spv` | 1.6 KB，没有对应 `.comp` | 删 |

验证：`build.bat` EXIT=0；`/W4` 三个 host 零 C4189；644k `--limit 6 --crf 0 --codec h264` 输出仍 **6124037 B**（逐位一致）；`model_test.exe testrun_644k` 逐层 OK、`final_u8` 差 1 级。

第二批（用户点头后执行）：

- 删 `src/vk_conv_test.cpp`、`make_testdata.py`、`make_testdata_nhwc.py`（`testdata/` 早已不存在，只测 C=64 SIMT 老内核），build.bat 的 `[2/3] C++ (unit tests)` 段整段去掉，节号改成 `[1/2]` / `[2/2]`。
- `src/vk_info.cpp` / `vk_cm_info.cpp` 保留（一次性设备探针，下次查 coopmat 支持还用得上，没进构建）。
- `nanovsr.cpp` 加权重目录回退：先 `models_rt/nanovsr_<size>`，再 `models/models_rt/nanovsr_<size>`（发布包布局），再按 644k/1.7m/5.4m 找已装的任意一档，启动打印 `权重目录: ...`。发布包里 `nanovsr.exe` 直接跑（不敲 `--model`）现在也能工作。
- `_release/nanovsr-cpp-v1.0.0-win64/` 重新打包：release exe（`/fp:fast /GL /LTCG`，输出与普通构建 md5 一致）+ 17 个 spv + 新 webui/pack_model/README/launchers，清掉 `conv3x3.spv`、`z.spv`、`__pycache__`、`work/`；`nanovsr-cpp-v1.0.0-win64.zip` 一并重生成（29.2 MB / 30 文件）。
补一个根因：根目录那个 14 字节 `SPIR-V` 垃圾文件不是误提交，而是 `build_release.bat` 里
`echo [1/2] GLSL -> SPIR-V` 的 `>` 被 cmd 当成重定向，每次打包都会原地重新生成（内容 `[1/2] GLSL -`）。
已改成 `echo [1/2] GLSL to SPIR-V` 并删掉文件，跑过一遍两边构建确认不再出现；详情见 `reflection/2026-11-17-batch-echo-redirect-junk-file.md`。
## 第四轮（2026-11-18 凌晨）：两份外部评审逐条核对

评审来源：`C:\Users\MECHREVO\Downloads\回应nanovsr-byGemini4.txt`、`回应nanovsr-by千问4.txt`。
核对方式：每条都回到 `src/nanovsr.cpp` 看代码再判定，不照单全收。

### 确认并已修（18 条）

| # | 条目 | 代码位置（改前） | 修法 |
|---|---|---|---|
| 1 | feat_extract 的 barrier 开关传反（除最后一帧外每帧都下 barrier，与注释意图相反） | `run_convg(..., act, i + 1 < t)` | 传 `false`，循环后已有一次 `g.barrier()` |
| 2 | `read_file` 用 32 位 `ftell`，权重 >2GB 时返回 -1 → vector 天文数字 | `read_file()` | `_fseeki64` / `_ftelli64` + 负值检查 |
| 3 | `WLayer::cm_off` 是 `int`，且用 `get()`（= `(int)getll`）读 | `struct WLayer` | `long long cm_off` + `getll("cm_off")` |
| 4 | 手动 `--tile` 大于帧尺寸时 `min(x, W - tw)` 为负 → 打包越界；且手动 tile 不守显存预算 | `grid_for()` / 手动指定块 | 入口夹 `tw=min(tw,W)`、`th=min(th,H)`；超预算按 sqrt 比例缩并打印（实测 2000x1000 → 592x320） |
| 5 | 清单缺 key 时 `getll` 返回 -1 → 偏移成 `(size_t)-1`，跑飞后只看到 DEVICE_LOST | 权重解析 | 启动时逐层核对 `w_off/b_off/cm_off` 是否越界，越界带层名与尺寸退出 |
| 6 | `char cmd[2048]` 拼编码命令，长路径/中文路径静默截断 | 编码器启动 | 改 `std::string` 拼接 |
| 7 | 帧率只取 `r_frame_rate`，VFR 素材（手机录屏常见 1000/1）输出时长错 | ffprobe 解析 | 同时取 `avg_frame_rate`，优先用它 |
| 8 | 权重缓冲走 `alloc()` = HOST_VISIBLE，非 ReBAR 机器上每个 dispatch 都在 PCIe 上读权重 | `bw` / `bwcm` | 改用 `alloc_device()` + staging 一次性上传后释放 staging |
| 9 | 描述符池 storage buffer 只给 32768，而每个 set 用 4 个 → 8192 个 set 后池满 | `vkCreateDescriptorPool` | `descriptorCount = 16384 * 4` |
| 10 | 核显（UMA）上 `alloc_host` 把所有堆都判成 DEVICE_LOCAL 而全跳过，退回到 WC 内存，合成时 CPU 读得极慢 | `alloc_host()` | 三趟策略：系统内存+cached → 系统内存 → 核显（DEVICE_LOCAL 但 cached） |
| 11 | 单帧图片仍按 chunk=30 分配时间维缓冲 | chunk 计算 | `if (in_is_image && limit > 0) chunk = min(chunk, limit)`；854x480 实测时间维 4.9 GB → 165.9 MB |
| 12 | 末尾 tile 被 `min(x, W-tw)` 往回推后重叠大于 `ov`，接缝仍用固定 `ov/2` → 重复写/漏写 | `composite_tile` | `TileRect` 增加 `cx0..cy1`，按**实际重叠区中线**切分 |
| 13 | `--codec hevc/av1` 匹配不到 `libx265`/`libsvtav1`，静默回退 x264 | 编码器候选 | 加别名匹配；未知编码器打印回退提示（实测 `--codec hevc` → `libx265 -crf 20`） |
| 14 | 单次提交可有 1500+ dispatch，大 tile 有 Windows TDR 风险 | `Gpu::dispatch` | `max_per_submit` 默认 512（SIMT 通路仍是 8） |
| 15 | `RowPool` 每个 worker 完成都 `notify_all`，24 核下每次 run 广播 23 次 | `RowPool::worker` | 只有最后一个完成的线程 `notify_one` |
| 16 | 进度行 `[tile N/N]` 两个参数都是 `tiles.size()`，永远是满值 | 进度打印 | 改成 `[N 块/帧]` |
| 17 | 写线程失败后主循环继续把整片算完 | chunk 循环 | 每 chunk 开头 `if (wq_err) break;` |
| 18 | `--chunk` 超 30 被静默夹断 | chunk 计算 | 打印截断提示 |

### 判定为误报或不改（记录理由）

- **收尾 flush「槽位顺序颠倒」**：两个槽各自有 fence，`wait(0)` 等的是 slot0 自己的 fence，不会串到别人的提交上；合成区已按不重叠划分，先后顺序不影响像素结果，只影响先等哪一块（总时长相同）。
- **「把 composite_tile 挪到 submit 之后」**：合成读的是本槽 `bstage_out`，而 GPU 下一块最后也会写同一份 `bstage_out` → 时序上有竞态，要安全得再加第三份 sout（4K 下几百 MB）。而 CPU 侧实测只占 1.1%，收益为零。
- **「前后向链成对派发省一半 barrier」**：需要第二套 `bp0/bp1/bcatus`；但实测等 GPU 占 96%，barrier 不是瓶颈，节省的只是 GPU 上的同步气泡（未测出）。
- **「后半段扩成 mini-batch」**：要重排 `bcat/bfused/bconv1/bup1/bconv2/bup2` 全部缓冲，GPU 计算量不变，收益同样未证实。
- **chunk 交界加 overlap 帧**：上一轮实测 chunk 2/8/30 的逐帧亮度差在窗口边界**没有尖峰**，画质差 ~1 级；不做重叠（chunk 是显存/吞吐旋钮，不是画质旋钮）。
- **加 NVENC 候选进默认序列**：只加别名匹配，不改默认顺序（软件 AV1/HEVC 画质更稳，且 NVENC 会改变所有人默认输出）。
- **不做**：`--debug` 打开验证层、buffer/fence/pipeline 的 RAII 化、`obuf` 从 2×chunk 降到按帧（改这三项都会动到已经验证过的双槽/写线程逻辑，收益与风险不成比例，先记录在案）。

### 第四轮实测（RTX 5060 Laptop，ComfyUI 已退出）

| 项目 | 结果 |
|---|---|
| 644k / kn_in 第 5 帧 / crf 0 / h264 | **45.0817 dB**（MSE 2.0180, maxabs 16）与改前一致 |
| 5.4m / 同上 | **45.0842 dB**（MSE 2.0168, maxabs 15）与改前一致 |
| 1.7m / clip_1080 / --tile 200x144 / 10 帧 | 14.72 s（改前 14.79~14.84 s），等 GPU 96.1%，CPU 侧 1.1% |
| 5.4m / clip_1080 / 自动 tile 336x336 | 账本 5.66 GB / 5.98 GB（余 321 MB），正常出片 |
| 单帧图片 / 644k | 时间维 165.9 MB（改前 30 帧 ≈4.9 GB），tile 更大、更快 |
| `--tile 2000x1000`（帧只有 854x480） | 自动缩到 592x320 并提示，不再越界 |
| `--codec hevc` | 正确选中 `libx265 -crf 20`（原来静默回退 x264） |

结论：这一轮全是「正确性 + 小钱」型修补，GPU 端计算量未变，墙钟变化在噪声内；真正剩下的性能天花板仍是「等 GPU 96%」。

## 第五轮（2026-11-18）：第 5 份评审的 4 条 + shader 微优化实测 + 重叠自适应

### C++ 四条（Gemini 第 5 份评审）

| 位置 | 问题 | 结论 |
|---|---|---|
| 权重越界检查 | 用 `w_off + need_rt > size()` 判越界，`w_off == (size_t)-1` 时无符号回绕成小值 → 漏检 | **真 bug，已修**：改成减法写法（`need_rt > size() - w_off`），三处偏移都改 |
| `max_per_submit = 512` | 每个 tile 约 2700 次 dispatch，中途 `submit(wait(begin))` 会把双槽流水打回串行 | 已改成 8192（一个 4K tile 墙钟约 0.4 s，远低于 2 s TDR）。实测 90 tile 场景 14.72 → 14.66 s，**噪声级**——因为本来就 GPU 饱和，排空队列不额外浪费 GPU 时间 |
| `alloc_host` 四趟顺序 | pass 2 未校验 cached，核显 UMA 上会挑到 write-combined 内存 | 已改成严格四趟：系统+cached → 核显+cached → 系统+WC → 核显+WC |
| flush 顺序 | 奇数块时先等后提交的槽 | 已改成按 `hold`（提交序号）升序等待 |

### shader 实测（Gemini 提的三条，结论与它相反）

- 已实施：`s_out` 落盘阶段 8 次 `barrier()` → `subgroupBarrier()`（每个 warp 只碰自己那 256 个 float，跨 warp 无数据交换；需要 `--target-env vulkan1.1`，build 两个 bat 已改）；权重片段 `fb` 预取一级与 mma 重叠。
- 结果：1.7m 自动 tile 4.75 → 4.72 s（-0.6%，噪声）。**barrier 不是瓶颈。**
- 为什么：每 warp 一轮 `(tap,cc)` 是 4 次 A 片段（shared）+ 1 次 B 片段（global）+ 4 次 mma，即 **1.25 个 512 B 片段载入/mma**。shared 带宽 128 B/cycle → 每轮约 20 cycle，而 4 次 mma 约 8 cycle → shared 载入带宽超配 2.5 倍。这与实测 5.7~5.8 TFLOPS（约为 5060 Laptop 稠密峰值 20~25%）完全吻合。
- 所以「把 shared 从 33.5 KB 压到 30 KB 提升驻留度」这条：本机 per-SM shared 足够放 2 个 block（16 warp），驻留度本来就不是限制；改错方向。
- 真正能提速的是换 warp 分块：每 warp 32px×32oc（8 个累加器）→ 0.75 片段载入/mma，理论 +10~20%。但这是重写 kernel，收益还小于重叠浪费，暂不做。
- `shuffle_prelu` 的散写/银行冲突确实存在，但它只搬 32 通道 × 4x 分辨率 ≈ 300 MB/帧，约 1.4 ms，占 1.7m 单帧墙钟（约 400 ms）的 2%：不动。

### 重叠自适应（这一轮唯一的真收益）

把固定 `ov = 96` 改成 `ov = 2 * (nb + 4)`（空间感受野半径 = 每条链 3x3 层数 + 1，再留 4 px 余量），新增 `--ov N` 手工覆盖。

| 模型 | ov | tile（1024x576） | 10 帧墙钟 | 对 torch PSNR |
|---|---|---|---|---|
| 644k | 96 → 32 | 448x256 / 4 块 | — → **2.46 s（4.1 fps）** | 45.0816 dB（原 45.0817） |
| 1.7m | 96 → 48 | 416x336 → 544x224 / 6 块 | 4.76 → **4.16 s（2.4 fps）** | 45.0874 dB（原 45.0817） |
| 5.4m | 96 → 68 | 336x336 / 8 块 | 15.07 → **13.82 s（0.7 fps）** | 45.0839 dB（原 45.0842） |

接缝验证：同一素材 ov=96 与 ov=44/68 逐帧对拍 72.8 / 69.6 dB（maxabs 4），差异是 f16 舍入级，没有接缝线。

结论：GPU 仍然 84~97% 在忙，但**每减少一份重叠就是实打实的 GPU 计算量下降**。当前实算像素/有效像素比为 1.8（1.7m）、1.5（5.4m），继续往下压要靠「更小的感受野」（换更浅的模型）或「更大 tile」（显存不够），已经接近这条路的尽头。

## 第六轮（2026-11-18）：fusion 走张量核（1x1 流式内核）

分段计时发现的最贵一项：fusion（1x1，C=2F→F）只占全模型 0.56% 的 MAC，却吃掉 **20.6%** 的墙钟 ——
因为它 C=128/96/192 > 32，`run_convg` 把它交给标量 `conv_gen`（约 0.09 TFLOPS，比张量核慢约 60 倍）。

### 为什么不能直接复用 3x3 内核
3x3 版把整块 halo tile（`SH*SW*CPAD`）放进 shared：C=192 时 72 KB，超过设备 48 KB 上限。
1x1 不需要 halo，A 片段按 16 通道分块流式载入 + 双缓冲，shared 恒定 `2*128*16*2B = 8KB` + 落盘 8KB。

### 改动
| 文件 | 内容 |
|---|---|
| `shaders/conv1x1_cm.comp`（新增） | 256 线程 / 32x4 像素 / 8 warp，warp 分工与 3x3 版一致；一次 barrier 同时保证「下一块载入完成」和「本块已被读完」 |
| `src/nanovsr.cpp` | `cm1x1(C)` 惰性建管道（缺变体则打印并退回 SIMT）、`run_convg` 增加 `L.k==1 && cm_off>=0` 分支、权重越界检查的 `need_cm` 由固定 9 tap 改成 `k*k` |
| `pack_model.py` | 1x1 层也进张量核 blob（布局 `[tap][C/16][ocg][16][16]`，taps=1） |
| `src/model_test.cpp` / `bench_model.cpp` | `cm_off>=0` 且 `k==3` 才走张量核（这两个工具没接 1x1 内核，fusion 退回 SIMT） |
| `build.bat` / `build_release.bat` | 新增 `conv1x1_cm_{96,128,192}.spv` 三个变体 |
| `models_rt/*` | 重跑 `pack_model.py --weights-only`（`weights_rt.bin` 逐字节不变，只多出 fusion 的 cm 块） |

### 实测（RTX 5060 Laptop）
| 场景 | 模型 | 改前 | 改后 | 提升 |
|---|---|---|---|---|
| 1080p→4K，10 帧 | 644k | 2.46 s | **2.07 s** | −16% |
| 1080p→4K，10 帧 | 1.7m | 4.22 s | **3.36 s** | −20% |
| 1080p→4K，10 帧 | 5.4m | 13.82 s | **11.50 s** | −17% |

吞吐（含编码）：480p 输入 644k 7.7 / 1.7m 5.1 / 5.4m 1.3 fps；300p 输入 644k 26.6 / 1.7m 16.7 fps。

正确性：对 torch 参考 644k 45.0820 / 1.7m 45.0872 / 5.4m 45.0841 dB（改前 45.0817 / 45.0817 / 45.0842，差值是 f16 舍入级）；
`model_test.exe` 三套 testrun 的 `final_u8` 仍是「最大差 1 级、0 通道越界」。

### 为什么 480p 实时做不到（算一下就知道）
每输入像素的 MAC：644k ≈ 1.07 M、1.7m ≈ 2.38 M、5.4m ≈ 6.77 M。480p（410 k 像素）单帧就是
0.88 / 1.95 / 5.55 TFLOP；要 30 fps 分别是 **26 / 58 / 166 TFLOPS**，而 5060 Laptop 的 fp16 稠密峰值约 23 TFLOPS，
也就是说 1.7m/5.4m 的 480p 实时在 fp16 下**物理不可能**，不是实现问题。
换成 300p（120 k 像素）：644k 30 fps 需要 7.7 TFLOPS —— 正好是现在这个实现的水平（实测 26.6 fps），
论文说的「300p 实时」和这台机器的表现是对得上的。

现在剩下的两大块是 40 层链（62%）和 up1/up2（37%），都在张量核上跑，等效约 6.7 TFLOPS = 峰值的 30%；
下一档提速只能靠重写 warp 分块（32px×32oc，shared 载入从 1.25 片段/mma 降到 0.75，理论 1.5~2x）。

## 第六轮（2026-11-18）：conv3x3_cm 内核重写（拿来主义 + occupancy 修正）

目标：论文（arXiv 2607.10495）在 Jetson Orin NX @25W 上 644k 27.2 fps / 1.7m 19.6 fps（REDS4 480x270→1080p），
而我们 5060 Laptop @115W 只做到 25~30% 峰值 —— 要按「别人走过的方法」重写内核，不自己发明。

### 参照与实现

- 参照 llama.cpp `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`（KHR_cooperative_matrix GEMM）：
  协作把整块搬进 shared → barrier → 从 shared 切 fragment → mma → barrier；shared 行步长加 pad 防 bank 冲突。
- 新 `shaders/conv3x3_cm.comp`：workgroup = `TH` 个 warp，`warp = tid/32` 正好一个像素行；
  每个 warp 负责 `TW/16` 个 A 片段 × 4 个 B 片段（64 输出通道），权重按 workgroup 协作搬进 shared（双缓冲 + 预取下一轮），
  落盘用 `s_out[warp*256]` + `subgroupBarrier()`。tile 参数在顶部：`const int TW = 16; const int TH = 8;`。
- **改 tile 必须同步改 host dispatch**：`src/nanovsr.cpp:1027` 的 `constexpr int CM_TW = 16, CM_TH = 8;`
  用于 1295（run_conv64）与 1394（run_plan）两处；1335 那处是 1x1 内核（`shaders/conv1x1_cm.comp`，TW=32/TH=4），别动。
  `src/model_test.cpp` / `src/bench_model.cpp` 里各有一处同样的 (15/16, 7/8) dispatch。

### 逐项实测（1.7m，512x288，clip 10 帧；旧内核 13.26 fps 为基线）

| 改动 | 结果 | 结论 |
|---|---|---|
| 4 warp / tile 32x4，0.75 片段/mma + 权重进 shared + 预取 | 13.12 fps，输出 bit-identical | 片段/mma 比不是瓶颈 |
| 共享行步长加 pad（128B → 160B） | 13.26 fps（+1%） | 4 warp 时 bank 冲突不显著 |
| 删掉 K 循环里的 barrier() | 13.54 fps（+2%） | 同步不是瓶颈 |
| 只把 TW 改成 16（host dispatch 仍是 32） | 20.97 fps「+58%」 | **假加速**：宽度只算了一半 |
| **TW=16 / TH=8（8 warp，tile 128 px，共享 41 KB）** | **15.93 fps（+20%）** | 真收益，bit-identical |
| 再去掉 bank pad（shared 35 KB） | 14.97 fps | 8 warp 时 pad 反而必要，恢复 |

### 端到端（clip_1080.mp4，内部 1024x576 → 输出 4K，10 帧，crf 1，自动分块 + 自适应重叠）

| 模型 | 第五轮（TW=32/4 warp） | 本轮（TW=16/8 warp） | 提速 |
|---|---|---|---|
| 644k | 2.46 s / 4.1 fps（4 块） | **1.91 s / 5.25 fps**（4 块） | +29% |
| 1.7m | 4.16 s / 2.4 fps（6 块） | **2.89 s / 3.47 fps**（6 块） | +44% |
| 5.4m | 13.82 s | **9.24 s / 1.08 fps**（8 块） | +50% |

正确性：644k `kn_in.mp4` 6 帧 crf0 h264 输出 md5 `4a7f365599279892c03c76ea01f8dcee`，与旧内核逐位相同；
`model_test.exe testrun_s / testrun_644k / testrun_54m` 三档 `final_u8` 最大差 1 级、超 1 级通道 0 个。
即本轮只是更快，数值完全不变（45.0817 / 45.0842 dB 那些结论继续有效）。

### 结论与剩余空间

- 真正的杠杆是 **每个 SM 驻留的 warp 数**（不是片段/mma 比、不是 bank 冲突、不是 barrier）：
  halo 放 shared 后 48 KB 上限卡住 tile 尺寸，4 warp/块时每 SM 只有 128 线程在跑。
- 同机 torch cuDNN fp16 参照：1.7m 512x288 T=10 = 432.7 ms/chunk（23.1 fps）；单层 3x3 卷积 15~19 TFLOPS。
  我们现在 15.93 fps → 与 cuDNN 还差 **1.45x**；再往上要换结构（更大 warp tile + 更省 shared 的 halo 方案），不是调参。
- 内存：`ncu`/Nsight Compute 本机未安装，无法看 occupancy 与 stall 原因；上述结论来自逐项对照实验。
## 第七轮（2026-11-18）：按「bug 发现就治」做一轮边界审计

用户口径：优化是长期的，但 bug 必须发现就治。这轮不追性能，专门拿刁钻输入去撞，
结果撞出 4 个「不报错但结果错/退出码错」的问题 + 2 条 UX：

| # | 问题 | 触发条件 | 影响 | 修法 |
|---|---|---|---|---|
| 1 | 假加速：shader tile 改了、host dispatch 没改 | 改 TW 后 | 只算一半宽度，看起来 +58% | `CM_TW/CM_TH` 常量 + 4 处 dispatch 同步 |
| 2 | tile ≤ 重叠 → 步长退化成 1 像素 | `--tile 32x32`（ov=32） | 98x62 的图切 2077 块，慢 300 倍 | `grid_for` 强制 tile ≥ ov+16；太小则退回自动 |
| 3 | 静默成功 | 输出目录不存在 / ffmpeg 不可用 | rc=0，脚本以为成功 | 检查 `pclose` 退出码、`wq_err`、`total<=0` |
| 4 | 旋转元数据未处理 | 手机竖拍（rotate=90） | 输出斜切花屏，且 rc=0 | 探测 `stream_side_data=rotation`，90/270 交换 W/H |
| 5 | `--gpu N` 越界静默 | `--gpu 7` | 悄悄用默认卡 | 打印越界提示 |
| 6 | 无视频流的文件报错为空 | 音频输入 | 「无法读取视频信息:  」 | 保持 rc=1，补明确文案 |

审计覆盖的输入：642x362 / 98x62 / 641x361（yuv444p 奇数边）/ 中文+空格路径 / 单帧图片 / 10bit HEVC / rotate=90 / 音频文件 /
--tile 过大或过小 / --chunk 0,1,2 / --crf -5,51 / --codec av1,hevc / --out .mkv,.webm / 缺失 ffmpeg / 不存在目录。

回归：644k `kn_in` 6 帧 crf0 md5 `4a7f365599279892c03c76ea01f8dcee` 不变；5.4m 同条件 5792952 B；1.7m clip_1080 10 帧 2.88 s；
`model_test testrun_644k` final_u8 差 1 级；642x362 → 2568x1448、641x361 → 2564x1444、98x62 → 392x248 尺寸全对。

细节与踩坑记录见 `reflection/2026-11-18-boundary-audit-silent-bugs.md`。

---

## 第八轮（2026-11-18）：分段落盘 + EOF 校验 + 回退重来

用户要求：机器或进程被打断时，别让整段输出作废 —— 「做个 eof 检测，如果文件压根播不了，就回退这个切片重来」。

做法：默认把输出切成` <out>.parts/NNNNN.mkv`，每段写完立刻用 `ffprobe -count_packets` 数包校验（并核对帧数），结束时 `-f concat -c copy` 拼成最终文件。主循环开局先扫一遍已有分段：连续且校验通过的分段直接跳过推理（解码仍要消费，保证帧对齐），只重跑第一个坏段 —— 也就是「回退这个切片重来」。

| 项 | 结论 |
|---|---|
| 正确性 | 644k kn_in 6 帧 crf0：分段输出与 `--no-seg` 第 5 帧 **bit-identical**（PSNR 99 dB，maxabs 0） |
| 删除末段后续跑 | 「续跑: 已有 1 个完整分段」→ 只补算缺的段，40 包帧数正确 |
| 截断末段（6000 B / 200 B） | 「最后一段 1 不完整（0 帧），丢弃重来」→ 自动重算，最终仍是 40 包 |
| 硬杀（5.4m 跑到一半 taskkill） | 磁盘留 1 个完整段 + 0 字节残段 → 重跑跳过 1 段、补算其余，16 包正确 |
| 拼接总帧数核对 | 每段包数之和必须等于已处理帧数，不等则报错并保留分段目录（防「少一段还静默成功」） |
| 性能 | 默认 chunk 30：分段 1.85 s vs `--no-seg` 1.85 s；chunk 8 + AV1 + `--seg-chunks 4`：4.90 s vs 4.93 s（`--seg-chunks 2` 变慢到 5.23 s） |

坑：concat 的 list 必须写绝对路径，否则 ffmpeg 按 list 所在目录解析成 `<parts>/<parts>/x.mkv`。每段重开编码器有固定开销（1080p AV1 ≈ 0.61 s、x264 ≈ 0.10 s），所以一段要跨 4 个 chunk（`--seg-chunks`）把 init 摊薄；chunk 很小时段太短反而会变慢。图片输出不分段。