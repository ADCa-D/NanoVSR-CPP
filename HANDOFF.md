# nanovsr-cpp · 交接

NanoVSR 4× 视频超分的 **C++ + Vulkan compute** 实现，不依赖 torch / CUDA / Vulkan SDK。

```
480p（270×480）→ 1080×1920：52.9 ms/帧 ≈ 18.9 FPS（纯推理，实测 9.5 TFLOPS）
端到端（含 ffmpeg 解码 + x264 编码）：10.9 FPS
数值：最终 8bit 输出与 PyTorch fp32 参考最大差 1 级、0 个通道越界
```

## 1. 怎么跑

```bat
:: 一次性：编译（GLSL -> SPIR-V，再编译 5 个 exe）
build.bat

:: 你平时只需要这两个（双击即可）
start_webui.bat          启动面板并自动开浏览器（http://127.0.0.1:7861）
upscale.bat              把视频拖到这个 bat 上 → 同目录生成 xxx_4x.mp4

:: 命令行
nanovsr.exe --in D:\clip.mp4 --out D:\clip_4x.mp4 [--chunk 8] [--crf 18]
```

### 三档 preset × 三种模型

```bat
nanovsr.exe --in a.mp4 --out a_4x.mp4                      :: balanced + 1.7m（默认）
nanovsr.exe ... --preset preview   :: chunk 8 + CRF22，最快
nanovsr.exe ... --preset quality   :: chunk 30 + CRF14，编码更保真
nanovsr.exe ... --model-size 644k  :: 0.7M 参数，最快
nanovsr.exe ... --model-size 5.4m  :: 6.1M 参数，最慢最细
```

| 模型 | num_feat | blocks | fps（8 帧含预热） | 权重 |
|---|---|---|---|---|
| 644k | 48 | 12 | 2.68 | 2.5 + 1.2 MB |
| 1.7m | 64 | 20 | 1.78 | 6.5 + 3.2 MB |
| 5.4m | 96 | 30 | 1.23 | 20.8 + 10.3 MB |

张量核 shader 按 `num_feat` 编译了 48/64/96 三个变体（`-DC_OVERRIDE`），
host 从权重清单读 `num_feat`/`num_blocks` 自动选。
注意：C=96 时 shared 用到 47.4 KB（上限 48 KB），再大就放不下了。

参数：`--chunk N`（时间窗，默认 30）、`--budget GB`（显存预算，默认驱动建议 x0.8）、`--tile WxH`（手动指定分块）、`--crf 18`、`--limit N`、
`--bench`（跳过编码）、`--ffmpeg <path>`、`--model <dir>`。

**空间分块是自动的**：引擎按显存账本选 tile，块之间重叠 96 像素
（覆盖 40 层卷积 ±42 的接收野），只把每块的中心区合成回去。

**显存账本**（启动时打印，`--budget GB` 可覆盖，默认驱动建议 ×0.8）：
```
  权重             10.2 MB  (固定)
  静态 tile 缓冲   17.1 KB/px   ← bconv2 8 KB 最大
  时间维缓冲       0.92 KB/px/帧 ← chunk=30 时占 60%
  ── 合计 = 权重 + px×静态 + px×chunk×时间维
```
tile 不是"开方取整"：先按预算枚举 160..768 的 (tw,th)，用真实切块逻辑算出
**总处理像素**，取最小的那个。实测 960×544：

| 预算 | tile | 块数 | 放大 | fps |
|---|---|---|---|---|
| 2.5 GB | 240×208 | 24 | 2.29× | 1.46 |
| 3.5 GB | 240×320 | 12 | ~1.8× | 1.75 |
| 6.0 GB | 384×320 | 6 | 1.41× | 2.17 |

注意 tile **形状**比大小更关键：366×366 在 960 宽上切 4 列（1.07 M 像素），
而 384×320 只切 3 列（0.74 M），后者快 55%。

**默认值对齐训练分布**（`D:\nanovsr\train.py`）：
| 训练配置 | 推理默认 |
|---|---|
| `--patch_size 256`（Vimeo/REDS 都裁 256×256） | tile **256×256** |
| Phase 2 REDS `--long_num_frames 30` | chunk **30 帧** |
| Phase 1 Vimeo 7 帧 | 可用 `--chunk 7` 复现 |

注意：**tile 大小不影响像素值**（模型全卷积、无全局算子，输出只取决于感受野邻域），
只影响重叠浪费与显存占用。实测 `tile 256 + chunk 30` 与 `tile 432 + chunk 8` 两版成片
PSNR 52.4 dB，差异来自时间窗和两次编码，不是分块本身。
真正影响质量的是 **chunk**：递推每 chunk 重置一次，8 帧 → 30 帧后重建次数从 64 次降到 17 次。

## 2. 目录

```
shaders/
  conv3x3_cm.comp        张量核 3x3（C=64，O%64==0，主路径，覆盖 40 个 block + up1/up2）
  conv3x3_c64.comp       SIMT 3x3（C=64，备用/对照）
  conv3x3_smallc.comp    小通道 3x3（feat_extract C=3、conv_last C=32→3）
  conv_gen.comp          任意 C/O 的标量卷积（fusion 1x1 等）
  shuffle_prelu.comp     PixelShuffle(2) + PReLU
  bilinear4_add_u8.comp  4x 双线性 + 残差 + clamp + 打包 u8
  add.comp / cat2.comp / u8rgb_to_f32.comp
src/
  nanovsr.cpp            主程序：ffmpeg 管道 + 模型调度 + staging 拷贝
  bench_model.cpp        纯测速（分阶段计时，带预热轮）
  model_test.cpp         逐级对拍（读 testrun_*/ 的参考 bin）
  vk_info.cpp / vk_cm_info.cpp   设备能力探测
pack_model.py            权重打包（三种布局）+ 生成参考输出
models_rt/nanovsr_1.7m/  weights_rt.bin/json（SIMT 布局）+ weights_cm.bin（张量核布局）
webui.py / webui.html    Web 面板
```

## 3. 模型结构（deploy 后）

```
feat_extract   3→64  3x3 + LeakyReLU(0.1)
forward_net    64→64 3x3 + LeakyReLU   x20（Sequential，整段作用于 feat_t + prop）
backward_net   64→64 3x3 + LeakyReLU   x20（反向帧序）
fusion        128→64 1x1
upsample1      64→256 3x3 + PixelShuffle2 + PReLU(0.0263)
upsample2      64→128 3x3 + PixelShuffle2 + PReLU(0.0393)
conv_last      32→3   3x3
out = conv_last(...) + bilinear4x(input)  → clamp(0,1) → u8
```

**时间递推**：`prop_t = Net20(feat_t + prop_{t-1})`；`prop` 跨帧延续，只在序列开头清零。
`feat_t + prop` 每帧只加一次（不是每层加）。

## 4. 已踩的坑（详见全局 knowledge base）

1. `VK_KHR_cooperative_matrix` 的 glslang 写法与规范不同：类型 5 参数（无 Scope）、
   `coopMatLoad(out m, buf[], elemOffset, stride, layout)`。→ `reflection/2026-11-17-...`
2. CPU 直接读 ReBAR 显存映射极慢（0.23 fps），必须 GPU 拷贝到 `HOST_CACHED` 系统内存。
3. `_popen` 调 ffmpeg 时命令要再包一层引号，否则 cmd 会吃掉引号。
4. 对拍参考必须 `torch.backends.cudnn.allow_tf32 = False`，否则参考本身只有 5e-4 精度。
5. 测速必须跳过首次 dispatch（驱动 JIT），否则虚高 15 倍。
6. 全局屏障：每个 dispatch 后加 compute→compute barrier，否则相邻层读到旧数据。

## 5. 还能优化什么（按收益排序，含实测分析）

**当前实测**：480p（960×544→4K）balanced 约 2.3 fps；1080p 0.44 fps（21 块 × 1.61× 重叠放大）。

### 5.1 40 个 block 只有 12 TFLOPS（最大头）

不是指令瓶颈：每层约 2.6 M 条 warp 指令，按理论吞吐只要 8.6 µs，实测 780 µs（差 90×）。
**是权重流量打到了 DRAM**：每层 1015 个 workgroup × 74 KB 权重 = 74 MB，
而激活流 33 MB/层把 32 MB 的 L2 冲掉 → 每次都要回显存取。

按收益排序的解法：
1. **每个 workgroup 覆盖更多像素**（最直接）：现在一个 warp 是 16px × 16oc，
   改成同时守 2~4 个像素组（累加器 16~32 regs），B 片段只加载一次 → 权重流量 /2~4。
   受限：shared 已被 tile 占满，要么减小 tile 高，要么把 s_out 降到 4 KB。
2. **激活全 f16**（feat/fwd/bwd/bx/bout）：激活流减半 → L2 命中率上升；
   同时时间维显存减半 → chunk 可翻倍或 tile 更大。改动面较大（要加 add/cat/convert 的 f16 变体）。
3. 权重常驻 shared：C=64 时需要 73.7 KB，放不下；按 oc 分片常驻需 4× workgroup，总流量不变。

### 5.2 1080p

能跑（0.43 fps，21 块），但重叠放大 1.61× 吃掉了大半算力。
提速仍回到 5.1；另外 overlap 96 是按 ±42 感受野算的，不能再降。

### 5.3 已完成：bconv1/bconv2 改 f16

```
静态 tile 缓冲：13.7 → 9.2 KB/px（1080p 单块 141k → 159k px）
画质：PSNR 38.68 / 39.03 dB，与 fp32 版完全一致
```

### 5.4 其它

| 项 | 现状 | 做法 |
|---|---|---|
| conv_last（C=32→3） | 6.3 ms/帧 @480p | O=3 补到 16 走张量核，或 4 像素/线程 |
| shuffle1+2 | 4.1 ms/帧 | 与上采卷积的 store 融合 |

## 6. 边界 / 已知限制

- **时间窗**：递推每 chunk 重置一次（默认 30，训练 Phase 2 的值）。chunk 越小重建越频繁，
  2-3 帧时会看到周期性亮度跳动。
- **分块**：重叠 96 覆盖接收野，实测接缝不可见；每块独立跑时间链，极端运动下块边缘的
  时间一致性略弱于整帧。tile 尺寸不影响数值（全卷积），只影响速度/显存。
- 输入上限受显存约束：960×544 用 18 块 256×256、chunk 30（约 2.6 GB），实测 1.3 fps；
  1080p 会变成 60+ 块，速度按比例下降。
- 输出固定 RGB24 → yuv420p（x264 veryfast），编码参数可改 `--crf`。
- WebUI 是单机单进程，任务状态在内存里；重启即丢。

## 7. 验收方法（踩过坑，务必照做）

1. **逐帧对齐检查**（最重要）：把输出和源逐帧对拍，看"每帧最佳匹配的偏移"是否恒为 0。
   曾经因为合成时所有帧写进同一个缓冲区，导致每 chunk 只保留最后一帧，
   而"接缝截图"用的都是同一帧的副本，看起来完美。详见
   `reflection/2026-11-17-tiled-video-frame-buffer-bug.md`。
2. **看 PSNR 分布而不是平均值**：avg 15 / max 45 的双峰 = 部分帧对、部分帧错。
3. **用有运动的片段验收**，别用静态帧。
4. 正常值参考：输出缩回原尺寸与源对比 **PSNR 36-37 dB**（faithful restoration，不是生成）。
## 8. 编码器选择（实测）

启动时按 `av1_nvenc → hevc_nvenc → libsvtav1 → libx265 → libx264` 顺序**真跑一帧 64x64**
验证（不能只看 `-encoders` 列表）。本机结果：

```
av1_nvenc / hevc_nvenc 全部失败：
  [av1_nvenc] Driver does not support the required nvenc API version. Required: 13.1 Found: 13.0
→ 自动落到 libsvtav1（CPU AV1）
```

同一个 960×544→3840×2176、48 帧：

| 编码器 | 吞吐 | 文件 |
|---|---|---|
| libsvtav1 -crf 18 -preset 8 | 2.33 fps | 10446 KB |
| libx264 -crf 18 -veryfast | 2.31 fps | 11033 KB |

**编码不占墙钟时间**：它是独立进程，和我们 GPU 推理并行，所以 AV1 的 CPU 开销被掩盖了。
想用硬件编码需要把 NVIDIA 驱动升到 ≥610（NVENC API 13.1），否则这个 ffmpeg 的所有 nvenc 都用不了。

## 9. 下一步（按观感收益排序）

1. **chunk 间时间重叠（overlap carry）**：现在 chunk 30 硬切，递推每 30 帧重置一次。
   学 Topaz 的做法，把上一段末尾的 `prop` 传给下一段（反向链同理）→ 接缝处不再"重新适应"。
   改动点：`run_dir` 的清零逻辑 + chunk 循环，纯逻辑，风险低。
2. 40 个 block 的权重流量（见 §5.1）。