# NanoVSR-CPP v1.0.0

NanoVSR 4× 视频超分的 **C++ / Vulkan compute** 实现。不依赖 PyTorch、CUDA、Vulkan SDK。

---

## 2026-11-17 重新打包

- **双缓冲流水线**：打包 / 推理 / 合成交错，1080p 多 tile 场景端到端 +8%（GPU 已 88% 饱和，剩下的瓶颈是算力）。
- **显存**：中间缓冲全部改为 `DEVICE_LOCAL`（不再走 BAR 映射的系统内存）。
- **CPU 侧**：常驻线程池、常驻描述符集缓存、层索引预解析、解码与写盘各一个后台线程；1080p 下 CPU 侧占比降到 1%。
- **音轨 / 字幕**：Web 面板输出自动回封装源文件的音轨与字幕（`-c:a copy` / `-c:s copy`），历史记录可单条删除 / 清空并持久化。
- **权重目录回退**：`models_rt/` 与 `models/models_rt/` 都认，请求的档位没装时自动用已装的一档。
- 移除早期遗留的 `vk_conv_test`（单层 SIMT 对照，输入数据早已不在）与两个无源 `.spv`。

---

## ⚠️ 运行前必须准备权重

**本包不附带模型权重**（版权属于 NanoVSR 作者）。第一次运行会提示：

```
找不到模型权重: models_rt/nanovsr_1.7m\weights_rt.{bin,json}
```

三步解决（详见包内 `models/README.md`）：

1. 下载官方 checkpoint（`nanovsr_1.7m.pth` 等）
2. `python pack_model.py --tag 1.7m --out models_rt\nanovsr_1.7m --weights-only`
3. 确认 `models_rt\nanovsr_1.7m\` 下有三个文件（`weights_rt.bin`、`weights_rt.json`、`weights_cm.bin`）

---

## 快速开始

```bat
:: 命令行
nanovsr.exe --in a.mp4 --out a_4x.mp4

:: 或图形界面（需要 python）
start_webui.bat          :: 自动打开 http://127.0.0.1:7861
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model-size` | 1.7m | 644k / 1.7m / 5.4m |
| `--preset` | balanced | preview(快) / balanced / quality(编码保真) |
| `--chunk N` | 30 | 时间窗，递推每 chunk 重置一次 |
| `--max-out` | 4k | 输出像素上限：4k / 8k |
| `--budget GB` | 驱动建议×0.8 | 显存预算，决定分块大小 |
| `--codec` | auto | 自动选 libsvtav1 → libx265 → libx264 |

## 模型对比（同一帧，480p → 3416×1920，GT 为原 4K）

| 模型 | 参数量 | 与 PyTorch 参考差 | 速度 | 说明 |
|---|---|---|---|---|
| 644k | 0.7M | 44.34 dB | 最快 | 细节略少 |
| 1.7m | 1.9M | 42.35 dB | 默认 | 锐度/速度平衡 |
| 5.4m | 6.1M | 43.01 dB | ~1/1.5 | 最锐 |

> 这里的 dB 是"与 PyTorch 输出的接近程度"，**不是画质排名**——模型越锐，越偏离原图，这个数字反而可能更低。

## 性能（RTX 5060 Laptop 8 GB）

| 输入 → 输出 | 速度 |
|---|---|
| 960×544 → 3840×2176 | 2.3 fps（含 AV1 编码） |
| 1920×1080 → 4096×2304 | ~0.45 fps（模型 4× 跑满后缩到交付尺寸） |

## 实现要点

- **张量核卷积**：`VK_KHR_cooperative_matrix`，16×16×16 f16 / f32 累加；不做 im2col，
  A 片段直接从 shared 的 halo tile 按 tap 偏移读取。
- **空间分块**：按显存账本（静态 9.2 KB/px + 每帧 0.89 KB/px）搜索总处理量最小的 tile
  形状，块间重叠 96（覆盖 40 层卷积 ±42 的接收野），只合成中心区。
- **超采样**：输出超过 4K 时，模型仍按 4× 跑满，由编码端降采样（保锐度）。

## 已知限制

- 8K 输出需要 `--max-out 8k --codec libx265`（NVENC 在驱动 NVENC API < 13.1 时不可用）
- chunk 边界会重置时间递推
- 1.7m 在近奈奎斯特条纹（细格纹、密纹理）上可能出现摩尔纹——已确认是模型行为
  （与 PyTorch 输出一致）

## 许可

本仓库代码为 MIT。模型架构与权重来自 NanoVSR，同为 **MIT License**（© NanoVSR authors），因此可直接使用与再分发，保留版权声明即可。

## 2026-11-18 评审修补 + 重打包

按两份外部评审（Gemini / 千问）逐条核对，修掉 18 处确认问题，其余 7 条记为误报或明确不做（理由见 `HANDOFF_PERF.md` 第四轮）：

- 正确性：`read_file`/`cm_off` 的 32 位截断、手动 `--tile` 超帧尺寸的负坐标与超预算、清单缺 key 的越界偏移、编码命令行截断、VFR 帧率、末尾 tile 接缝按实际重叠区中线切、`feat_extract` barrier 开关传反。
- 性能：权重改住显存（DEVICE_LOCAL + staging 上传）、描述符池数量修正、`RowPool` 去广播风暴、单次提交 512 dispatch 上限防 TDR。
- 体验：`--codec hevc/av1` 别名生效、图片模式不再按 chunk 30 分配显存、写失败早停、`--chunk` 截断提示、进度行不再显示假的满值 tile 计数。

实测：644k / 5.4m 对 torch 仍 45.0817 / 45.0842 dB；1.7m clip_1080 --tile 200x144 十帧 14.72 s（改前 14.79~14.84 s）。

## 2026-11-18 fusion 改走张量核（−16~20%）

- 新增 `shaders/conv1x1_cm.comp`：1×1 卷积的张量核内核，按 16 通道流式载入 shared（恒定 16 KB），
  C=192 的 5.4m fusion 也能跑（3×3 版要 72 KB，超设备上限）。
- `run_convg` 对 `k==1 && cm_off>=0` 的层走新内核；fusion 原本落在标量 `conv_gen` 上，
  虽然只占 0.56% 的 MAC，却吃掉 20.6% 的时间。
- `pack_model.py` / `models_rt/*` 重新打包（`weights_rt.bin` 逐字节不变）。
- 实测 1080p→4K 十帧：644k 2.46→2.07 s、1.7m 4.22→3.36 s、5.4m 13.82→11.50 s；
  对 torch 仍是 45.08 dB；`model_test` 三档 `final_u8` 最大差 1 级。
- 吞吐：480p 输入 644k 7.7 / 1.7m 5.1 / 5.4m 1.3 fps；300p 输入 644k 26.6 / 1.7m 16.7 fps。

## 2026-11-18 边界审计（6 条）

- 修：--tile 小于感受野重叠时步长退化成 1 像素（98x62 的图被切成 2077 块、慢 300 倍）；现在强制 tile >= 重叠+16，太小则忽略并退回自动分块。
- 修：输出目录不存在、ffmpeg 不可用、一帧都没解码到，以前都返回 0；现在返回 1 并打印原因。
- 修：带 rotate=90/270 显示矩阵的手机视频会输出斜切花屏（ffprobe 报旋转前尺寸、ffmpeg 解码已转正）；现在按转正后的尺寸处理。
- 修：--gpu N 越界改为打印提示，不再静默用默认设备。
- 新增：分段落盘（默认开）—— 输出切成 `<out>.parts/*.mkv`，每段写完用 ffprobe 数包校验，结束时 `-c copy` 拼接；中断后重跑同命令会跳过已完成的段、只重算坏段。`--no-seg` 关掉，`--seg-chunks K` 调段大小（默认 4 个 chunk），`--keep-parts` 保留分段目录。
- 修：权重/层清单越界检查改成减法判断，避免 size_t 回绕漏检。
- 新增：shaders/conv1x1_cm.comp（1x1 卷积走张量核的基础版本）+ 内核 8 warp 重写，端到端 1.7m 3.47 fps / 5.4m 1.08 fps（对比上一版 +44% / +50%）。
