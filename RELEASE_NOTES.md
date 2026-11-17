# NanoVSR-CPP v1.0.0

NanoVSR 4× 视频超分的 **C++ / Vulkan compute** 实现。不依赖 PyTorch、CUDA、Vulkan SDK。

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

代码 MIT。模型权重与架构来自 NanoVSR，遵循其原始许可。
