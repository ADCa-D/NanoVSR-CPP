# nanovsr-cpp

NanoVSR 4× 视频超分的 **C++ / Vulkan compute** 实现 —— 不依赖 PyTorch、不依赖 CUDA、不依赖 Vulkan SDK。

模型是 [NanoVSR](https://github.com/) 的 1.7M / 5.4M / 644K 变体（双向循环 + 全卷积），
本仓库把它重写成纯 C++ 推理引擎：GLSL 计算着色器 + 手写 Vulkan 宿主 + 张量核卷积。

## 状态

| 项 | 结果 |
|---|---|
| 数值 | 与 PyTorch 逐级对拍：单层 maxdiff 5e-6，整模型最终 8bit 输出**最大差 1 级、0 通道越界** |
| 速度（480p→1080p） | 2.3 fps（含 AV1 编码），纯推理 52.9 ms/帧 |
| 速度（1080p→4K） | ~0.45 fps（模型 4× 跑满后缩到交付尺寸） |
| 依赖 | MSVC + 系统 `vulkan-1.dll` + glslang（12 MB 单文件） |

## 构建

```bat
:: 需要 MSVC 2022 BuildTools；Vulkan 头文件与 glslang 通过环境变量指定
set VKROOT=D:\path\to\vk        :: 内含 sdk\Vulkan-Headers-main\include 与 glslang\bin\glslang.exe
build.bat
```

产物：`nanovsr.exe`（主程序）、`bench_model.exe`（测速）、`model_test.exe`（对拍）。

`vulkan-1.lib` 不需要安装 SDK —— 从系统 `vulkan-1.dll` 用
`dumpbin /exports` + `lib /def` 现场生成即可。

## 权重

模型权重**不在仓库里**（原版权属于 NanoVSR 作者）。自行下载官方 checkpoint 后：

```bat
python pack_model.py --tag 1.7m --out models_rt\nanovsr_1.7m --weights-only
```

会生成 `weights_rt.bin/json`（SIMT/小通道用）与 `weights_cm.bin`（张量核分块布局，f16）。

## 用法

```bat
:: 视频超分
nanovsr.exe --in a.mp4 --out a_4x.mp4

:: 三档 preset × 三种模型
nanovsr.exe --in a.mp4 --out b.mp4 --preset preview|balanced|quality
nanovsr.exe --in a.mp4 --out b.mp4 --model-size 644k|1.7m|5.4m

:: Web 面板（纯标准库，无 pip 依赖）
python webui.py --port 7861
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--preset` | balanced | preview(chunk8) / balanced(chunk30) / quality(crf14) |
| `--model-size` | 1.7m | 644k / 1.7m / 5.4m |
| `--chunk N` | 30 | 时间窗（递推每 chunk 重置一次） |
| `--max-out` | 4k | 输出像素上限：4k(8.9Mp) / 8k(33.2Mp) |
| `--budget GB` | 驱动建议×0.8 | 显存预算，决定分块大小 |
| `--tile WxH` | 自动 | 手动指定空间分块 |
| `--codec` | auto | auto / libsvtav1 / libx265 / libx264 |
| `--downscale` | lanczos | 预缩放算法（lanczos/swscale 或 libplacebo 核） |

## 实现要点

- **张量核卷积**：`VK_KHR_cooperative_matrix`，16×16×16 f16、f32 累加。
  不做 im2col —— A 片段直接从 shared 中的 halo tile 按 tap 偏移读取。
  按输出通道切 warp 使每层权重流量从 601 MB 降到 150 MB。
- **空间分块**：按显存账本（每像素静态 9.2 KB + 每帧 0.89 KB）搜索总处理量最小的
  tile 形状，块间重叠 96（覆盖 40 层卷积 ±42 的接收野），只合成中心区。
- **超分链路**：feat → 20 层前向 → 20 层反向（逐帧递推）→ fusion → 两级 PixelShuffle
  → conv_last → 4× 双线性残差 → u8。
- **视频管道**：ffmpeg 解码 → GPU → 推理 → 打包 u8 → 编码；CPU/GPU 之间走
  `HOST_CACHED` staging（ReBAR 显存直接读会掉到 0.23 fps）。

## 已知限制

- 1080p 输入在 4K 上限下需要 21 个 tile，约 0.45 fps。
- 8K 输出需要 `--max-out 8k --codec libx265`（NVENC 因驱动 NVENC API 13.0 < 13.1 不可用）。
- 1.7m 在近奈奎斯特条纹（如细纹理衣领）上会出现摩尔纹 —— 已确认是模型行为
  （与 torch 输出一致，PSNR 48.4 dB）。
- chunk 边界会重置时间递推。

## 许可

代码 MIT。模型权重与架构来自 NanoVSR，遵循其原始许可。
