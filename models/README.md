# 模型文件（不放仓库）

本仓库**不包含模型权重**（原版权属于 NanoVSR 作者）。按下面三步自行生成。

## 1. 下载官方 checkpoint

从 NanoVSR 官方仓库的 Releases / 网盘下载 `nanovsr_1.7m.pth`（推荐）或
`nanovsr_644k.pth` / `nanovsr_5.4m.pth`，放到：

```
D:\nanovsr\checkpoints\
    nanovsr_644k.pth
    nanovsr_1.7m.pth
    nanovsr_5.4m.pth
```

（路径可在 `export_weights.py` 顶部的 `NANOVSR` 变量里改。）

## 2. 打包成运行时布局

```bat
python pack_model.py --tag 1.7m --out models_rt\nanovsr_1.7m --weights-only
```

会生成三个文件：

| 文件 | 内容 | 用途 |
|---|---|---|
| `weights_rt.bin` | `[O][k*k][C]` fp32 + bias | SIMT / 小通道卷积 |
| `weights_rt.json` | 层清单（C/O/k/偏移/cm_off） | host 按名字取权重 |
| `weights_cm.bin` | `[tap][C][O]` f16 分块 | 张量核卷积 |

## 3. 放到可执行文件旁边

```
nanovsr.exe
shaders\*.spv
models_rt\
    nanovsr_644k\  (weights_rt.bin, weights_rt.json, weights_cm.bin)
    nanovsr_1.7m\
    nanovsr_5.4m\
```

然后：

```bat
nanovsr.exe --in a.mp4 --out a_4x.mp4 --model-size 1.7m
```

## 说明

- 三个模型的 `num_feat` 不同（48 / 64 / 96），张量核 shader 按 `num_feat` 编译变体
  （`conv3x3_cm_{48,64,96}.spv`），构建脚本会一起生成。
- 只想要一个模型的话，下载对应权重跑第 2 步即可，其余目录可以不存在 —— 但启动时用
  `--model-size` 指定的那个必须存在。
