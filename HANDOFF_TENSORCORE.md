# 附录：644k / 5.4m 的张量核路径

## 结论（2026-11-17 已解决）

| 模型 | C | 修复前 | 修复后 |
|---|---|---|---|
| 1.7m | 64 | 48.4 dB ✅ | 45.0873 dB |
| 644k | 48 | 32.4 dB ❌ | **45.0817 dB** |
| 5.4m | 96 | 32.3 dB ❌ | **45.0842 dB** |

同一测法（kn_in.mp4 854x480 前 6 帧、取第 5 帧、h264 crf 0、对 torch 参考）。
三档差值 < 0.01 dB，**说明张量核 shader 本身对 C=48/64/96 都是对的**，不需要
按模型各写一套 shader，也不需要把 (C,O) 都做成编译期常量。

45 dB 与更早那个 48.4 dB 的差异来自测量链路，不是精度问题：编码器（crf 18 时
三档都掉到 42.4 dB）+ f16 权重/累加。

## 真正的根因：host 端把 64 写死了（不是 shader）

src/nanovsr.cpp 里有三处硬编码 64，只有 num_feat==64 的 1.7m 恰好正确：

1. **src/nanovsr.cpp:981**（原 976）每个 block 的输出通道数：
   `run_conv64(name, *src, 0, *scratch[w], 0, H, W, 64, act)` —— 应该是 `(int)F`。
   644k 传 64 而实际 O=48 → shader 里 `p.ON=64`，输出按 64 通道跨步写，
   下一层按 48 通道读 → 全盘错位（"结构对但整体偏软"就是这个）。
   5.4m 更糟：`p.k=(64+63)/64*64=64`（应为 128），连权重块 stride 都错了。
2. **src/nanovsr.cpp:1021**（原 1016）up1 之后的 shuffle：`p2.C_ = 64` → `(int)F`。
   644k 输入是 4C=192 通道，按 256 读 → 又一处错位。
3. **shaders/shuffle_prelu.comp:32** `shared float s[TILEX*4*64]` 而载入上界是
   `TILEX*4*C`：C=96 时写 6144 float 进 4096 float 的数组 → shared 越界。
   已改成 `TILEX*4*96`（24 KB，仍在 48 KB 限制内）。

顺带清理：
- `run_conv64` 的非 CM 兜底原来无条件用 `conv3x3_c64.spv`（C 硬编码 64）；
  现在 `F != 64` 时改走通用的 `conv_gen.spv`（按 `p.C_` 循环，正确）。
- 缺 `conv3x3_cm_{C}.spv` / `_f16` 变体时**直接启动失败**，不再静默回退到 C=64 内核。
- 删掉 "644k/5.4m 仅供实验" 的过时警告。

## 工具现状（已泛化到三档）

- src/model_test.cpp 与 src/bench_model.cpp 不再写死 64：通道数全部取自
  weights_rt.json 的 `num_feat` / `num_blocks`（老清单没有 num_blocks 时按
  `forward_*` 层名兜底），即 `F=num_feat`、`up1O=up1.O`、`up2O=up2.O`、
  `clC=conv_last.C`；CM 变体按 `conv3x3_cm_{C}.spv` 选（C=64 仍是 conv3x3_cm.spv）。
- build.bat 现在也编译 model_test.exe（之前**没有**这一项，旧 exe 是手工产物）。
- 生成对拍数据用 pack_model.py：
  `python pack_model.py --tag 644k --h 32 --w 40 --t 3 --out testrun_644k`
  （5.4m 同理，输出 testrun_54m；testrun_s 是 1.7m 的老清单，正好验证兜底路径）。
- 三档实测（H=32 W=40 T=3）：`final_u8` 最大差都只有 1 级、0 个通道超 1 级。
  逐层的 maxdiff 绝对值看着大（5.4m 到 0.25），但相对幅值只有 2~7e-3，是 fp16 累积的正常量级；
  **验收看 final_u8，别用绝对容差 2e-2 判**（compare() 现在同时接受相对 5e-3）。
- 计时（同上小分辨率，只看比例）：1.7m 2.13、644k 1.51、5.4m 3.68 ms/帧。
- 端到端对拍配方：

```
python torch_check.py 644k kn_in.mp4 ref644_5.png 5
./nanovsr.exe --in kn_in.mp4 --model-size 644k --limit 6 --crf 0 --codec h264 --out v644.mp4
ffmpeg -i v644.mp4 -vf "select=eq(n\,5)" -frames:v 1 out644_5.png
python psnr.py ref644_5.png out644_5.png
```

  （psnr.py 就在仓库根目录）

## 如果以后还要动这块

- 权重布局 `[tap][cc][ocg][k][n]`（pack_model.py 的
  `reshape(9,C//16,16,O//16,16).transpose(0,1,3,2,4)`）与 shader 的
  `((tap*(C/16)+cc)*(k>>4)+(oc0>>4))*256` 是自洽的，实测三档步长都对得上，别乱改。
- `p.ON` 必须是**真实**输出通道数，`p.k` 是补齐到 64 倍数的权重跨度 —— 二者不能混用。
