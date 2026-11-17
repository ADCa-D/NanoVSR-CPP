# 附录：644k / 5.4m 的张量核路径（下次开工第一件事）

## 现状

| 模型 | C | block 的 O | 与 torch 的 PSNR |
|---|---|---|---|
| 1.7m | 64 | 64 | **48.4 dB** ✅ 唯一验证通过 |
| 644k | 48 | 48 | 32.4 dB ❌ |
| 5.4m | 96 | 96 | 32.3 dB ❌ |

关键线索：**改了三处代码后 5.4m 的输出逐位不变（32.322660）** —— 说明它根本没走新代码，
而是落到了 `conv3x3_c64.comp`（SIMT，C 写死 64）的兜底路径。

已试过但**无效**的修补（记录以免重做）：

1. `tap * 4` → `tap * (C / 16)`（权重块索引）
2. shared tile 通道跨度补齐到 32 half（`CPAD`）
3. `cat2` 里硬编码的 64 → `(int)F`
4. 权重 O 补齐到 64 的倍数 + 输出写 guard + 权重 stride 单独传参

## 根因判断

“一套 shader 用运行时变量同时支持 C=48/64/96 和 O 任意”，把
shared 尺寸、coopMatLoad stride、权重块布局、输出写索引全变成运行时计算，
任何一处不匹配都表现为“结构对但整体偏软”（32 dB），且不会崩、不会出噪点。

## 正确做法：编译期定死 (C, O)

一个 `(C, O)` 组合一个 SPIR-V，运行时零分支：

```glsl
#ifndef C_OVERRIDE
#define C_OVERRIDE 64
#endif
#ifndef O_OVERRIDE
#define O_OVERRIDE 64
#endif
const int C = C_OVERRIDE;
const int O = O_OVERRIDE;      // 不再有 p.ON、不再有补齐、不再有 WSTRIDE
```

需要的组合：

```
blocks:  (48,48)   (64,64)   (96,96)
up1:     (48,192)  (64,256)  (96,384)
up2:     (48,128)  (64,128)  (96,128)
```

每个再加一个 f16 输出变体（up1/up2 用）→ 共 18 个 SPIR-V，构建多花 ~2 秒。

## 落地顺序

1. shader 改成双常量，删掉 `p.ON` / `CPAD` / `WSTRIDE` 相关逻辑
2. `pack_model.py` 去掉输出通道补齐（O 与编译期一致，不需要 pad）
3. `build.bat` 生成 9（+f16 6）个变体，命名 `conv3x3_cm_{C}_{O}.spv`
4. host：`cm_path = "shaders/conv3x3_cm_" + F + "_" + O + ".spv"`，缺失就报错而不是静默回退
5. 逐个对拍：先 (64,64) 确认 1.7m 不回退到 48.4 dB，再验 (48,48)、(96,96)

**重要**：第 4 步里“缺失就报错”很关键 —— 现在的静默回退让 644k/5.4m 用了错误的
C=64 内核却看不出来。宁可启动失败，也不要给出错误的画面。

## 调试辅助（建议顺手加）

```cpp
std::printf("%s: cm_off=%d C=%d O=%d -> %s\n", name.c_str(), L.cm_off, L.C, O,
            use_cm ? "CM" : "SIMT");
```

逐层打印能一眼看出某层是否落到了 SIMT 兜底。
