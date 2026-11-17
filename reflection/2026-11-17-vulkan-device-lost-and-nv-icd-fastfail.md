# 2026-11-17 Vulkan：device lost 后 NVIDIA ICD 在 vkCreateDevice 里 __fastfail

## 报错原文

1. 第一次：`vkQueueSubmit failed: -4`（VK_ERROR_DEVICE_LOST），src/nanovsr.cpp:326
2. 从此以后运行任何选中 NVIDIA 设备的程序都变成：
   ```
   进程退出码 -1073740791 = 0xC0000409 (__fastfail)
   procdump 转储解析：ExceptionCode=0xC0000409, ExceptionInformation[0]=0x7 (PRIV_INSTRUCTION)
   ExceptionAddress=0x7ff84c6807cd → 模块 nvoglv64.dll+0x10907cd
   ```
   崩溃点在 `VK_CHECK(vkCreateDevice(...))` 内部（之前的 printf 都打出来了，后面一行没有）。

## 根因

- 报错 1 的根因是**我的代码**：双缓冲改造后 `begin(slot)` 只做了 `vkBeginCommandBuffer`，
  没有 `vkResetCommandBuffer`，第二次重录同一个 cb 是未定义行为 → GPU hang → device lost。
  修法：`wait(slot); vkResetCommandBuffer(cb, 0); vkBeginCommandBuffer(...)`。
- 报错 2 的根因**不在项目里**：那次 device lost 之后 NVIDIA 的 Vulkan ICD 进入了坏状态，
  之后任何程序（包括 40 行最小复现、HEAD 源码重编的二进制）在 `vkCreateDevice` 都直接 __fastfail；
  同一时刻选 AMD 核显一切正常；同一时刻别的 exe 有时能过、有时不能 —— 非确定性 = 驱动状态问题。
  已排除：扩展/特性链（最小 pNext=nullptr、extCount=0 仍崩）、instance apiVersion 1.1~1.4、
  本地 vulkan-1.dll、ImplicitLayers 注册表、其它 Vulkan 进程占用。

## 修法 / 下次怎么办

- 代码侧：任何"重录命令缓冲"的地方都要先 reset；双槽流水里在 `begin(slot)` 里做，一次覆盖全部调用点。
- 环境侧：`vkCreateDevice` 0xC0000409 + nvoglv64.dll 命中 = 驱动坏了，**重启机器**（非管理员无法重启驱动服务）。
  重启后先用最小程序确认 `vkCreateDevice` 能过，再跑项目，别浪费时间怀疑代码。
- 想拿核显当对照组时注意两件事：AMD 上这套 cooperative-matrix shader 输出**全黑**（只能验流程不能验数值）；
  核显显存 ~2 GB，1080p 的 12 块 tile 会 `vkQueueSubmit -2`（VK_ERROR_OUT_OF_DEVICE_MEMORY），要压小素材。
## 后续（重启后定位到的第二个、更阴的 bug）

- 现象：流水线改造后输出“几乎全黑”（值域 0..1），但 `--serial` 也黑，两边对拍 PSNR=99 dB（逐位一致）——A/B 反而骗人。
- 根因：老代码在 tile 中间有一对 `g.reset_pool(); g.begin();`（原本用来分隔两次提交）。改成单 cb 后忘了删，
  `begin()` 会 `vkResetCommandBuffer`，把刚录好的前半段（copy/f32/feat/双向链）全部丢掉，只剩后半段在一个没有输入的缓冲上跑。
- 修法：删掉第二次 `begin()`；`begin(int slot = -1)` 默认取当前 `cur`，避免中途切槽。
- 验收教训：**必须用 torch 参考（kn_in 第 5 帧 644k 应 ~45.08 dB）**，只做“改前 vs 改后”自对拍会漏掉整段丢失。
