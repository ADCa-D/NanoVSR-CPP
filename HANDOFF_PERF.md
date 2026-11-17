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
