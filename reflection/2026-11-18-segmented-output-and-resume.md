# 2026-11-18 分段落盘（EOF 校验 / 回退重来 / 断点续跑）踩的坑

## 1. concat 的 list 必须写绝对路径

现象：分段都写好了、校验也过了，拼接这一步报

```
[concat @ ...] Impossible to open 's1.mp4.parts/s1.mp4.parts/00000.mkv'
```

根因：ffmpeg 的 concat demuxer 解析 list 里的相对路径时，基准目录是 **list 文件自己所在的目录**，
所以 `work/00000.mkv`（相对 cwd）被拼到 `s1.mp4.parts/` 后面，变成双份。

修法：写 list 时用 `std::filesystem::absolute(seg_file(i)).string()`；单引号在 Windows 路径里不会出现，但仍按 `'` -> `'\''` 规则转义（跨平台兜底）。

## 2. 每段重开编码器不是免费的

按 chunk 一段（`chunk 8`、AV1、1080p）实测比不分段慢 37%：libsvtav1 每次启动固定开销 ≈ 0.61 s，
x264 ≈ 0.10 s。GPU 快的时候这个开销被掩盖，GPU 慢或者 chunk 小的时候它直接变成瓶颈（`chunk 2` 时 +128%）。

修法：`--seg-chunks K`（默认 4），一个分段跨 K 个 chunk，编码器跨 chunk **常驻**（段开始 popen、段内连续 fwrite、段末 pclose 校验）。
代价是失去「运行内原地重编该段」的能力（那段的 GPU 结果已经释放），改成把坏段留在磁盘上、提示重跑同命令。

## 3. 校验必须在 pclose 之后

最初想在写线程里边写边数包 —— 不行：ffmpeg 还在写容器头/簇，未 close 的文件数包不准。
正确顺序是 `pclose_w(f) -> probe_packets(seg)`，并且要区分两种失败：退出码非 0（编码器崩/磁盘满）、退出码 0 但包数不对（写了一半被杀）。

## 4. 续跑判据要按「段」而不是「chunk」

加了 `--seg-chunks` 之后，`done_chunks` 语义从 chunk 数变成段数，跳过条件必须写成
`chunk_idx / seg_chunks < done_chunks`，否则会跳过数量级的错误（实测表现为少帧或多算）。

## 5. 只信「整 chunk 边界」的段

被硬杀时正在写的那段，大小可能是任意字节。判据用 `got > 0 && got % chunk == 0`：
- 0 字节 / 只有容器头 → probe 返回 0 或 -1 → 删掉重算；
- 半截簇 → 包数不是 chunk 的整数倍 → 删掉重算；
- 正好停在 chunk 边界 → 认为完整（此时文件本身是合法可播的，最坏只是最后一段短一点）。
另外最后还有一道总账：所有段的包数之和必须等于已处理帧数（`total`），不等就报错并保留分段目录，
否则「少一段还静默成功」这种坑会一直藏着。

## 6. 实测结论（RTX 5060 Laptop）

| 项 | 结果 |
|---|---|
| 分段 vs `--no-seg`（644k kn_in 6 帧 crf0） | 第 5 帧 bit-identical（PSNR 99 dB，maxabs 0） |
| 默认配置性能（chunk 30） | 1.85 s vs 1.85 s，无损 |
| chunk 8 + AV1 + `--seg-chunks 4` | 4.90 s vs 4.93 s，无损 |
| 硬杀后续跑 | 跳过完整段、补算残段，帧数正确 |