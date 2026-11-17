# -*- coding: utf-8 -*-
"""给定 tag / 输入视频 / 输出 PNG，跑 torch 参考。用法: torch_check.py 644k in.mp4 out.png [frame]"""
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\nanovsr-cpp")
sys.path.insert(0, r"D:\nanovsr")
from export_weights import build  # noqa: E402

FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
tag, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
FR = int(sys.argv[4]) if len(sys.argv) > 4 else 5

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

probe = subprocess.run([FF, "-v", "error", "-i", src, "-frames:v", str(FR + 1),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True).stdout
m, nf, nb = build(tag)
m = m.cuda().eval()
# 需要知道分辨率：从字节数反推（rgb24）
px = len(probe) // (3 * (FR + 1))
H = int(round((px / (16 / 9)) ** 0.5))
while px % H:
    H -= 1
W = px // H
print(f"{tag}: feat={nf} blocks={nb} size={W}x{H} frames={FR+1}")

x = np.frombuffer(probe, np.uint8).reshape(FR + 1, H, W, 3).astype(np.float32) / 255.0
xt = torch.from_numpy(x).permute(0, 3, 1, 2)[None].cuda()
with torch.inference_mode():
    y = m(xt)
img = (y[0, FR].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                      "-s", f"{W*4}x{H*4}", "-i", "-", "-frames:v", "1", out], stdin=subprocess.PIPE)
p.communicate(img.tobytes())
print("wrote", out)
