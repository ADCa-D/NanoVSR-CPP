# -*- coding: utf-8 -*-
"""用法: psnr.py a.png b.png  -- 同尺寸 RGB 图像 PSNR(dB) + 最大/均值绝对误差"""
import sys
import numpy as np
from PIL import Image

a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), np.float64)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), np.float64)
if a.shape != b.shape:
    print("尺寸不同", a.shape, b.shape)
    sys.exit(2)
mse = float(((a - b) ** 2).mean())
psnr = 99.0 if mse == 0 else 10.0 * np.log10(255.0 * 255.0 / mse)
d = np.abs(a - b)
print(f"PSNR={psnr:.4f} dB  MSE={mse:.4f}  maxabs={d.max():.0f}  meanabs={d.mean():.3f}")
