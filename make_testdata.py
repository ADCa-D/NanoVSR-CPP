# -*- coding: utf-8 -*-
"""给 C++/Vulkan 端造单层卷积的测试向量（fp32 原始文件，避免解析 pth）。

输出到 testdata/：
  x.bin       [C,H,W]        fp32
  w.bin       [O,C,3,3]      fp32
  b.bin       [O]            fp32
  ref.bin     [O,H,W]        fp32（PyTorch F.conv2d + 可选 leaky）
  meta.json   C/H/W/O/slope
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"D:\nanovsr")
from export_weights import build  # noqa: E402

OUT = os.path.join(HERE, "testdata")


def main():
    os.makedirs(OUT, exist_ok=True)
    m, nf, nb = build("1.7m")
    conv = m.forward_net[0].rbr_reparam          # 已 reparameterize 的 3×3
    w = conv.weight.detach().float().cpu().numpy()
    b = conv.bias.detach().float().cpu().numpy()
    O, C, kh, kw = w.shape
    H, W = 16, 20
    rng = np.random.default_rng(0)
    x = rng.standard_normal((C, H, W)).astype(np.float32) * 0.5

    with torch.no_grad():
        y = F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w),
                     torch.from_numpy(b), padding=1)
        y = F.leaky_relu(y, 0.1)[0].numpy()

    for name, arr in (("x", x), ("w", w), ("b", b), ("ref", y)):
        arr.astype(np.float32).tofile(os.path.join(OUT, f"{name}.bin"))
    meta = dict(C=C, H=H, W=W, O=O, kh=kh, kw=kw, slope=0.1)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"测试向量: C={C} H={H} W={W} O={O}  "
          f"x/w/b/ref = {x.nbytes}B/{w.nbytes}B/{b.nbytes}B/{y.nbytes}B")
    print(f"输出到 {OUT}")


if __name__ == "__main__":
    main()
