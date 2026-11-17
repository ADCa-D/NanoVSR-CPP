# -*- coding: utf-8 -*-
"""生成 NHWC 布局的测试向量（配合 shared-memory 版 shader）。

  x.bin   [H,W,C]     NHWC
  w.bin   [O,9,C]     W[oc][ky*3+kx][c]（同一 kernel 位置的 C 连续，便于 warp 广播）
  b.bin   [O]
  ref.bin [H,W,O]     NHWC 输出
  meta.json
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
    conv = m.forward_net[0].rbr_reparam
    nw = conv.weight.detach().float().cpu()          # [O,C,3,3]
    nbc = conv.bias.detach().float().cpu()
    O, C = nw.shape[0], nw.shape[1]
    H, W = 16, 20

    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((C, H, W)).astype(np.float32) * 0.5
    x = np.ascontiguousarray(x_nchw.transpose(1, 2, 0))          # [H,W,C]
    w = np.ascontiguousarray(nw.permute(0, 2, 3, 1).reshape(O, 9, C))  # [O,9,C]
    b = nbc.numpy().astype(np.float32)

    with torch.no_grad():
        y = F.conv2d(torch.from_numpy(x_nchw)[None], nw, nbc, padding=1)
        y = F.leaky_relu(y, 0.1)[0]                              # [O,H,W]
    ref = np.ascontiguousarray(y.permute(1, 2, 0).numpy())       # [H,W,O]

    for name, arr in (("x", x), ("w", w), ("b", b), ("ref", ref)):
        arr.astype(np.float32).tofile(os.path.join(OUT, f"{name}.bin"))
    meta = dict(C=C, H=H, W=W, O=O, slope=0.1, layout="nhwc")
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"NHWC 测试向量: H={H} W={W} C={C} O={O}  "
          f"x={x.nbytes}B w={w.nbytes}B ref={ref.nbytes}B")


if __name__ == "__main__":
    main()
