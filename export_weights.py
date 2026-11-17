# -*- coding: utf-8 -*-
"""把 NanoVSR 导出成 Vulkan 端能直接吃的格式。

按 inference 时的真实结构导出（`switch_to_deploy()` 之后）：
    feat_extract.rbr_reparam      3x3  3 -> F
    forward_net.{i}.rbr_reparam   3x3  F -> F   (i = 0..N-1)
    backward_net.{i}.rbr_reparam  3x3  F -> F
    fusion                        1x1  2F -> F
    upsample1.conv                3x3  F -> 4F   (+pixel_shuffle 2x + PReLU)
    upsample2.conv                3x3  F -> 4*32 (+pixel_shuffle 2x + PReLU)
    conv_last                     3x3  32 -> 3

输出：
    nanovsr_<tag>.json   每层的名字/形状/文件偏移/类型
    nanovsr_<tag>.bin    紧凑排列的 fp16 权重（[out, in, kh, kw] 顺序）+ fp32 bias
"""
import json
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NANOVSR = r"D:\nanovsr"
sys.path.insert(0, NANOVSR)

from models.nanovsr import NanoVSR  # noqa: E402

CKPTS = {"644k": "nanovsr_644k.pth", "1.7m": "nanovsr_1.7m.pth", "5.4m": "nanovsr_5.4m.pth"}


def build(tag):
    """不依赖 utils.py（它顶层 import cv2，便携包里没有）。"""
    path = os.path.join(NANOVSR, "checkpoints", CKPTS[tag])
    sd = torch.load(path, map_location="cpu")
    for k in ("params_ema", "params", "model_state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    nf = nb = None
    for k, v in sd.items():
        if "feat_extract" in k and "rbr_dense" in k and k.endswith("weight"):
            nf = int(v.shape[0])
        if k.startswith("forward_net.") and k.split(".")[1].isdigit():
            nb = max(nb or 0, int(k.split(".")[1]) + 1)
    m = NanoVSR(num_feat=nf or 32, num_blocks=nb or 7)
    m.load_state_dict(sd, strict=False)
    m.switch_to_deploy()             # 3 分支塌成单个 3×3 卷积
    m.eval()
    return m, nf, nb


def layers_of(m):
    """按执行顺序列出 (name, conv_module, kind)。"""
    out = [("feat_extract", m.feat_extract.rbr_reparam, "conv3x3_relu")]
    for i, b in enumerate(m.forward_net):
        out.append((f"forward_{i}", b.rbr_reparam, "conv3x3_relu"))
    for i, b in enumerate(m.backward_net):
        out.append((f"backward_{i}", b.rbr_reparam, "conv3x3_relu"))
    out.append(("fusion", m.fusion, "conv1x1"))
    out.append(("up1", m.upsample1.conv, "conv3x3_shuffle_prelu"))
    out.append(("up2", m.upsample2.conv, "conv3x3_shuffle_prelu"))
    out.append(("conv_last", m.conv_last, "conv3x3"))
    # PReLU 的斜率是**可学习参数**（默认 0.25，训练后会变），必须一起导出
    out.append(("prelu1_w", m.upsample1.prelu.weight, "param"))
    out.append(("prelu2_w", m.upsample2.prelu.weight, "param"))
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "1.7m"
    outdir = os.path.join(HERE, "models")
    os.makedirs(outdir, exist_ok=True)
    m, nf, nb = build(tag)

    manifest = {"tag": tag, "scale": 4, "num_feat": nf, "num_blocks": nb, "layers": []}
    blob = bytearray()
    for name, mod, kind in layers_of(m):
        if kind == "param":                     # PReLU 斜率，fp32 直存
            v = mod.detach().float().cpu().contiguous()
            raw = v.numpy().tobytes()
            off = len(blob)
            blob += raw
            manifest["layers"].append(dict(
                name=name, kind=kind, out_ch=int(v.numel()), in_ch=0, kh=1, kw=1,
                w_off=off, w_bytes=len(raw), b_off=off, b_bytes=0))
            continue
        w = mod.weight.detach().float().cpu().contiguous()      # [out,in,kh,kw]
        b = mod.bias.detach().float().cpu().contiguous()        # [out]
        w16 = w.half().numpy().tobytes()
        b32 = b.numpy().tobytes()
        off_w, off_b = len(blob), len(blob) + len(w16)
        blob += w16 + b32
        manifest["layers"].append(dict(
            name=name, kind=kind,
            out_ch=int(w.shape[0]), in_ch=int(w.shape[1]),
            kh=int(w.shape[2]), kw=int(w.shape[3]),
            w_off=off_w, w_bytes=len(w16),
            b_off=off_b, b_bytes=len(b32)))

    bin_path = os.path.join(outdir, f"nanovsr_{tag}.bin")
    json_path = os.path.join(outdir, f"nanovsr_{tag}.json")
    with open(bin_path, "wb") as f:
        f.write(blob)
    manifest["total_bytes"] = len(blob)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"{tag}: {nf}ch / {nb}块 → {len(manifest['layers'])} 层")
    print(f"  {bin_path}  {len(blob)/2**20:.2f} MB")
    print(f"  {json_path}")
    for L in manifest["layers"][:3] + [manifest["layers"][-1]]:
        print(f"   {L['name']:14s} {L['kind']:22s} {L['in_ch']:3d}->{L['out_ch']:3d} "
              f"k={L['kh']}x{L['kw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
