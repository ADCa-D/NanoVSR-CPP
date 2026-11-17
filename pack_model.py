# -*- coding: utf-8 -*-
"""把导出权重打包成 C++ 运行时布局，并生成整模型参考输出（分阶段）。

运行时布局（所有 conv 统一）：
  w   [O][kh*kw][C]  fp32   —— 与 shader 里 vec4 视图 w4[oc*9*16 + tap*16 + c4] 对齐
  bias[O]            fp32

参考输出全部用 NHWC + fp32 落盘，便于 C++ 侧逐级比对：
  x.bin        [T][H][W][3]
  feat.bin     [T][H][W][64]         feat_extract 之后
  fwd.bin      [T][T][H][W][64]      第 i 帧的第 j 个 forward 输出（j 只需最后一帧？不，全部）
  ...
为省盘，只落：feat / fwd_last / bwd_last / fused / up1 / up2 / conv_last / final。

用法：
  python pack_model.py --tag 1.7m --h 32 --w 40 --t 3 --out testrun_s
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"D:\nanovsr")
sys.path.insert(0, r"D:\nanovsr-webui")
from export_weights import build  # noqa: E402


def to_nhwc(t):                       # [N,C,H,W] -> [N,H,W,C] contiguous numpy
    return np.ascontiguousarray(t.detach().permute(0, 2, 3, 1).float().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1.7m")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--w", type=int, default=40)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--out", default="testrun_s")
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)

    # 关掉 TF32：cuDNN 默认用 TF32（10 位有效位）算 fp32 卷积，
    # 会让"参考"本身只有 5e-4 相对精度，逐级放大后完全没法对拍。
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    m, nf, nb = build(a.tag)
    m = m.to("cuda").eval()

    # ---- 1) 权重打包（注意：导出文件是 fp16，这里转 fp32 供当前 fp32 shader 用）----
    blob = bytearray()
    man = {"tag": a.tag, "num_feat": nf, "layers": []}
    convs = [("feat_extract", m.feat_extract.rbr_reparam)]
    for i in range(nb):
        convs.append((f"forward_{i}", m.forward_net[i].rbr_reparam))
    for i in range(nb):
        convs.append((f"backward_{i}", m.backward_net[i].rbr_reparam))
    convs.append(("fusion", m.fusion))
    convs.append(("up1", m.upsample1.conv))
    convs.append(("up2", m.upsample2.conv))
    convs.append(("conv_last", m.conv_last))

    for name, conv in convs:
        w = conv.weight.detach().float().cpu().numpy()      # [O,C,kh,kw]
        b = conv.bias.detach().float().cpu().numpy()
        O, C, kh, kw = w.shape
        wp = np.ascontiguousarray(w.transpose(0, 2, 3, 1).reshape(O, kh * kw, C))
        # 256 字节对齐（minStorageBufferOffsetAlignment）
        blob += b'\0' * (-len(blob) % 256)
        wo = len(blob)
        blob += wp.tobytes()
        bo = len(blob)
        blob += np.ascontiguousarray(b, dtype=np.float32).tobytes()
        man["layers"].append(dict(name=name, C=C, O=O, k=kh, w_off=wo, w_n=wp.size,
                                  b_off=bo, b_n=O))

    man["prelu"] = [float(m.upsample1.prelu.weight.detach().cpu()),
                    float(m.upsample2.prelu.weight.detach().cpu())]
    man["act_slope"] = 0.1                     # RepVGGBlock 里的 LeakyReLU(0.1)
    man["shape"] = dict(T=a.t, H=a.h, W=a.w)
    man["num_blocks"] = nb

    # ---- 1b) 张量核用的权重布局 [tap][C][O]，f16 ----
    cm_blob = bytearray()
    for name, conv in convs:
        w = conv.weight.detach().float().cpu().numpy()      # [O,C,kh,kw]
        O, C, kh, kw = w.shape
        if C % 16 != 0 or kh != 3 or O % 16 != 0:
            man["layers"][[l["name"] for l in man["layers"]].index(name)]["cm_off"] = -1
            continue
        wc = np.ascontiguousarray(w.transpose(2, 3, 1, 0).reshape(kh * kw * C, O))  # [tap*C][O]
        # 张量核友好布局：[tap][16通道块][16输出通道块][k][n]，每个 16x16 块连续
        wc = np.ascontiguousarray(
            wc.reshape(9, C // 16, 16, O // 16, 16).transpose(0, 1, 3, 2, 4))
        cm_blob += b'\0' * (-len(cm_blob) % 256)
        off = len(cm_blob)
        cm_blob += wc.astype(np.float16).tobytes()
        man["layers"][[l["name"] for l in man["layers"]].index(name)]["cm_off"] = off

    with open(os.path.join(out, "weights_cm.bin"), "wb") as f:
        f.write(bytes(cm_blob))
    print(f"张量核权重: {len(cm_blob)/1e6:.2f} MB")

    with open(os.path.join(out, "weights_rt.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(out, "weights_rt.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"权重: {len(blob)/1e6:.2f} MB -> {out}/weights_rt.bin  ({len(convs)} 层)")

    if a.weights_only:
        print("只导出权重，跳过参考输出")
        return

    # ---- 2) 参考输出（用同一份 fp32 权重重灌一次，确保与运行时完全同源）----
    torch.manual_seed(a.seed)
    x = torch.rand(a.t, 3, a.h, a.w, device="cuda")

    g = {}
    with torch.inference_mode():
        feats = m.feat_extract(x).view(1, a.t, nf, a.h, a.w)       # [1,T,F,H,W]
        g["feat"] = to_nhwc(feats[0])

        fwd, prop = [], torch.zeros_like(feats[:, 0])
        for i in range(a.t):
            prop = m.forward_net(feats[:, i] + prop)
            fwd.append(prop)
        g["fwd_last"] = to_nhwc(fwd[-1])
        g["add1"] = to_nhwc(feats[:, 1] + fwd[0])
        g["fwd0"] = to_nhwc(fwd[0])

        bwd, prop = [None] * a.t, torch.zeros_like(feats[:, 0])
        for i in range(a.t - 1, -1, -1):
            prop = m.backward_net(feats[:, i] + prop)
            bwd[i] = prop
        g["bwd_last"] = to_nhwc(bwd[0])          # i=0 是反向链的最后一次输出

        u1l, u2l, cl, fin = [], [], [], []
        for i in range(a.t):
            f = m.fusion(torch.cat([fwd[i], bwd[i]], 1))
            u1 = m.upsample1(f)
            u2 = m.upsample2(u1)
            y = m.conv_last(u2)
            base = F.interpolate(x[i:i + 1], scale_factor=4, mode="bilinear",
                                 align_corners=False)
            u1l.append(u1)
            u2l.append(u2)
            cl.append(y)
            fin.append((y + base).clamp(0, 1))

        g["blk0"] = to_nhwc(m.forward_net[0](feats[:, 0]))
        cur = feats[:, 0]
        for b in range(3):
            cur = m.forward_net[b](cur)
            g[f"chain{b + 1}"] = to_nhwc(cur)
        g["fused0"] = to_nhwc(m.fusion(torch.cat([fwd[-1], bwd[-1]], 1)))
        g["up1_0"] = to_nhwc(u1l[-1])
        g["up2_0"] = to_nhwc(u2l[-1])
        g["conv_last_0"] = to_nhwc(cl[-1])
        g["x"] = to_nhwc(x)
        final = torch.stack([f[0] for f in fin], 0)                 # [T,3,4H,4W]
        g["final"] = to_nhwc(final)
        g["final_u8"] = (final.clamp(0, 1).permute(0, 2, 3, 1).mul(255).round()
                         .byte().cpu().numpy())

    for k, v in g.items():
        v.tofile(os.path.join(out, f"{k}.bin"))
    print("参考输出: " + ", ".join(f"{k}{list(v.shape)}" for k, v in g.items()))


if __name__ == "__main__":
    main()

