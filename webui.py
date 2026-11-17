# -*- coding: utf-8 -*-
"""NanoVSR 4x 超分 WebUI —— 纯标准库实现，底层调用 nanovsr.exe。

    python webui.py                 # 默认 http://127.0.0.1:7861
    python webui.py --port 8000

依赖：本目录下的 nanovsr.exe、shaders/*.spv、models_rt/nanovsr_1.7m/、
      ffmpeg（默认 D:\\ffmpeg-9.0.1\\ffmpeg.exe，可用 --ffmpeg 指定）。

特性：
  * 音轨 / 字幕轨原样保留（-c copy 回封装，有字幕则自动改 mkv）
  * 历史记录持久化到 work/history.json，可单条删除、可清空
  * chunk / tile / crf / 限帧 全部可手调，留空则跟随档位
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
HIST_PATH = os.path.join(WORK, "history.json")
JOBS = {}          # id -> 运行中的任务
HIST = []          # 已完成记录，最新在最前
LOCK = threading.Lock()
ARGS = None

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".flv", ".wmv")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
HIST_MAX = 60


# ---------------------------------------------------------------- 历史记录
def hist_load():
    global HIST
    try:
        with open(HIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        HIST = data if isinstance(data, list) else []
    except Exception:
        HIST = []


def _hist_save_locked():
    try:
        os.makedirs(WORK, exist_ok=True)
        tmp = HIST_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(HIST[:HIST_MAX], f, ensure_ascii=False)
        os.replace(tmp, HIST_PATH)
    except Exception:
        pass


def hist_put(rec):
    with LOCK:
        for i, r in enumerate(HIST):
            if r.get("id") == rec.get("id"):
                HIST.pop(i)
                break
        HIST.insert(0, rec)
        del HIST[HIST_MAX:]
        _hist_save_locked()


def find_rec(jid):
    j = JOBS.get(jid)
    if j:
        return j
    with LOCK:
        for r in HIST:
            if r.get("id") == jid:
                return r
    return None


# ---------------------------------------------------------------- ffmpeg 辅助
def ffprobe_exe():
    base = os.path.dirname(ARGS.ffmpeg)
    cand = os.path.join(base, "ffprobe.exe")
    return cand if os.path.exists(cand) else "ffprobe"


def probe_frames(path):
    """优先用 nb_frames；没有的话用 duration x fps 估算（不逐帧解码，快）。"""
    try:
        cmd = [ffprobe_exe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=nb_frames,duration,r_frame_rate", "-of", "csv=p=0", path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        parts = [p for p in out.split(",") if p != ""]
        num = 0
        for p in parts:
            if p.isdigit():
                num = int(p)
                break
        if num > 0:
            return num
        dur, rate = 0.0, 0.0
        for p in parts:
            if "/" in p:
                a, b = p.split("/")[:2]
                if int(b or 1) != 0:
                    rate = int(a) / int(b)
            elif p and p[0].isdigit() and "." in p:
                dur = float(p)
        return int(round(dur * rate)) if dur > 0 and rate > 0 else 0
    except Exception:
        return 0


def probe_streams(path):
    """-> dict(video, audio, subs, w, h)；探测失败返回 None。"""
    try:
        cmd = [ffprobe_exe(), "-v", "error", "-show_entries",
               "stream=codec_type,codec_name,width,height", "-of", "json", path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        d = json.loads(out or "{}")
    except Exception:
        return None
    info = {"video": False, "audio": False, "subs": False, "w": 0, "h": 0}
    for s in d.get("streams", []) or []:
        t = s.get("codec_type")
        if t == "video" and not info["w"]:
            info["video"] = True
            info["w"] = int(s.get("width") or 0)
            info["h"] = int(s.get("height") or 0)
        elif t == "audio":
            info["audio"] = True
        elif t == "subtitle":
            info["subs"] = True
    return info


def mux_av(rec, in_path, video_path):
    """把源文件的音轨 / 字幕轨原样接回结果（-c copy）。返回最终路径。"""
    info = probe_streams(in_path)
    if not info or not (info["audio"] or info["subs"]):
        return video_path
    rec["audio"], rec["subs"] = info["audio"], info["subs"]
    base = os.path.splitext(video_path)[0]
    if info["subs"]:
        dst = base + "_av.mkv"
        tries = [
            ["-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-c", "copy"],
            ["-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"],
            ["-map", "0:v:0", "-c:v", "copy"],
        ]
    else:
        dst = base + "_av.mp4"
        tries = [
            ["-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart"],
            ["-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart"],
            ["-map", "0:v:0", "-c:v", "copy", "-movflags", "+faststart"],
        ]
    for maps in tries:
        cmd = [ARGS.ffmpeg, "-y", "-v", "error", "-i", video_path, "-i", in_path]
        cmd += maps
        cmd += ["-map_metadata", "1", "-map_chapters", "1", dst]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except Exception:
            continue
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst
    rec["mux_err"] = "音轨 / 字幕接回失败，已保留纯视频"
    return video_path


# ---------------------------------------------------------------- 任务
def weights_dir(size):
    """权重目录：兼容 <根>/models_rt 与 <根>/models/models_rt 两种打包布局。"""
    for base in (os.path.join(HERE, "models_rt"), os.path.join(HERE, "models", "models_rt")):
        d = os.path.join(base, "nanovsr_" + size)
        if os.path.isfile(os.path.join(d, "weights_rt.json")):
            return d
    return None


def build_cmd(in_path, out_path, o):
    cmd = [os.path.join(HERE, "nanovsr.exe"), "--in", in_path, "--out", out_path,
           "--preset", o["preset"], "--ffmpeg", ARGS.ffmpeg]
    wd = weights_dir(o["model"])
    if wd:
        cmd += ["--model", wd]
    else:
        cmd += ["--model-size", o["model"]]
    if o.get("chunk", 0) > 0:
        cmd += ["--chunk", str(o["chunk"])]
    if o.get("tile"):
        cmd += ["--tile", o["tile"]]
    if o.get("crf", -1) >= 0:
        cmd += ["--crf", str(o["crf"])]
    if o.get("limit", 0) > 0:
        cmd += ["--limit", str(o["limit"])]
    return cmd


def run_job(jid, rec, in_path, video_path, o):
    job = rec
    cmd = build_cmd(in_path, video_path, o)
    job["cmd"] = " ".join(cmd)
    try:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1)
        for raw in iter(p.stdout.readline, ""):
            for line in raw.replace("\r", "\n").split("\n"):
                line = line.strip()
                if not line:
                    continue
                m = re.search(r"处理 (\d+) 帧", line)
                if m:
                    job["frames"] = int(m.group(1))
                m = re.search(r"chunk = (\d+)", line)
                if m:
                    job["chunk"] = int(m.group(1))
                m = re.search(r"(\d+)x(\d+)\s+([\d.]+) fps", line)
                if m:
                    job["w"], job["h"], job["fps_src"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
                job["log"] = (job.get("log", "") + line + "\n")[-4000:]
        p.wait()
        job["rc"] = p.returncode
        if p.returncode == 0:
            final = mux_av(job, in_path, video_path) if not o.get("is_img") else video_path
            job["out"] = final
            job["state"] = "done"
            if final != video_path:
                try:
                    os.remove(video_path)
                except Exception:
                    pass
        else:
            job["state"] = "error"
            job["err"] = "退出码 %d\n" % p.returncode + job.get("log", "")[-1500:]
    except Exception as e:                     # noqa: BLE001
        job["state"] = "error"
        job["err"] = str(e)
    finally:
        job["t1"] = time.time()
        job["elapsed"] = job["t1"] - job["t0"]
        job["frames"] = job.get("frames", 0)
        try:
            job["size"] = os.path.getsize(job["out"]) if os.path.exists(job["out"]) else 0
        except Exception:
            job["size"] = 0
        hist_put({k: v for k, v in job.items() if k != "log"})
        with LOCK:
            JOBS.pop(jid, None)


# ---------------------------------------------------------------- HTTP
def ctype_of(fn):
    e = os.path.splitext(fn)[1].lower()
    return {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm",
            ".mov": "video/quicktime", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp", ".bmp": "image/bmp"}.get(
        e, "application/octet-stream")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                 # 安静点
        pass

    def _send(self, code, body=b"", ctype="application/json; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code, obj):
        return self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    # ---------------- GET ----------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path = u.path
        q = urllib.parse.parse_qs(u.query)
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "webui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/health":
            import socket
            ok = {"webui": "ok"}
            try:
                ok["ethernet"] = "ok" if socket.gethostbyname_ex(socket.gethostname())[2] else "unknown"
            except Exception:
                ok["ethernet"] = "unknown"
            try:
                out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                      "(Get-NetIPAddress -IPAddress 66.66.66.67 -ErrorAction SilentlyContinue) -ne $null"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().lower()
                ok["easytier_ip"] = "ok" if out == "true" else "missing"
            except Exception:
                ok["easytier_ip"] = "unknown"
            try:
                s = socket.create_connection(("66.66.66.67", 7861), timeout=2)
                s.close()
                ok["forwarder"] = "ok"
            except Exception:
                ok["forwarder"] = "dead"
            return self._json(200, ok)
        if path == "/api/status":
            jid = q.get("id", [""])[0]
            rec = find_rec(jid)
            if not rec:
                return self._json(404, {"error": "no such job"})
            with LOCK:
                d = dict(rec)
            d["log"] = (rec.get("log") or "")[-1200:]
            return self._json(200, d)
        if path == "/api/history":
            with LOCK:
                items = []
                for r in HIST:
                    d = {k: v for k, v in r.items() if k != "log"}
                    d["exists"] = bool(r.get("out")) and os.path.exists(r["out"])
                    items.append(d)
            return self._json(200, {"items": items})
        if path.startswith("/api/out/"):
            jid = path.rsplit("/", 1)[-1]
            rec = find_rec(jid)
            if not rec or rec.get("state") != "done":
                return self._json(404, {"error": "not ready"})
            fn = rec.get("out") or ""
            if not os.path.exists(fn):
                return self._json(404, {"error": "missing"})
            return self._serve_file(fn, rec.get("name") or os.path.basename(fn))
        return self._json(404, {"error": "not found"})

    def _serve_file(self, fn, orig_name):
        size = os.path.getsize(fn)
        stem, ext = os.path.splitext(orig_name)
        out_name = stem + "_4x" + ext
        ct = ctype_of(fn)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                if end < start:
                    end = size - 1
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Disposition",
                         'attachment; filename="%s"' % out_name.encode("ascii", "replace").decode())
        self.end_headers()
        with open(fn, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                blk = f.read(min(1 << 20, left))
                if not blk:
                    break
                try:
                    self.wfile.write(blk)
                except Exception:
                    return
                left -= len(blk)

    # ---------------- POST ----------------
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        path, q = u.path, urllib.parse.parse_qs(u.query)
        if path == "/api/run":
            return self._api_run(q)
        if path == "/api/history/delete":
            jid = q.get("id", [""])[0]
            with LOCK:
                for i, r in enumerate(HIST):
                    if r.get("id") == jid:
                        HIST.pop(i)
                        break
                _hist_save_locked()
            d = os.path.join(WORK, jid)
            if jid and os.path.isdir(d) and os.path.abspath(d).startswith(os.path.abspath(WORK) + os.sep):
                shutil.rmtree(d, ignore_errors=True)
            return self._json(200, {"ok": True})
        if path == "/api/history/clear":
            with LOCK:
                ids = [r.get("id") for r in HIST]
                HIST.clear()
                _hist_save_locked()
            for jid in ids:
                d = os.path.join(WORK, jid)
                if jid and os.path.isdir(d) and os.path.abspath(d).startswith(os.path.abspath(WORK) + os.sep):
                    shutil.rmtree(d, ignore_errors=True)
            return self._json(200, {"ok": True, "removed": len(ids)})
        return self._json(404, {"error": "not found"})

    def _api_run(self, q):
        name = os.path.basename(q.get("name", ["input.mp4"])[0]) or "input.mp4"
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return self._json(400, {"error": "empty upload"})
        jid = uuid.uuid4().hex[:12]
        work = os.path.join(WORK, jid)
        os.makedirs(work, exist_ok=True)
        in_path = os.path.join(work, re.sub(r"[^\w.\-]", "_", name))
        remaining = n
        with open(in_path, "wb") as f:
            while remaining > 0:
                blk = self.rfile.read(min(1 << 20, remaining))
                if not blk:
                    break
                f.write(blk)
                remaining -= len(blk)

        preset = q.get("preset", ["balanced"])[0]
        if preset not in ("preview", "balanced", "quality"):
            preset = "balanced"
        model = q.get("model", ["1.7m"])[0]
        if model not in ("644k", "1.7m", "5.4m"):
            model = "1.7m"
        limit = max(0, int(q.get("limit", ["0"])[0] or 0))
        try:
            chunk = int(q.get("chunk", ["0"])[0] or 0)
        except Exception:
            chunk = 0
        chunk = max(0, min(30, chunk))
        try:
            crf = int(q.get("crf", ["-1"])[0] or -1)
        except Exception:
            crf = -1
        crf = -1 if (crf < 0 or crf > 51) else crf
        tile = (q.get("tile", [""])[0] or "").strip().lower()
        if not re.match(r"^\d{2,5}x\d{2,5}$", tile):
            tile = ""
        is_img = os.path.splitext(name)[1].lower() in IMAGE_EXT
        out_path = os.path.join(work, "out.png" if is_img else "out.mp4")
        job = dict(id=jid, state="running", frames=0, total=(1 if is_img else probe_frames(in_path)),
                   out=out_path, t0=time.time(), preset=preset, model=model, name=name,
                   chunk=chunk, crf=crf, tile=tile, limit=limit, is_img=is_img,
                   created=time.time(), elapsed=0, size=0)
        JOBS[jid] = job
        threading.Thread(target=run_job, args=(jid, job, in_path, out_path, job), daemon=True).start()
        return self._json(200, {"id": jid})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG") or (
        r"D:\ffmpeg-9.0.1\ffmpeg.exe" if os.path.exists(r"D:\ffmpeg-9.0.1\ffmpeg.exe") else "ffmpeg"))
    ARGS = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    hist_load()
    try:                                   # Windows 下 SO_REUSEADDR 允许多个进程抢同一端口，
        s = socket.create_connection((ARGS.host, ARGS.port), timeout=0.6)   # 先探一下
        s.close()
        print("端口 %d 已被占用（面板可能已经在跑）: http://%s:%d" % (ARGS.port, ARGS.host, ARGS.port))
        sys.exit(2)
    except OSError:
        pass
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print("NanoVSR WebUI: http://%s:%d  (ffmpeg=%s, history=%d)" % (
        ARGS.host, ARGS.port, ARGS.ffmpeg, len(HIST)))
    srv.serve_forever()


if __name__ == "__main__":
    main()
