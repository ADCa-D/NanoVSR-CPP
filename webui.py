# -*- coding: utf-8 -*-
"""NanoVSR 4x 超分 WebUI —— 纯标准库实现，底层调用 nanovsr.exe。

    python webui.py                 # 默认 http://127.0.0.1:7861
    python webui.py --port 8000

依赖：本目录下的 nanovsr.exe、shaders/*.spv、models_rt/nanovsr_1.7m/、
      ffmpeg（默认 D:\\ffmpeg-9.0.1\\ffmpeg.exe，可用 --ffmpeg 指定）。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = {}                 # id -> dict(state, frames, total, out, err, t0, fps)
LOCK = threading.Lock()
ARGS = None


def probe_frames(ffprobe, path):
    """优先用 nb_frames；没有的话用 duration x fps 估算（不逐帧解码，快）。"""
    try:
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
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


def run_job(job_id, in_path, out_path, preset, model, limit):
    job = JOBS[job_id]
    cmd = [os.path.join(HERE, "nanovsr.exe"), "--in", in_path, "--out", out_path,
           "--preset", preset, "--model-size", model, "--ffmpeg", ARGS.ffmpeg]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    job["cmd"] = " ".join(cmd)
    try:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1)
        buf = ""
        while True:
            ch = p.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                line = buf.strip()
                buf = ""
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
            else:
                buf += ch
        p.wait()
        job["rc"] = p.returncode
        job["state"] = "done" if p.returncode == 0 else "error"
        job["err"] = "" if p.returncode == 0 else f"退出码 {p.returncode}\n" + job.get("log", "")[-1500:]
    except Exception as e:                     # noqa: BLE001
        job["state"] = "error"
        job["err"] = str(e)
    finally:
        job["t1"] = time.time()
        job["elapsed"] = job["t1"] - job["t0"]


class Handler(BaseHTTPRequestHandler):
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

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "webui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/status":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            jid = q.get("id", [""])[0]
            job = JOBS.get(jid)
            if not job:
                return self._send(404, b'{"error":"no such job"}')
            with LOCK:
                d = {k: v for k, v in job.items() if k != "log"}
                d["log"] = job.get("log", "")[-1200:]
            return self._send(200, json.dumps(d, ensure_ascii=False).encode())
        if path.startswith("/api/out/"):
            jid = path.rsplit("/", 1)[-1]
            job = JOBS.get(jid)
            if not job or job.get("state") != "done":
                return self._send(404, b'{"error":"not ready"}')
            fn = job["out"]
            if not os.path.exists(fn):
                return self._send(404, b'{"error":"missing"}')
            size = os.path.getsize(fn)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{os.path.basename(fn)}"')
            self.end_headers()
            with open(fn, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
            return
        return self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/run":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name = os.path.basename(q.get("name", ["input.mp4"])[0]) or "input.mp4"
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                return self._send(400, b'{"error":"empty upload"}')
            jid = uuid.uuid4().hex[:12]
            work = os.path.join(HERE, "work", jid)
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
            limit = int(q.get("limit", ["0"])[0] or 0)
            out_path = os.path.join(work, "out.mp4")
            ffprobe = os.path.join(os.path.dirname(ARGS.ffmpeg), "ffprobe.exe")
            job = dict(state="running", frames=0, out=out_path, t0=time.time(), preset=preset,
                       total=probe_frames(ffprobe, in_path),
                       name=name, model=model)
            JOBS[jid] = job
            threading.Thread(target=run_job, args=(jid, in_path, out_path, preset, model, limit),
                             daemon=True).start()
            return self._send(200, json.dumps({"id": jid}).encode())
        return self._send(404, b'{"error":"not found"}')


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG") or (r"D:\ffmpeg-9.0.1\ffmpeg.exe" if os.path.exists(r"D:\ffmpeg-9.0.1\ffmpeg.exe") else "ffmpeg"))
    ARGS = ap.parse_args()
    os.makedirs(os.path.join(HERE, "work"), exist_ok=True)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"NanoVSR WebUI: http://{ARGS.host}:{ARGS.port}  (ffmpeg={ARGS.ffmpeg})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
