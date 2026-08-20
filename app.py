"""
Face Catch Web UI - Flask backend.
Resim / video dosyası / YouTube akışı üzerinde tüm özellikleri çalıştırır,
sonucu görüntüler ve hız istatistiklerini döndürür.
"""
import os, io, json, time, threading, subprocess, uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_from_directory, render_template_string
from werkzeug.utils import secure_filename

import engine
from engine import UnifaceEngine

# --- LD_LIBRARY_PATH: onnxruntime CUDA provider için ---
import sys as _sys
_os = os
_ld = os.environ.get("LD_LIBRARY_PATH", "")
_venv = os.environ.get("UNIFACE_VENV", "")
_site_dirs = [p for p in _sys.path if p.endswith(os.sep + "site-packages")]
if _venv:
    _site_dirs.insert(0, os.path.join(_venv, "lib", "site-packages"))
_nvidia = next((os.path.join(d, "nvidia") for d in _site_dirs
                if os.path.isdir(os.path.join(d, "nvidia"))), None)
if _nvidia:
    for _p in ["cu13/lib", "cudnn/lib"]:
        _d = os.path.join(_nvidia, _p)
        if _d not in _ld:
            _ld = _d + ":" + _ld
if os.path.isdir("/usr/local/lib/ollama/cuda_v13") and "/usr/local/lib/ollama/cuda_v13" not in _ld:
    _ld = "/usr/local/lib/ollama/cuda_v13:" + _ld
os.environ["LD_LIBRARY_PATH"] = _ld

# --- global canlı config (stream her karede bunu okur; frontend /api/config ile günceller) ---
LIVE_OPTS = {
    "recognition": True, "age_sex": True, "emotion": True, "face_states": True,
    "headpose": True, "gaze": True, "quality": True, "spoofing": True,
    "landmark106": False, "facemesh": False, "tracking": False,
    "parsing": False, "blur": False, "age_sex_model": "agegender",
    "ori_speed": True,
}

def default_opts():
    return dict(LIVE_OPTS)


app = Flask(__name__)

# --- güvenlik limitleri ---
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB: upload/cam_frame DoS koruması

# --- dinleme adresi: varsayılan yalnızca localhost; LAN erişimi için UNIFACE_HOST=0.0.0.0 ---
HOST = os.environ.get("UNIFACE_HOST", "127.0.0.1")

# --- Motor (tek örnek, tüm istekler paylaşır) ---
_lock = threading.Lock()
eng = None

def get_engine():
    global eng
    if eng is None:
        with _lock:
            if eng is None:
                print("[uniface] Motor başlatılıyor (modeller yükleniyor)...")
                t0 = time.time()
                eng = UnifaceEngine()
                # çoğu modeli önden yükleyelim ki ilk istek hızlı olsun
                eng._ensure("age_gender")
                print(f"[uniface] Motor hazır ({time.time()-t0:.1f} sn).")
    return eng


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

VOUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VOUT_DIR, exist_ok=True)

ALL_FEATURES = [
    "recognition", "age_sex", "emotion", "face_states",
    "headpose", "gaze", "quality", "spoofing",
    "landmark106", "facemesh", "tracking",
]


def parse_opts():
    o = {}
    for f in ALL_FEATURES:
        o[f] = request.form.get(f, "false").lower() == "true"
    o["parsing"] = request.form.get("parsing", "false").lower() == "true"
    o["age_sex_model"] = request.form.get("age_sex_model", "agegender")
    o["blur"] = request.form.get("blur", "false").lower() == "true"
    return o


def bgr_to_jpeg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/api/status")
def status():
    e = get_engine()
    return jsonify({
        "ready": True,
        "models_loaded": sorted(e._loaded),
        "device": "CUDA" if os.environ.get("LD_LIBRARY_PATH") else "?",
        "features": ALL_FEATURES,
        "load_times": e.timer,
        "refs_loaded": sum(1 for r in e._references if r is not None),
        "refs_max": len(e._references),
    })


@app.route("/api/reference", methods=["POST"])
def set_reference():
    """idx'li slota aranan kişi yüzü yaz (recognition karşılaştırması)."""
    e = get_engine()
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "resim gerekli"}), 400
    idx = request.form.get("idx", "0")
    try:
        idx = int(idx)
    except ValueError:
        return jsonify({"error": "idx sayı olmalı"}), 400
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "geçersiz görüntü"}), 400
    try:
        info = e.set_reference(img, idx)
        return jsonify({"ok": True, **info})
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400


@app.route("/api/clear_reference", methods=["POST"])
def clear_reference():
    e = get_engine()
    idx = request.form.get("idx")
    if idx is None or idx == "":
        res = e.clear_reference(None)
    else:
        try:
            res = e.clear_reference(int(idx))
        except (ValueError, IndexError):
            return jsonify({"error": "idx geçersiz"}), 400
    return jsonify({"ok": True, **res})


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Tek resim analizi -> işlenmiş JPEG + JSON istatistik."""
    e = get_engine()
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "resim gerekli"}), 400
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "geçersiz görüntü"}), 400
    opts = parse_opts()
    t0 = time.time()
    out, stats = e.process_image(img, opts)   # blur dahil (engine içi)
    total_ms = (time.time() - t0) * 1000
    stats["total_ms"] = round(total_ms, 1)
    jpg = bgr_to_jpeg(out)
    # JSON + görüntüyü tek yanıtta (JSON body içine base64)
    b64 = __import__("base64").b64encode(jpg).decode()
    return jsonify({
        "stats": stats,
        "image_b64": b64,
        "total_ms": round(total_ms, 1),
    })


# ------------------------- Video / YouTube akışı -------------------------

def _opts_from_query():
    o = {}
    for f in ALL_FEATURES:
        o[f] = request.args.get(f, "false").lower() == "true"
    o["parsing"] = request.args.get("parsing", "false").lower() == "true"
    o["age_sex_model"] = request.args.get("age_sex_model", "agegender")
    o["blur"] = request.args.get("blur", "false").lower() == "true"
    return o


def _open_capture(src_kind, src):
    """Video dosyası veya YouTube URL'sinden VideoCapture açar."""
    if src_kind == "file":
        # ponytail: path traversal fix — only allow basenames inside UPLOAD_DIR
        safe_name = os.path.basename(str(src))
        if not safe_name or safe_name.startswith('.'):
            raise RuntimeError("geçersiz dosya adı")
        path = os.path.join(UPLOAD_DIR, safe_name)
        real = os.path.realpath(path)
        base = os.path.realpath(UPLOAD_DIR) + os.sep
        if not real.startswith(base):
            raise RuntimeError("dosya dizini dışında")
        cap = cv2.VideoCapture(real)
        if not cap.isOpened():
            raise RuntimeError(f"dosya açılamadı: {safe_name}")
        return cap, real
    # YouTube/Dailymotion URL - yt-dlp ile akışı çöz, sonra OpenCV veya ffmpeg ile aç
    import subprocess, re
    # SSRF mitigation — allow YouTube + Dailymotion (yt-dlp supports both)
    YT_RE = re.compile(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com|dailymotion\.com|www\.dailymotion\.com|dai\.ly)/', re.I)
    if not YT_RE.match(str(src) or ''):
        raise RuntimeError("desteklenen yalnızca YouTube ve Dailymotion URL'leri")
    try:
        is_yt = "youtu" in str(src).lower()
        if is_yt:
            cmd = ["yt-dlp", "-f", "b[height<=720][vcodec!=av01]", "-g", "--no-warnings",
                   "--extractor-args", "youtube:player_client=android", src]
        else:
            cmd = ["yt-dlp", "-f", "b[height<=720]", "-g", "--no-warnings", src]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        url = (r.stdout or "").strip().splitlines()
        if not url:
            raise RuntimeError(f"yt-dlp akış bulamadı: {r.stderr[:200]}")
        stream_url = url[0]
        print("[yt] stream:", stream_url[:80], "...")
        # Dailymotion VOD'da cv2.VideoCapture(m3u8) EOF'ta temiz kapanmiyor (son kare donuyor),
        # bu yuzden Dailymotion icin dogrudan ffmpeg pipe kullan -> bitince stdout kapanir, gen_frames break eder.
        if not is_yt:
            cmd2 = ["ffmpeg", "-i", stream_url, "-f", "image2pipe", "-vcodec", "mjpeg",
                    "-preset", "ultrafast", "-q:v", "4", "pipe:1"]
            proc = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return FfmpegPipe(proc), stream_url
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            return cap, stream_url
        # cv2 açamadı -> ffmpeg pipe (stdout'tan okunur)
        cmd2 = ["ffmpeg", "-i", stream_url, "-f", "image2pipe", "-vcodec", "mjpeg",
                "-preset", "ultrafast", "-q:v", "4", "pipe:1"]
        proc = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return FfmpegPipe(proc), stream_url
    except Exception as ex:
        raise RuntimeError(f"YouTube akış açılamadı: {ex}")


class FfmpegPipe:
    """ffmpeg mjpeg pipe'ını cv2.VideoCapture arayüzüyle saran küçük sarmalayıcı."""
    def __init__(self, proc):
        self.proc = proc
        self._buf = b""
    def read(self):
        # mjpeg frame ayracı (FFD8...FFD9) bulana kadar oku
        while True:
            s = self._buf.find(b"\xff\xd9")
            if s != -1:
                frame = self._buf[:s+2]
                self._buf = self._buf[s+2:]
                img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return True, img
            chunk = self.proc.stdout.read(8192)
            if not chunk:
                return False, None
            self._buf += chunk
    def release(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


def gen_frames(src_kind, src, opts):
    e = get_engine()
    cap, _ = _open_capture(src_kind, src)
    # yeni kaynak: izleyiciyi bir kez sıfırla (track ID'leri 1'den başlar)
    if LIVE_OPTS.get("tracking"):
        e.reset_tracking()
    fps_report = {"frames": 0, "times": []}
    # orijinal hız modu: kareleri kaynak kaydın kendi zamanlamasına göre yay.
    # Takip (tracking) AÇIKKEN de zorunlu pacing: BYTETracker kareler arası IoU'ya
    # dayanır; akış gerçek hızdan hızlı beslenirse yüzler kareler arası çok
    # kaydığından aynı kişiye her seferinde YENİ ID açılır (22 -> 45 gibi).
    # Bu yüzden takip açıkken akış kaynak hızına otomatik yavaşlar.
    period = 0.0
    if LIVE_OPTS.get("ori_speed") or LIVE_OPTS.get("tracking"):
        f = 0.0
        try:
            if hasattr(cap, "get"):
                f = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        except Exception:
            f = 0.0
        if not (0 < f <= 1000):
            f = 25.0
        period = 1.0 / f
    next_t = time.time()
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            # canlı config: her karede güncel LIVE_OPTS'u oku (frontend değişince anında geçerli)
            out, stats = e.process_image(frame, LIVE_OPTS)
            dt = (time.time() - t0) * 1000
            fps_report["frames"] += 1
            fps_report["times"].append(dt)
            # FPS/ms üst köşeye çiz
            avg = sum(fps_report["times"][-30:]) / len(fps_report["times"][-30:])
            cv2.putText(out, f"FPS ~{1000/avg:.1f}  {avg:.0f}ms/kare  yuz:{stats['faces']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   bgr_to_jpeg(out, 80) + b"\r\n")
            # orijinal hız: bir sonraki kareye kadar bekle (kaynak fps'e göre)
            if period:
                next_t += period
                wait = next_t - time.time()
                if wait > 0:
                    time.sleep(wait)
    finally:
        if hasattr(cap, "release"):
            cap.release()


@app.route("/api/config", methods=["POST"])
def set_live_config():
    """Canlı stream yapılandırmasını güncelle (özellikler). Anında geçerli olur."""
    global LIVE_OPTS
    data = request.get_json(force=True, silent=True) or {}
    # sadece bilinen anahtarları kabul et
    for k in LIVE_OPTS:
        if k in data:
            LIVE_OPTS[k] = data[k]
    return jsonify({"ok": True, "live": LIVE_OPTS})


@app.route("/api/stream")
def stream():
    src_kind = request.args.get("src", "file")
    src = request.args.get("url", "")
    if src_kind == "file":
        # ponytail: sanitize filename from query
        safe_name = os.path.basename(str(src)) if src else ""
        if not safe_name:
            files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            if not files:
                return jsonify({"error": "video yüklenmemiş"}), 400
            files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIR, f)), reverse=True)
            src = files[0]
        else:
            src = safe_name
    if src_kind == "yt":
        import re
        YT_RE = re.compile(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com|dailymotion\.com|www\.dailymotion\.com|dai\.ly)/', re.I)
        if not src or not YT_RE.match(src):
            return jsonify({"error": "geçersiz YouTube/Dailymotion URL"}), 400
    return Response(gen_frames(src_kind, src, None),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cam_frame", methods=["POST"])
def cam_frame():
    """Web kamerası karesini analiz edip işlenmiş JPEG döndürür.
    WSL2 kamerası göremediğinden tarayıcı (kameranın olduğu Windows tarafı)
    her kareyi buraya POST eder; analiz GPU'da yapılır, canlı görüntü döner."""
    data = request.get_data()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "hata: kare çözülemedi", 400
    e = get_engine()
    out, _stats = e.process_image(frame, LIVE_OPTS)
    jpg = bgr_to_jpeg(out, 80)
    return Response(jpg, mimetype="image/jpeg")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "video gerekli"}), 400
    # Werkzeug dosya adını temizlemez: path traversal saldırılarına karşı
    # asla istemci adını kullanma; uuid'li güvenli isim üret.
    safe = secure_filename(file.filename or "") or f"video{uuid.uuid4().hex[:8]}.mp4"
    ext = os.path.splitext(safe)[1].lower() or ".mp4"
    name = f"{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    file.save(path)
    return jsonify({"ok": True, "file": name, "path": path})


# ----------------- ANALİZLİ MP4 DIŞA AKTARMA (kontrollü oynatma) -----------------
# Canlı MJPEG akışı aynen korunur; bu iş arka plan thread'inde kaynağı (dosya VEYA
# YouTube) kare kare analiz edip H.264 MP4'e (faststart) kodlar. Frontend ilerlemeyi
# /api/video_out/progress ile yoklar. Sonuç <video controls> ile seek/sürükleme,
# ayrıca özel hız butonları (0.25x-4x) ile izlenir.

JOBS = {}
_active_job = {"id": None}
_job_lock = threading.Lock()
HARD_CAP_SECONDS = 600  # uzunluğu bilinmeyen (canlı) akışlarda tavan


def _prune_videos(keep=5):
    try:
        fs = [os.path.join(VOUT_DIR, f) for f in os.listdir(VOUT_DIR) if f.endswith(".mp4")]
        if len(fs) > keep:
            fs.sort(key=os.path.getmtime)
            for p in fs[:-keep]:
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception:
        pass


def _export_worker(job_id, src_kind, src, opts, seconds):
    job = JOBS[job_id]
    e = get_engine()
    out_path = os.path.join(VOUT_DIR, f"out_{job_id}.mp4")
    tmp_path = out_path + ".part.mp4"
    proc = None
    cap = None
    try:
        cap, _src = _open_capture(src_kind, src)
        if hasattr(cap, "isOpened") and not cap.isOpened():
            raise RuntimeError("kaynak açılamadı")
        # yeni kaynak: izleyiciyi bir kez sıfırla (track ID'leri 1'den başlar)
        if opts.get("tracking"):
            e.reset_tracking()
        fps = 0.0
        total = 0
        w = h = 0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            pass  # FfmpegPipe'ta get() yok -> ilk kareden al
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("ilk kare alınamadı")
        if not w or not h:
            h, w = frame.shape[:2]
        if not fps or fps > 1000:
            fps = 25.0
        if seconds and seconds > 0:
            max_frames = int(seconds * fps)
        elif total > 0:
            max_frames = None
        else:
            max_frames = int(HARD_CAP_SECONDS * fps)
        # ilerleme, asıl işlenecek kare sayısına göre olsun
        if max_frames is not None and total > max_frames:
            total = max_frames
        job["total"] = total
        job["fps"] = round(fps, 1)
        job["res"] = f"{w}x{h}"
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "pipe:0",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        n = 0
        t_sum = 0.0
        first = True
        recent = []
        while not job.get("cancel"):
            if not first:
                ret, frame = cap.read()
                if not ret:
                    break
            first = False
            if max_frames is not None and n >= max_frames:
                break
            t0 = time.time()
            out, stats = e.process_image(frame, opts)
            dt = (time.time() - t0) * 1000
            t_sum += dt
            recent.append(dt)
            avg = sum(recent[-30:]) / len(recent[-30:])
            cv2.putText(out, f"MP4 {avg:.0f}ms/kare  yuz:{stats.get('faces', 0)}  kare:{n + 1}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            try:
                proc.stdin.write(out.tobytes())
            except (BrokenPipeError, OSError):
                break
            n += 1
            job["frame"] = n
            job["progress"] = round(n / total * 100, 1) if total > 0 else None
        if n == 0:
            raise RuntimeError("kare işlenemedi (iptal edildi mi?)")
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=120)
        job["frames"] = n
        job["avg_ms"] = round(t_sum / n, 1)
        if job.get("cancel"):
            job["status"] = "cancelled"
        else:
            os.replace(tmp_path, out_path)
            job["status"] = "done"
            job["url"] = "/videos/" + os.path.basename(out_path)
            job["size_mb"] = round(os.path.getsize(out_path) / 1048576, 1)
    except Exception as ex:
        job["status"] = "error"
        job["error"] = str(ex)
    finally:
        try:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception:
            pass
        if cap is not None and hasattr(cap, "release"):
            try:
                cap.release()
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        with _job_lock:
            if _active_job["id"] == job_id:
                _active_job["id"] = None


def _pick_latest_upload():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(UPLOAD_DIR, f)), reverse=True)
    return files[0]


@app.route("/api/video_out", methods=["POST"])
def video_out_start():
    data = request.get_json(force=True, silent=True) or {}
    src_kind = data.get("src", "file")
    src = (data.get("url") or "").strip()
    if src_kind == "file" and not src:
        src = _pick_latest_upload()
        if not src:
            return jsonify({"error": "video yüklenmemiş"}), 400
    # ponytail: validate src kind + file name
    if src_kind == "file":
        safe_name = os.path.basename(str(src))
        if not safe_name or '.' not in safe_name:
            return jsonify({"error": "geçersiz dosya adı"}), 400
    if src_kind == "yt":
        import re
        YT_RE = re.compile(r'^(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com|dailymotion\.com|www\.dailymotion\.com|dai\.ly)/', re.I)
        if not src or not YT_RE.match(src):
            return jsonify({"error": "YouTube/Dailymotion URL gerekli"}), 400
    if src_kind == "yt" and not src:
        return jsonify({"error": "YouTube URL gerekli"}), 400
    try:
        seconds = int(data.get("seconds", 0) or 0)
    except (TypeError, ValueError):
        seconds = 0
    if seconds < 0:
        seconds = 0
    # opts: frontend checkbox durumunu (getOpts) kullan; eksik anahtarlar LIVE_OPTS'tan
    o = dict(LIVE_OPTS)
    opts = data.get("opts") or {}
    for k in LIVE_OPTS:
        if k in opts:
            o[k] = opts[k]
    with _job_lock:
        prev = _active_job["id"]
        if prev is not None and JOBS.get(prev, {}).get("status") == "running":
            return jsonify({"error": "Zaten bir MP4 işi çalışıyor"}), 409
        job_id = uuid.uuid4().hex[:10]
        JOBS[job_id] = {"status": "running", "frame": 0, "total": 0, "progress": 0.0,
                        "url": None, "error": None, "cancel": False}
        _active_job["id"] = job_id
    _prune_videos()
    threading.Thread(target=_export_worker, args=(job_id, src_kind, src, o, seconds),
                     daemon=True).start()
    return jsonify({"ok": True, "job": job_id})


@app.route("/api/video_out/progress")
def video_out_progress():
    job_id = request.args.get("job", "")
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "job bulunamadı"}), 404
    return jsonify({
        "status": j["status"], "frame": j.get("frame", 0), "total": j.get("total", 0),
        "progress": j.get("progress"), "url": j.get("url"), "error": j.get("error"),
        "frames": j.get("frames"), "size_mb": j.get("size_mb"), "avg_ms": j.get("avg_ms"),
        "fps": j.get("fps"), "res": j.get("res"),
    })


@app.route("/api/video_out/cancel", methods=["POST"])
def video_out_cancel():
    job_id = request.args.get("job", "")
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "job bulunamadı"}), 404
    j["cancel"] = True
    return jsonify({"ok": True})


@app.route("/videos/<path:name>")
def serve_video(name):
    # ponytail: block path traversal in video serving
    safe = os.path.basename(name)
    if safe != name or safe.startswith('.'):
        return "", 404
    return send_from_directory(VOUT_DIR, safe)


if __name__ == "__main__":
    # önce motoru yükle (ilk isteği beklemeden)
    get_engine()
    print(f"Face Catch UI başlatılıyor... http://{HOST}:8127  https://{HOST}:8443")
    # Kamera sekmesi (getUserMedia) HTTPS ister; self-signed sertifikayla 8443'te aç.
    import ssl as _ssl
    _cert = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs", "cert.pem")
    _key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs", "key.pem")
    if os.path.exists(_cert) and os.path.exists(_key):
        _ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
        _ctx.load_cert_chain(_cert, _key)
        _th = threading.Thread(target=lambda: app.run(host=HOST, port=8443,
                                                      ssl_context=_ctx, threaded=True),
                               daemon=True)
        _th.start()
        print(f"[tls] https://{HOST}:8443 yayında (self-signed)")
    else:
        print("[tls] certs bulunamadı, HTTPS kapalı")
    app.run(host=HOST, port=8127, threaded=True)