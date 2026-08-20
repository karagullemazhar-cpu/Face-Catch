"""
UniFace test motoru.
Tek bir sınıfta tüm modelleri yükler, resim/video karesi üzerinde
seçilen özellikleri çalıştırır, sonuç görüntüsünü çizer ve hızı ölçer.
"""
import os, time, json, logging
import numpy as np
import cv2

# Model cache'i hızlı (WSL ev dizini) olsun; /mnt/c üzerindeki IO yavaş.
CACHE_DIR = os.path.expanduser("~/.cache/uniface")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    from uniface import set_cache_dir
    set_cache_dir(CACHE_DIR)
except Exception:
    pass

from uniface.detection import SCRFD, RetinaFace
from uniface.recognition import ArcFace
from uniface.constants import ArcFaceWeights
from uniface.landmark import Landmark106, FaceMesh, PIPNet
from uniface.attribute import AgeGender, FairFace, Emotion, FaceAttribNet
from uniface.gaze import MobileGaze
from uniface.headpose import HeadPose
from uniface.quality import EDifFIQA
from uniface.spoofing import MiniFASNet
from uniface.parsing import BiSeNet, XSeg
from uniface.tracking import BYTETracker
from uniface.privacy import BlurFace
from uniface import compute_similarity
from uniface import draw


def _bbox_iou(a, b):
    """İki [x1,y1,x2,y2] kutusunun IoU'su (track_id-yüz eşleştirmesi için)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _draw_pose_axes(img, cx, cy, size, yaw_deg, pitch_deg, roll_deg):
    """Bas durusunu IRI, KONTRASTLI 3D eksenlerle cizer.
    FIX 2026-08-21: Z ekseni 4 yonde ters -> sadece Z terslendi (X,Y dogru)."
    Onceki surum cok kucuk/cilginca seffaf -> anlasilmiyordu. Simdi:
    - Eksenler %40 daha uzun, 2x daha kalin, siyah dis kontur + renk dolgu
    - Perspektif korunuyor (yaw/pitch/roll'e gore kisa/uzun)
    - Ucta harf yerine kalin daire + harf (X/Y/Z net)
    - Merkeze kucuk beyaz nokta (eksen kokeni belli)
    - Yazi arka planli, her zeminde okunur.
    X=kirmizi, Y=yesil, Z=mavi (BGR)."""
    import math as _m
    yaw, pitch, roll = _m.radians(yaw_deg), _m.radians(pitch_deg), _m.radians(roll_deg)
    cy_, sy = _m.cos(yaw), _m.sin(yaw)
    cp, sp = _m.cos(pitch), _m.sin(pitch)
    cr, sr = _m.cos(roll), _m.sin(roll)
    R = [
        (cr * cy_ + sr * sp * sy, -sr * cp, cr * sy - sr * sp * cy_),
        (sr * cy_ - cr * sp * sy,  cr * cp, sr * sy + cr * sp * cy_),
        (cp * sy,                   sp,      cp * cy_),
    ]
    def proj(px, py, pz, depth0, focal):
        Xc = R[0][0] * px + R[0][1] * py + R[0][2] * pz
        Yc = R[1][0] * px + R[1][1] * py + R[1][2] * pz
        Zc = R[2][0] * px + R[2][1] * py + R[2][2] * pz
        dep = depth0 + Zc
        if dep < 1e-3:
            dep = 1e-3
        return int(cx + focal * Xc / dep), int(cy + focal * Yc / dep), dep, Xc, Yc

    # onceki: L=size  -> simdi %45 daha uzun ki yuz kutusundan net ciksin
    L = int(size * 1.45)
    depth0 = max(80, size * 3.2)
    focal = size * 2.6
    o = (int(cx), int(cy))
    # merkeze beyaz nokta + siyah hale (koken belli)
    cv2.circle(img, o, 6, (0,0,0), -1)
    cv2.circle(img, o, 4, (255,255,255), -1)
    cv2.circle(img, o, 2, (0,0,0), -1)
    axes = [((L, 0, 0), (0, 0, 255), "X"), ((0, L, 0), (0, 255, 0), "Y"), ((0, 0, -L), (255, 0, 0), "Z")]
    for (px, py, pz), col, lab in axes:
        ex, ey, dep, Xc, Yc = proj(px, py, pz, depth0, focal)
        # yakin eksen daha kalin
        t_outer = 7
        t_inner = 4
        # dis kontur siyah (her zeminde kontrast)
        cv2.line(img, o, (ex, ey), (0,0,0), t_outer + 4)
        cv2.line(img, o, (ex, ey), col, t_inner + 2)
        # uc ok basi (siyah dis + renk ic)
        cv2.circle(img, (ex, ey), 9, (0,0,0), -1)
        cv2.circle(img, (ex, ey), 7, col, -1)
        # harf: siyah dis + beyaz ic, okunur
        tx, ty = ex + 10, ey - 10
        # harf arka kutu
        (tw, th), bl = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (tx-3, ty-th-4), (tx+tw+3, ty+bl+2), (0,0,0), -1)
        cv2.rectangle(img, (tx-3, ty-th-4), (tx+tw+3, ty+bl+2), col, 1)
        cv2.putText(img, lab, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    # aci etiketi: yaw/pitch/roll sayilari (merkezin altina)
    try:
        txt = f"Y{int(round(yaw_deg))} P{int(round(pitch_deg))} R{int(round(roll_deg))}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx, by = int(cx - tw//2), int(cy + L*0.55)
        # goruntu disina tasmayi kirp
        H, W = img.shape[:2]
        bx = max(2, min(bx, W - tw - 4))
        by = max(th+4, min(by, H-4))
        cv2.rectangle(img, (bx-4, by-th-6), (bx+tw+4, by+bl+2), (0,0,0), -1)
        cv2.putText(img, txt, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1, cv2.LINE_AA)
    except Exception:
        pass


def _draw_gaze_arrow(img, cx, cy, length, yaw_deg, pitch_deg):
    """Bakis yonunu IRI, KONTRASTLI isinla cizer. Onceki ince/soluk ok
    hic fark edilmiyordu. Simdi:
    - Siyah dis kontur + turuncu dolgu, 2x kalin
    - Ucta buyuk daire + ok basi
    - Karsiya bakista bile belirgin isaret (esmerkez daireler)
    - Hedefte acilari gosteren etiket.
    Renk: turuncu (0,140,255 BGR)."""
    import math as _m
    gx = -_m.sin(_m.radians(yaw_deg))
    gy = -_m.sin(_m.radians(pitch_deg))
    n = _m.hypot(gx, gy)
    col = (0, 140, 255)
    o = (int(cx), int(cy))
    if n < 1e-4:
        cv2.circle(img, o, 10, (0,0,0), -1)
        cv2.circle(img, o, 8, col, -1)
        cv2.circle(img, o, 4, (255,255,255), -1)
        cv2.circle(img, o, 2, (0,0,0), -1)
        return
    # acinin buyuklugune gore uzunluk: kucuk acida bile %60'luk min boy
    mag = max(0.55, min(1.0, n * 2.2))
    L = int(length * mag)
    ex = int(cx + gx / n * L)
    ey = int(cy + gy / n * L)
    # dis siyah kontur + ic turuncu (her zeminde gorunur)
    cv2.line(img, o, (ex, ey), (0,0,0), 10)
    cv2.line(img, o, (ex, ey), col, 6)
    # uc nokta: siyah hale + turuncu daire + beyaz merkez
    cv2.circle(img, (ex, ey), 11, (0,0,0), -1)
    cv2.circle(img, (ex, ey), 9, col, -1)
    cv2.circle(img, (ex, ey), 3, (255,255,255), -1)
    # ok basi ucgeni (siyah dis + turuncu ic)
    ang = _m.atan2(ey - cy, ex - cx)
    ah = 14
    aw = 10
    p1 = (ex, ey)
    p2 = (int(ex - ah*_m.cos(ang) + aw*_m.sin(ang)), int(ey - ah*_m.sin(ang) - aw*_m.cos(ang)))
    p3 = (int(ex - ah*_m.cos(ang) - aw*_m.sin(ang)), int(ey - ah*_m.sin(ang) + aw*_m.cos(ang)))
    tri = __import__('numpy').array([p1,p2,p3], __import__('numpy').int32)
    cv2.fillPoly(img, [tri], (0,0,0))
    # ic ucgen biraz kucuk
    p2i = (int(ex - (ah-3)*_m.cos(ang) + (aw-3)*_m.sin(ang)), int(ey - (ah-3)*_m.sin(ang) - (aw-3)*_m.cos(ang)))
    p3i = (int(ex - (ah-3)*_m.cos(ang) - (aw-3)*_m.sin(ang)), int(ey - (ah-3)*_m.sin(ang) + (aw-3)*_m.cos(ang)))
    tri2 = __import__('numpy').array([p1,p2i,p3i], __import__('numpy').int32)
    cv2.fillPoly(img, [tri2], col)



def _adaptive_match_threshold(pf, base=0.50):
    """Kalite agirlikli adaptif esik (3 nolu optimizasyon).
    - Kalite yuksek -> esik dusuk (karsidan net yuzde 0.42'de bile esles)
    - Kalite dusuk / bulanik -> esik yuksek (yanlis maviyi engelle 0.58)
    - Bas acisi buyuk (yaw/pitch/roll >30) -> esigi 0.05 dusur (yan profili kacirma)
    Donus: 0.40-0.62 arasi kirpilmis esik.
    """
    th = base
    q = pf.get("quality")
    if q is not None:
        try:
            qv = float(q)
            if qv >= 0.70:
                th = 0.42
            elif qv >= 0.50:
                th = 0.46
            elif qv >= 0.30:
                th = 0.50
            elif qv >= 0.20:
                th = 0.55
            else:
                th = 0.58
        except Exception:
            pass
    # bas acisi: en buyuk acilimi baz al
    max_ang = 0.0
    for k in ("yaw", "pitch", "roll"):
        v = pf.get(k)
        if v is not None:
            try:
                max_ang = max(max_ang, abs(float(v)))
            except Exception:
                pass
    if max_ang > 30:
        th -= 0.05
        if max_ang > 45:
            th -= 0.03
    # kirp
    if th < 0.40:
        th = 0.40
    if th > 0.62:
        th = 0.62
    return round(float(th), 3)


class UnifaceEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self._loaded = set()
        self.timer = {}
        # Detector (baz olarak hep lazım)
        self.detector = SCRFD()
        # Recognizer — 5 nolu optimizasyon: ArcFace MNET -> RESNET (R50, w600k_r50)
        # Gerekce: ~3-4x parametre, ayni 512D, ayni arayuz; zor poz/isikta
        # dogruluk belirgin artar (4060'da ~5ms ek gecikme). Istenirse env ile
        # UNIFACE_RECOGNIZER=mnet|resnet|adaface_ir101 ile degistirilebilir.
        # Ilk acilista indirir (~160MB), sonrasinda cache'ten yukler.
        def _make_recognizer():
            pref = os.environ.get("UNIFACE_RECOGNIZER", "resnet").lower()
            if pref in ("adaface", "adaface_ir101", "ir101"):
                try:
                    from uniface.recognition import AdaFace
                    from uniface.constants import AdaFaceWeights
                    return AdaFace(model_name=AdaFaceWeights.IR_101)
                except Exception as e:
                    print(f"[uniface] AdaFace IR101 acilamadi ({e}), RESNET'e dusuluyor")
            if pref in ("mnet", "arcface_mnet"):
                return ArcFace(model_name=ArcFaceWeights.MNET)
            try:
                return ArcFace(model_name=ArcFaceWeights.RESNET)
            except Exception as e:
                print(f"[uniface] ArcFace RESNET acilamadi ({e}), MNET fallback")
                return ArcFace(model_name=ArcFaceWeights.MNET)
        self.recognizer = _make_recognizer()
        try:
            _rn = type(self.recognizer).__name__
            _mn = getattr(self.recognizer, 'model_path', '')
            print(f"[uniface] recognizer: {_rn} ({os.path.basename(str(_mn)) or 'ok'}) ")
        except Exception:
            pass
        # Opsiyonel modeller - ilk kullanımda yüklenir
        self.age_gender = None
        self.fairface = None
        self.emotion = None
        self.face_states = None
        self.gaze = None
        self.headpose = None
        self.quality = None
        self.spoofer = None
        self.parser = None
        self.xseg = None
        self.landmark106 = None
        self.facemesh = None
        self.pipnet = None
        self.tracker = BYTETracker()
        self.blur = BlurFace()
        self._references = [None] * 4      # 4 slotlu aranan kişi (embedding); None = boş
        # --- re-ID görünüm cache'i: track_id -> son görülen embedding ---
        self._track_embs = {}              # canlı track'lerin yüz embedding'i
        self._last_io_ids = []             # bir önceki karedeki IoU id listesi (iç tutarlılık)

    # ---- model yükleme (lazy) ----
    def _ensure(self, name):
        if name in self._loaded:
            return getattr(self, name)
        t0 = time.time()
        if name == "age_gender":
            self.age_gender = AgeGender()
        elif name == "fairface":
            self.fairface = FairFace()
        elif name == "emotion":
            self.emotion = Emotion()
        elif name == "face_states":
            self.face_states = FaceAttribNet()
        elif name == "gaze":
            self.gaze = MobileGaze()
        elif name == "headpose":
            self.headpose = HeadPose()
        elif name == "quality":
            self.quality = EDifFIQA()
        elif name == "spoofer":
            self.spoofer = MiniFASNet()
        elif name == "parser":
            self.parser = BiSeNet()
        elif name == "xseg":
            self.xseg = XSeg()
        elif name == "landmark106":
            self.landmark106 = Landmark106()
        elif name == "facemesh":
            self.facemesh = FaceMesh()
        self._loaded.add(name)
        self.timer[f"load_{name}"] = round(time.time() - t0, 2)
        return getattr(self, name)

    # ---- referans embedding (aranan kişi, 4 slot) ----
    def set_reference(self, image_bgr, idx=0, max_refs=4):
        """idx (0-3) numaralı slota kişi embedding'i yazar."""
        if idx < 0 or idx >= max_refs:
            raise ValueError(f"idx 0-{max_refs-1} arası olmalı")
        faces = self.detector.detect(image_bgr)
        if not faces:
            raise ValueError("Referans görüntüde yüz bulunamadı")
        face = faces[0]
        emb = self.recognizer.get_normalized_embedding(image_bgr, face.landmarks)
        self._references[idx] = emb
        n = sum(1 for e in self._references if e is not None)
        return {"idx": idx, "refs": n, "max": max_refs, "faces": len(faces), "emb_shape": list(emb.shape)}

    def clear_reference(self, idx=None):
        """idx'li slotu boşalt; idx None ise tümünü."""
        if idx is None:
            self._references = [None] * len(self._references)
        else:
            self._references[idx] = None
        return {"refs": sum(1 for e in self._references if e is not None)}

    def reset_tracking(self):
        """Yeni bir video/akış başlangıcında izleyiciyi sıfırla (ID'ler 1'den başlar)."""
        self.tracker.reset()
        self._track_embs.clear()
        self._last_io_ids = []

    # ---- ana analiz ----
    def process_image(self, image_bgr, opts, is_reference=False):
        """
        opts: dict of enabled features + params.
        Returns (annotated_bgr, json_stats)
        """
        stats = {"faces": 0, "per_face": [], "timings": {}}
        work = image_bgr.copy()

        # 1) Detection
        t0 = time.time()
        faces = self.detector.detect(work)
        stats["timings"]["detection_ms"] = round((time.time() - t0) * 1000, 1)
        stats["faces"] = len(faces)
        if not faces:
            return work, stats

        # 2) Recognition embedding + izleme
        face_track_ids = None
        if opts.get("tracking"):
            # tracker burada sıfırlanmaz; yeni bir kaynağa başlarken reset_tracking() çağrılır.
            # (Her karede reset, ID'leri sıfırdan başlatırdı -> kareler arası takip çalışmazdı.)
            dets = np.array([[f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.confidence] for f in faces], dtype=np.float32)
            tracks = self.tracker.update(dets)
            # DİKKAT: BYTETracker.update() takip edilen kutuları KENDİ sıralamasında döner
            # (girdi dets sırasıyla hizalı DEĞİL). Bu yüzden track_id'yi yüzle IoU eşleştirmesiyle
            # ilişkilendiriyoruz, pozisyonel indeksle değil.
            io_ids = [None] * len(faces)
            if len(tracks) > 0 and len(faces) > 0:
                for row in tracks:  # row = [x1,y1,x2,y2, track_id]
                    tid = int(row[4])
                    tbox = row[:4]
                    best_i, best_iou = -1, 0.0
                    for j, f in enumerate(faces):
                        iou = _bbox_iou(f.bbox[:4], tbox)
                        if iou > best_iou:
                            best_iou, best_i = iou, j
                    if best_iou >= 0.3 and io_ids[best_i] is None:
                        io_ids[best_i] = tid

            # --- re-ID: görünüm (ArcFace embedding) ile ID stabilizasyonu ---
            # BYTETracker saf IoU; akış hızlı/atlama-atlama karelerde aynı kişiye
            # sürekli YENİ ID açar (22 -> 45). Burada her yüzün embedding'ini bilinen
            # track'lerle karşılaştırıp, aynı kişi görülen eski ID'ye geri gömüyoruz.
            face_track_ids = list(io_ids)
            embs = []
            for f in faces:
                try:
                    emb = self.recognizer.get_normalized_embedding(work, f.landmarks)
                except Exception:
                    emb = None
                embs.append(emb)
            REID_TH = 0.45
            # 1) IoU ile güvenilir atanmış track'lerin embedding'lerini cache'e işle
            for i, tid in enumerate(io_ids):
                if tid is not None and embs[i] is not None:
                    self._track_embs[tid] = embs[i]
            # 2) YoU ile atanan (ama görünümce başka bir tanıdık track'i çağrıştıran)
            #    yeni ID'leri, o eski track'e remap et (churn engeli)
            for i, tid in enumerate(io_ids):
                if tid is None or embs[i] is None:
                    continue
                # bu track'i cache'ten düşmeden önce en benzerini ara (kendisi hariç)
                best_t, best_s = None, REID_TH
                for t, e in self._track_embs.items():
                    if t == tid:
                        continue
                    s = compute_similarity(embs[i], e, normalized=True)
                    if s > best_s:
                        best_s, best_t = s, t
                if best_t is not None:
                    face_track_ids[i] = best_t   # yeni id yerine eski tanıdık id
            # 3) IoU ile ID alamayan yüzleri embedding ile tanı
            for i in range(len(faces)):
                if face_track_ids[i] is not None or embs[i] is None:
                    continue
                best_t, best_s = None, REID_TH
                for t, e in self._track_embs.items():
                    s = compute_similarity(embs[i], e, normalized=True)
                    if s > best_s:
                        best_s, best_t = s, t
                face_track_ids[i] = best_t
            # cache'i canlı/aktif ID'lerle sınırla (ölü/yedek BYTETracker ID büyümesini engelle):
            # bu karede re-ID sonrası kullanılanlar + bir önceki karedeki aktifler
            active = set(t for t in face_track_ids if t is not None)
            active |= set(t for t in self._last_io_ids if t is not None)
            self._track_embs = {t: e for t, e in self._track_embs.items() if t in active}
            self._last_io_ids = io_ids
        # face_track_ids: faces sıralamasıyla hizalı liste; her eleman track_id veya None

        # 3) Parsing / XSeg (görüntü genelinde overlay) - sadece ilk yüz için basit gösterim
        parsing_overlay = None

        for i, face in enumerate(faces):
            pf = {"idx": i, "bbox": [round(float(x), 1) for x in face.bbox], "conf": round(float(face.confidence), 3)}
            x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            crop = work[y1:y2, x1:x2]

            # --- Recognition (embedding + karşılaştırma) ---
            if opts.get("recognition"):
                t = time.time()
                emb = self.recognizer.get_normalized_embedding(work, face.landmarks)
                face.embedding = emb
                pf["emb_shape"] = list(emb.shape)
                if any(e is not None for e in self._references):
                    best_sim = -1.0; best_idx = -1
                    for ri, ref in enumerate(self._references):
                        if ref is None:
                            continue
                        sim = compute_similarity(ref, emb, normalized=True)
                        if sim > best_sim:
                            best_sim = sim; best_idx = ri
                    pf["match_score"] = round(float(best_sim), 3)
                    pf["match_ref"] = best_idx
                stats["timings"]["recognition_ms"] = round((time.time() - t) * 1000, 1)

            # --- Face Quality ---
            if opts.get("quality") and face.landmarks is not None and len(face.landmarks) > 0:
                t = time.time()
                try:
                    q = self._ensure("quality").predict(work, face.landmarks)
                    pf["quality"] = round(float(q.score), 3)
                    stats["timings"]["quality_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["quality_err"] = str(e)

            # --- Age / Sex ---
            if opts.get("age_sex"):
                t = time.time()
                try:
                    if opts.get("age_sex_model", "agegender") == "fairface":
                        r = self._ensure("fairface").predict(work, face)
                        pf["sex"] = r.sex; pf["age_group"] = r.age_group; pf["race"] = r.race
                    else:
                        r = self._ensure("age_gender").predict(work, face)
                        pf["sex"] = r.sex; pf["age"] = r.age
                    stats["timings"]["age_sex_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["age_sex_err"] = str(e)

            # --- Emotion ---
            if opts.get("emotion"):
                t = time.time()
                try:
                    r = self._ensure("emotion").predict(work, face)
                    pf["emotion"] = r.emotion
                    pf["emotion_conf"] = round(float(r.confidence), 3)
                    stats["timings"]["emotion_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["emotion_err"] = str(e)

            # --- Face States (göz/gözlük/maske) ---
            if opts.get("face_states"):
                t = time.time()
                try:
                    r = self._ensure("face_states").predict(work, face)
                    pf["eyeglasses"] = r.eyeglasses
                    pf["sunglasses"] = r.sunglasses
                    pf["mask"] = r.mask
                    pf["left_eye_open"] = r.left_eye_open
                    pf["right_eye_open"] = r.right_eye_open
                    stats["timings"]["face_states_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["face_states_err"] = str(e)

            # --- Head Pose ---
            if opts.get("headpose") and crop.size > 0:
                t = time.time()
                try:
                    r = self._ensure("headpose").estimate(crop)
                    pf["pitch"] = round(float(r.pitch), 1)
                    pf["yaw"] = round(float(r.yaw), 1)
                    pf["roll"] = round(float(r.roll), 1)
                    stats["timings"]["headpose_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["headpose_err"] = str(e)

            # --- Gaze ---
            if opts.get("gaze") and crop.size > 0:
                t = time.time()
                try:
                    r = self._ensure("gaze").estimate(crop)
                    pf["gaze_pitch"] = round(float(r.pitch), 1)
                    pf["gaze_yaw"] = round(float(r.yaw), 1)
                    stats["timings"]["gaze_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["gaze_err"] = str(e)

            # --- Anti-spoofing ---
            if opts.get("spoofing"):
                t = time.time()
                try:
                    r = self._ensure("spoofer").predict(work, face.bbox[:4])
                    pf["is_real"] = bool(r.is_real)
                    pf["spoof_conf"] = round(float(r.confidence), 3)
                    stats["timings"]["spoofing_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["spoofing_err"] = str(e)

            # --- Dense landmarks (106) ---
            if opts.get("landmark106"):
                t = time.time()
                try:
                    lm = self._ensure("landmark106").get_landmarks(work, face.bbox[:4])
                    pf["landmark106_n"] = int(len(lm))
                    # çizim
                    for x, y in lm.astype(int):
                        cv2.circle(work, (int(x), int(y)), 2, (0, 200, 255), -1)
                    stats["timings"]["landmark106_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["landmark106_err"] = str(e)

            # --- Face Mesh (3D, 468) ---
            if opts.get("facemesh"):
                t = time.time()
                try:
                    res = self._ensure("facemesh").predict(work, [face])
                    if res:
                        pts = res[0].points_2d
                        pf["mesh_n"] = int(len(pts))
                        draw.draw_mesh(work, pts)
                    stats["timings"]["facemesh_ms"] = round((time.time() - t) * 1000, 1)
                except Exception as e:
                    pf["facemesh_err"] = str(e)

            if face_track_ids is not None and face_track_ids[i] is not None:
                pf["track_id"] = face_track_ids[i]
            else:
                pf["track_id"] = None
            stats["per_face"].append(pf)

        # 4) Görsel etiketler + bbox
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox[:4]]
            pf = stats["per_face"][i] if i < len(stats["per_face"]) else {}
            label_parts = []
            if pf.get("sex") and (pf.get("age_group") or pf.get("age")):
                label_parts.append(f"{pf.get('sex')},{pf.get('age_group') or pf.get('age')}")
            if pf.get("emotion"):
                label_parts.append(pf["emotion"])
            # Yüz durumları (maske/gözlük) artık bbox altında ikonlarla gösteriliyor;
            # üst etikette tekrarlanmasına gerek yok.
            if pf.get("is_real") is not None:
                label_parts.append("GERCEK" if pf["is_real"] else "SAHTE!")
            if pf.get("match_score") is not None:
                # adaptif esigi kucukce goster: eslesme=0.48/0.42 gibi
                _th2 = pf.get("match_threshold", 0.50)
                label_parts.append(f"eslesme={pf['match_score']:.2f}/{_th2:.2f}")
            if pf.get("track_id") is not None:
                label_parts.append(f"#{pf['track_id']}")
            label = " ".join(label_parts)
            # Kutu rengi: aranan kisiyla eslesme (adaptif esik) = MAVI; yoksa cinsiyet: Kadin yesil, Erkek kirmizi
            # 3 nolu optimizasyon: sabit 0.50 yerine kalite+basa gore 0.42-0.58 arasi degisir
            ms = pf.get("match_score")
            _th = _adaptive_match_threshold(pf, base=0.50)
            pf["match_threshold"] = _th
            is_match = ms is not None and ms >= _th
            # etiket icin esigi de goster (debug): pf['match_threshold'] zaten stats'ta olacak
            if is_match:
                box_color = (255, 140, 0)  # BGR parlak mavi
            elif pf.get("sex") == "Male":
                box_color = (0, 0, 255)   # BGR kırmızı
            else:
                box_color = (0, 200, 0)   # BGR yeşil
            cv2.rectangle(work, (x1, y1), (x2, y2), box_color, 3 if is_match else 2)
            if label:
                draw.draw_text_label(work, label, x1, y1 - 4, bg_color=(40, 80, 40), font_scale=0.55)
            # Baş duruşu (x,y,z eksen okları) ve Bakış yönü oku — yüz merkezinden
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            face_w = max(1, x2 - x1)
            if pf.get("pitch") is not None and pf.get("yaw") is not None and pf.get("roll") is not None:
                _draw_pose_axes(work, cx, cy, int(face_w * 0.55),
                                pf["yaw"], pf["pitch"], pf["roll"])
            if pf.get("gaze_pitch") is not None and pf.get("gaze_yaw") is not None:
                _draw_gaze_arrow(work, cx, cy, int(face_w * 0.8),
                                 pf["gaze_yaw"], pf["gaze_pitch"])
            # --- Yüz Durumları: alt alta küçük font, yeşil ✓ / kırmızı ✗ ---
            # cv2 putText Heshey fontu ✓/✗ glifini çizmez; ikonu elle çiziyoruz.
            state_keys = [("eyeglasses", "gozluk"), ("sunglasses", "gunluk"),
                          ("mask", "maske"), ("left_eye_open", "sol goz"),
                          ("right_eye_open", "sag goz")]
            present = [k for k, _ in state_keys if k in pf]
            if present:
                H, W = work.shape[:2]
                panel_h = len(present) * 16 + 6
                # alt taşarsa kutu üstüne, yoksa altına yerleştir
                if y2 + panel_h > H:
                    row = y1 - 10 - 16 * (len(present) - 1)
                else:
                    row = y2 + 14
                for k, lab in state_keys:
                    if k not in pf:
                        continue
                    # FaceAttribNet olasılık (float) döner; 0.5 üstü = var/açık
                    val = bool(pf[k] > 0.5)
                    col = (0, 255, 0) if val else (0, 0, 255)  # BGR: yeşil / kırmızı
                    ix, iy = x1 + 4, row
                    if val:  # ✓ onay (iki çizgi)
                        cv2.line(work, (ix, iy), (ix + 4, iy + 4), col, 2)
                        cv2.line(work, (ix + 4, iy + 4), (ix + 10, iy - 4), col, 2)
                    else:    # ✗ çarpı (iki çapraz çizgi)
                        cv2.line(work, (ix, iy - 5), (ix + 10, iy + 5), col, 2)
                        cv2.line(work, (ix, iy + 5), (ix + 10, iy - 5), col, 2)
                    cv2.putText(work, lab, (ix + 16, iy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                    row += 16

        # 5) Parsing overlay (opsiyonel, görüntü geneli)
        if opts.get("parsing") and faces:
            try:
                parser = self._ensure("parser")
                # ilk yüzün bbox'ını büyüterek yüz bölgesi
                fx1, fy1, fx2, fy2 = [int(v) for v in faces[0].bbox[:4]]
                pad = 60
                fx1, fy1 = max(0, fx1 - pad), max(0, fy1 - pad)
                fx2, fy2 = min(work.shape[1], fx2 + pad), min(work.shape[0], fy2 + pad)
                crop = work[fy1:fy2, fx1:fx2]
                if crop.size > 0:
                    mask = parser.parse(crop)
                    vis = draw.vis_parsing_maps(crop, mask)
                    work[fy1:fy2, fx1:fx2] = vis
                    stats["timings"]["parsing_ms"] = 1
            except Exception as e:
                stats["parsing_err"] = str(e)

        # 6) Anonimleştirme: yüz bölgelerini bulanıklaştır (resim + video + MP4 her yolunda)
        if opts.get("blur"):
            try:
                self.blur.anonymize(work, faces, inplace=True)
                stats["blurred"] = True
            except Exception as e:
                stats["blur_err"] = str(e)

        return work, stats

    # ---- Anonimleştirme (ayrı, resmi değiştirir) ----
    def anonymize(self, image_bgr):
        faces = self.detector.detect(image_bgr)
        if faces:
            self.blur.anonymize(image_bgr, faces, inplace=True)
        return image_bgr


def make_engine():
    return UnifaceEngine(device="cuda")


if __name__ == "__main__":
    eng = make_engine()
    print("Engine hazir. Modeller yükleniyor...")
    eng._ensure("age_gender")
    print("age_gender", eng.timer)