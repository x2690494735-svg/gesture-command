# -*- coding: utf-8 -*-
"""隔空手势 v7 · 手势+语音 双通道（全离线、本地、免费）。
本轮（依据 MediaPipe 社区/专家避坑指南 + 用户实测误判复盘）：
  1. 手指伸展判定改为【方向向量】法（指尖-近节 在 近节-掌心 方向上才叫伸）——比绝对距离 1.30 倍数
     对姿态/镜头角度天然鲁棒，解决"V 被算成 3 / 食指被算 2"的偏移。
  2. OK(捏合)判定双条件：拇-食指尖 < 0.05 且 食-中指尖 > 2.5×拇-食距——"撩头发被识别成继续做"的
     主因是单条件距离，发丝/并拢手指也会贴近；加中指距离后只有真捏合才成立。
  3. 关键点时间平滑（近 3 帧平均）+ 丢手宽容期（8 帧内不清投票窗，快速手势中途遮挡不失效）。
  4. 检测置信 0.65→0.7（降假检）；投票 4/6；冷却 1.2s。
  5. 实时判定写入 工具/gesture-debug.csv（环形 3000 行）——每次误判后我有数据可查，不用"光靠眼。
用法: python gesture-command.py [camera_index=0]   (Esc 退出; 窗口聚焦时【空格】=语音)
"""
import sys, os, time, ctypes
from collections import deque, Counter
import numpy as np
import cv2
sys.path.insert(0, r"D:\mp")   # ASCII 包：绕开中文用户名让 mediapipe C++ 找不到资源
import mediapipe as mp
import pyperclip

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
COOLDOWN = 1.2
OK_D = 0.075         # OK: 拇-食指尖近(留噪声余量,真捏合≥0.065 仍过)
OK_SPREAD = 2.5      # OK: 食-中指尖 ≥ 2.5×拇-食距（防"数指"误判）
VOTE_N = 4           # 窗内多数票
BOX = (0.28, 0.72, 0.20, 0.80)
DEBUG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture-debug.csv")

MAPPING = {5: "可以", 0: "不行，改一下", "ok": "继续做，别停", 1: "停一下", 2: "再来一个"}
COLORS = {5: (0, 255, 0), 0: (0, 0, 255), "ok": (0, 255, 255), 1: (255, 200, 0), 2: (255, 160, 0)}

# ---------- 特征与识别：模型优先，规则兜底（特征唯一来源=本文件，训练/测试脚本 head-exec 同源提取） ----------
def to_features(hand):
    lm = np.array([[p.x, p.y] for p in hand.landmark])
    lm = lm - lm[0]                       # 腕为原点
    span = max(np.linalg.norm(lm[i]) for i in range(21)) + 1e-9
    lm = lm / span                        # 跨度归一
    palm = np.mean([lm[i] for i in (0, 1, 2, 5, 9, 13, 17)], axis=0)
    d48 = float(np.linalg.norm(lm[4] - lm[8]))
    d812 = float(np.linalg.norm(lm[8] - lm[12]))
    thumb_out = float(np.linalg.norm(lm[4] - palm))
    return list(lm.flatten()) + [d48, d812, thumb_out, float(palm[0]), float(palm[1])]


_MODEL = None
MODEL_LABEL = {"open": 5, "fist": 0, "one": 1, "v": 2, "ok": "ok"}
try:
    import joblib
    _mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture-model.pkl")
    if os.path.exists(_mp):
        _MODEL = joblib.load(_mp)
except Exception as _e:
    pass


# ---------- 手部特征（方向向量法，姿态不敏感；模型出现后作为 rules-first 兜底） ----------
def finger_stats(hand):
    lm = [(p.x, p.y) for p in hand.landmark]
    palm = np.mean([lm[i] for i in (0, 1, 2, 5, 9, 13, 17)], axis=0)
    stats = {}
    names = {8: "idx", 12: "mid", 16: "ring", 20: "pinky"}
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        t = np.array(lm[tip]); p = np.array(lm[pip])
        v1 = p - palm                     # 指根相对掌心
        v2 = t - p                        # 指节伸出方向
        v1n, v2n = np.linalg.norm(v1), np.linalg.norm(v2)
        cos = float(np.dot(v1, v2) / (v1n * v2n + 1e-9))
        ext = cos > 0.55 and v2n > v1n * 0.45   # 伸开=沿指根方向且长度足够
        stats[names[tip]] = (float(np.linalg.norm(t - p)), cos)
    d48 = float(np.linalg.norm(np.array(lm[4]) - np.array(lm[8])))
    d812 = float(np.linalg.norm(np.array(lm[8]) - np.array(lm[12])))
    thumb_out = float(np.linalg.norm(np.array(lm[4]) - palm))
    palm_ok = np.linalg.norm(palm - np.array(lm[9])) < 0.35  # 置信粗筛：掌心轮廓正常
    return stats, d48, d812, thumb_out, lm, palm, palm_ok


def rules_classify(hand):
    """规则链：方向向量法 + OK 双条件（主判定，稳定不摇摆）。"""
    stats, d48, d812, thumb_out, lm, palm, palm_ok = finger_stats(hand)
    if not palm_ok:
        return None, "no-palm"
    ext = [stats[k][1] > 0.55 and stats[k][0] > 0.0 for k in ("idx", "mid", "ring", "pinky")]
    n = sum(1 for e in ext if e)
    if d48 < OK_D and d812 > OK_D * OK_SPREAD:
        return "ok", "OK捏合"
    if n == 4 and thumb_out > 0.15:
        return 5, "张掌"
    if n == 0:
        return 0, "握拳"
    if n == 1:
        return 1, "食指"
    if n == 2:
        return 2, "V"
    return None, f"n={n}"


def classify(hand):
    """主判定=规则链；模型仅在规则空手且高置信(≥0.72)时补位（避免单帧摇摆/过收 ok）。"""
    global _MODEL
    rk, rdesc = rules_classify(hand)
    if _MODEL is not None:
        try:
            feats = np.array([to_features(hand)], dtype=float)
            proba = _MODEL.predict_proba(feats)[0]
            i = int(np.argmax(proba))
            label = _MODEL.classes_[i]
            mk = MODEL_LABEL.get(label)
            if rk is None and proba[i] >= 0.72 and mk is not None:
                return mk, f"model:{label}={proba[i]:.2f}"
        except Exception:
            pass
    return rk, rdesc


# ---------- 官方 GestureRecognizer（主判）：真实手型分类器 ----------
_GR = None
CLS_MAP = {"Open_Palm": 5, "Closed_Fist": 0, "Pointing_Up": 1, "Victory": 2,
           "Thumb_Up": "ok", "ILoveYou": None, "None": None}
_MP_OLD = None


def recognize_vision(frame_bgr):
    """返回 (hand, (key, desc))；无手/模型失败 → (None, (None, 原因))。
    hand 是含 .landmark(21点, 含 x/y/z) 的轻量对象，供绘制与框判定用。"""
    global _GR, _MP_OLD
    try:
        if _GR is None:
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
            opts = vision.GestureRecognizerOptions(
                base_options=mpt.BaseOptions(model_asset_path=r"D:\mp\models\gesture_recognizer.task"),
                num_hands=1,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            _GR = vision.GestureRecognizer.create_from_options(opts)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = _GR.recognize(mimg)
        if res.gestures and res.gestures[0]:
            g = res.gestures[0][0]
            k = CLS_MAP.get(g.category_name)
            if res.hand_landmarks:
                return (type("H", (), {"landmark": res.hand_landmarks[0]})(),
                        (k, f"{g.category_name}={g.score:.2f}"))
    except Exception as e:
        # 回退：ASCII 副本里的 0.10.13 solutions（保留手绘制与规则分类兜底）
        try:
            if _MP_OLD is None:
                import importlib
                sys.path.insert(0, r"D:\mp")
                _MP_OLD = importlib.import_module("mediapipe")
            sol = _MP_OLD.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                                min_detection_confidence=0.65, min_tracking_confidence=0.5)
            res = sol.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                k, d = rules_classify(hand)
                return hand, (k, f"rules:{d}")
        except Exception:
            pass
        return None
    return None


# ---------- 窗口/输入 ----------
def activate_cc():
    u = ctypes.windll.user32
    # 必需：声明 argtypes，否则 EnumWindows 回调转换失败 -> 一确认就崩溃（2026-08-25 实测根因）
    PROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    u.EnumWindows.argtypes = [PROC, ctypes.c_void_p]
    u.EnumWindows.restype = ctypes.c_bool
    claude_hits, term_hits = [], []
    CALLBACK = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        if u.IsWindowVisible(hwnd):
            ln = ctypes.create_unicode_buffer(256)
            u.GetWindowTextW(hwnd, ln, 256)
            t = ln.value.lower()
            if 'claude' in t or 'codex' in t:
                claude_hits.append(hwnd)
            elif 'powershell' in t or 'pwsh' in t or 'terminal' in t or 'cmd' in t:
                term_hits.append(hwnd)
        return True
    u.EnumWindows(cb, 0)
    for hw in list(claude_hits + term_hits)[::-1]:
        if hw:
            u.SetForegroundWindow(hw); time.sleep(0.25); return True
    return False


def send(text):
    pyperclip.copy(text)
    u = ctypes.windll.user32
    try:
        activate_cc()
    except Exception:
        pass     # 激活窗口失败不致命：粘贴给当前焦点窗口
    u.keybd_event(0x11, 0, 0, 0); u.keybd_event(0x56, 0, 0, 0)
    u.keybd_event(0x56, 0, 2, 0); u.keybd_event(0x11, 0, 2, 0)


class Recorder:
    def __init__(self, out, sr=16000):
        import wave as _w
        self.out, self.sr, self._w = out, sr, _w
        self._q, self._stop, self._t = [], None, None
    def _run(self):
        import sounddevice as sd
        def cb(indata, frames, t, status):
            if not self._stop.is_set():
                self._q.append(indata.copy())
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="int16", callback=cb) as st:
            with self._w.open(self.out, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.sr)
                while not self._stop.is_set():
                    if self._q:
                        w.writeframes(b"".join(a.tobytes() for a in self._q)); self._q.clear()
                    time.sleep(0.15)
                w.writeframes(b"".join(a.tobytes() for a in self._q))
    def start(self):
        import threading as _t
        self._stop = _t.Event()
        self._t = _t.Thread(target=self._run, daemon=True); self._t.start()
    def stop(self):
        self._stop.set(); self._t.join(timeout=3)


def transcribe(wav):
    import subprocess, tempfile, glob
    clean = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
                    "-af", "afftdn,highpass=f=200,lowpass=f=8200",
                    "-ar", "16000", "-ac", "1", clean], check=True)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from faster_whisper import WhisperModel
    path = sorted(glob.glob(os.path.expanduser(
        "~/.cache/modelscope/models/AI-ModelScope--faster-whisper-small/snapshots/*/")))
    model = WhisperModel(path[0] if path else "small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(clean, language="zh", vad_filter=True, beam_size=1)
    text = "".join(s.text for s in segs).strip()
    os.unlink(clean) if os.path.exists(clean) else None
    return text


def log_csv(line):
    try:
        with open(DEBUG_CSV, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


print("摄像头启动中……", flush=True)
cap = cv2.VideoCapture(CAM)
if not cap.isOpened():
    print(f"[!] 摄像头 {CAM} 打不开 —— 换个索引或检查设备"); sys.exit(1)
cap.set(3, 1280); cap.set(4, 720)
W, H = 1280, 720

state, done_at, sent_val = 0, 0.0, ""
history = []
GK = deque(maxlen=6)          # 投票窗
LM_AVG = deque(maxlen=3)      # 手部关键点平滑
lost_frames = 0               # 丢手宽容期
is_speaking = False
rec_obj, spoke_at = None, 0.0
wav = os.path.join(os.path.expanduser("~"), "ds-gesture.wav")

WIN = "DS 隔空手势 v7 · 手势+语音"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, W, H)
hwnd = ctypes.windll.user32.FindWindowW(None, WIN)
if hwnd:
    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)

SKIP = 5          # 每 5 帧识别一次（轻量档：显存/CPU 减负，手势保持判定不受影响）
frame_i = 0
_last_lm21 = None
while True:
    ok, frame = cap.read()
    if not ok:
        # 摄像头被其他应用(如飞书)抢占 -> 重连而非退出（修复"关飞书它也关"）
        cap.release()
        time.sleep(0.8)
        cap = cv2.VideoCapture(CAM)
        continue
    frame = cv2.flip(frame, 1)
    frame_i += 1

    # 有效范围框
    bx1, bx2, by1, by2 = int(BOX[0] * W), int(BOX[1] * W), int(BOX[2] * H), int(BOX[3] * H)
    for i in range(bx1, bx2, 26):
        cv2.line(frame, (i, by1), (i + 14, by1), (255, 120, 0), 2)
        cv2.line(frame, (i, by2), (i + 14, by2), (255, 120, 0), 2)
    for i in range(by1, by2, 26):
        cv2.line(frame, (bx1, i), (bx1, i + 14), (255, 120, 0), 2)
        cv2.line(frame, (bx2, i), (bx2, i + 14), (255, 120, 0), 2)
    cv2.putText(frame, "有效范围：把手放这里", (bx1 + 12, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

    # 识别：官方 GestureRecognizer（真实手型模型），失败回退规则链；隔 SKIP 帧轻量跑
    gesture, in_box, dbg = None, None, ""
    lm21 = recognize_vision(frame) if frame_i % SKIP == 1 else _last_lm21
    _last_lm21 = lm21
    if lm21 is not None:
        hand, vdesc = lm21
        xs = [p.x for p in hand.landmark]; ys = [p.y for p in hand.landmark]
        in_box = bx1 / W < min(xs) < max(xs) < bx2 / W and by1 / H < min(ys) < max(ys) < by2 / H
        col = (0, 255, 0) if in_box else (0, 0, 255)
        for p in hand.landmark:
            cv2.circle(frame, (int(p.x * W), int(p.y * H)), 3, (0, 200, 0), -1)
        cv2.rectangle(frame, (int(min(xs) * W), int(min(ys) * H)),
                      (int(max(xs) * W), int(max(ys) * H)), col, 2)
        cv2.putText(frame, "IN" if in_box else "OUT", (int(max(xs) * W) + 6, int(min(ys) * H) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        gesture, dbg_check = vdesc
        dbg = dbg_check
        lost_frames = 0
    else:
        lost_frames += 1
        if lost_frames <= 8:      # 宽容期：不清投票窗
            in_box = False
        else:
            GK.clear()

    # 底部横幅
    cv2.rectangle(frame, (0, H - 92), (W, H), (18, 18, 18), -1)
    if is_speaking:
        msg, mcol = "LINE 语音中… 再按【空格】结束并转写", (0, 0, 255)
    elif state == 1:
        msg, mcol = f"✓ 已输入「{sent_val}」 (【空格】语音派活)", (0, 255, 0)
    elif gesture is not None:
        k = MAPPING.get(gesture)
        if k:
            msg, mcol = f"手势:「{k}」 保持（多数票≥{VOTE_N}）确认", COLORS.get(gesture, (255, 255, 255))
        else:
            msg, mcol = f"手势: {dbg} 无映射", (180, 180, 180)
    else:
        msg, mcol = "v7 等待 · 张掌=可以 握拳=不行改一下 OK=继续做 食指=停一下 V=再来一个 · 【空格】说话", (80, 255, 80)
    cv2.putText(frame, msg, (20, H - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, mcol, 2)
    if dbg:
        cv2.putText(frame, dbg[:46], (20, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 历史
    for i, txt in enumerate(history[-4:]):
        yy = 70 + i * 30
        cv2.rectangle(frame, (W - 340, yy - 22), (W - 12, yy + 6), (28, 28, 28), -1)
        cv2.putText(frame, ("#" + str(len(history) - 4 + i + 1) + " " + txt[:18]),
                    (W - 328, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # 手势状态机（多数票）
    now = time.time()
    if state != 0:
        if now - done_at > COOLDOWN:
            state = 0
    elif gesture is not None:
        if in_box:
            GK.append(gesture)
            major, vote = Counter(GK).most_common(1)[0]
            if major in MAPPING and vote >= VOTE_N:
                state = 1; sent_val = MAPPING[major]; done_at = now
                send(sent_val); history.append(sent_val); GK.clear()
                log_csv(f"{time.strftime('%H:%M:%S')},{major},{dbg}")
        else:
            GK.clear()

    # 语音（空格 + 20s 超时收尾）
    if rec_obj is not None and now - spoke_at > 20:
        is_speaking = False; rec_obj.stop(); rec_obj = None
        try:
            txt = transcribe(wav)
            if txt:
                send(txt); history.append(txt); state = 1; sent_val = txt; done_at = now
        except Exception as e:
            print("[!] 转写失败:", e)

    cv2.imshow(WIN, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 32:
        if is_speaking:
            is_speaking = False
            if rec_obj:
                rec_obj.stop(); rec_obj = None
                try:
                    txt = transcribe(wav)
                    if txt:
                        send(txt); history.append(txt); state = 1; sent_val = txt; done_at = now
                except Exception as e:
                    print("[!] 转写失败:", e)
        else:
            is_speaking = True
            rec_obj = Recorder(wav); rec_obj.start(); spoke_at = now

cap.release(); cv2.destroyAllWindows()
