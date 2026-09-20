# -*- coding: utf-8 -*-
"""隔空手势语音输入 v1 —— 摄像头 MediaPipe 手部识别（全离线、本地）：
   五指张开 = 开始说话（录麦克风）
   握拳     = 说完了（降噪+Whisper 转写 → 剪贴板 → 自动 Ctrl+V 进 Claude Code）
用法: python gesture-voice.py [camera_index=0]   (Esc 退出)
"""
import sys, os, time, wave, ctypes, subprocess, tempfile, threading
import numpy as np
import cv2
import mediapipe as mp
import pyperclip

DIR = os.path.dirname(os.path.abspath(__file__))
CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ---------- 录音（后台线程，增量写 wav，停即结） ----------
class Recorder:
    def __init__(self, out, sr=16000):
        self.out, self.sr = out, sr
        self._q, self._stop, self._t = [], None, None
    def _run(self):
        import sounddevice as sd
        def cb(indata, frames, t, status):
            if not self._stop.is_set(): self._q.append(indata.copy())
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="int16", callback=cb) as st:
            with wave.open(self.out, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.sr)
                while not self._stop.is_set():
                    if self._q:
                        w.writeframes(b"".join(a.tobytes() for a in self._q)); self._q.clear()
                    time.sleep(0.15)
                w.writeframes(b"".join(a.tobytes() for a in self._q))
    def start(self):
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
    def stop(self):
        self._stop.set(); self._t.join(timeout=3)

# ---------- 手势度量：4 指尖伸展度（张掌大/握拳小） ----------
def open_score(hand):
    lm = [(p.x, p.y) for p in hand.landmark]
    palm = np.mean([lm[i] for i in (0, 1, 2, 3, 4, 5, 6, 9, 13, 17)], axis=0)
    vals = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        d1 = np.linalg.norm(np.array(lm[tip]) - np.array(lm[pip]))
        d2 = np.linalg.norm(np.array(lm[tip]) - palm)
        vals.append((d1 + d2) / 2)
    return float(np.mean(vals))

# ---------- 转写（ffmpeg 降噪 + faster-whisper） ----------
def transcribe(wav) -> str:
    clean = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav,
                    "-af", "afftdn,highpass=f=200,lowpass=f=8200",
                    "-ar", "16000", "-ac", "1", clean], check=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from faster_whisper import WhisperModel
    path = None
    try:
        from modelscope import snapshot_download
        path = snapshot_download("AI-ModelScope/faster-whisper-small")
    except Exception:
        pass
    model = WhisperModel(path or "small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(clean, language="zh", vad_filter=True, beam_size=1)
    text = "".join(s.text for s in segs).strip()
    os.unlink(clean) if os.path.exists(clean) else None
    return text

def ctrl_v():
    u = ctypes.windll.user32
    u.keybd_event(0x11, 0, 0, 0); u.keybd_event(0x56, 0, 0, 0)
    u.keybd_event(0x56, 0, 2, 0); u.keybd_event(0x11, 0, 2, 0)

# ---------- 主循环 ----------
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                    min_detection_confidence=0.65, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(CAM)
if not cap.isOpened():
    print(f"[!] 摄像头 {CAM} 打不开 —— 试试其他索引或检查设备"); sys.exit(1)
cap.set(3, 640); cap.set(4, 480)

LABEL_Z = {"idle": "> 五指张开：开始说话", "rec": "> 说话中… 握拳结束", "proc": "> 转写中，稍等…", "done": "> 已粘贴，按回车发送"}
state = "idle"
open_cnt = close_cnt = 0
STALL = 6            # 连续 6 帧确认（防抖）
MAX_REC = 20.0       # 单次录音最长 20s
rec_started = 0.0
done_at = 0.0
rec = None
wav = os.path.join(tempfile.gettempdir(), "ds-gesture.wav")

def set_status(s):
    global state
    state = s

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    gesture = None
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        for p in hand.landmark:
            cv2.circle(frame, (int(p.x * 640), int(p.y * 480)), 3, (0, 255, 0), -1)
        score = open_score(hand)
        gesture = "open" if score > 0.30 else "closed"   # 阈值实测可调
    cv2.rectangle(frame, (0, 0), (640, 56), (20, 20, 20), -1)
    cv2.putText(frame, LABEL_Z[state], (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255) if state == "rec" else (80, 255, 80), 2)

    now = time.time()
    if state == "idle":
        open_cnt = open_cnt + 1 if gesture == "open" else 0
        if open_cnt >= STALL:
            open_cnt = 0
            rec = Recorder(wav); rec.start()
            rec_started = now
            set_status("rec")
    elif state == "rec":
        if gesture == "closed":
            close_cnt += 1
            if close_cnt >= STALL:
                close_cnt = 0
                rec.stop(); rec = None
                set_status("proc")
                try:
                    text = transcribe(wav)
                    if text:
                        pyperclip.copy(text); ctrl_v(); done_at = now; set_status("done")
                    else:
                        set_status("idle")
                except Exception as e:
                    print("[!] 转写失败:", e); set_status("idle")
        else:
            close_cnt = 0
        if now - rec_started > MAX_REC:   # 超时自动结
            rec.stop(); rec = None
            set_status("proc")
            try:
                text = transcribe(wav)
                if text:
                    pyperclip.copy(text); ctrl_v(); done_at = now; set_status("done")
                else:
                    set_status("idle")
            except Exception as e:
                print("[!] 转写失败:", e); set_status("idle")
    elif state == "done":
        open_cnt = close_cnt = 0
        if now - done_at > 2.5:
            set_status("idle")

    cv2.imshow("DS 隔空语音 · Esc 退出", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

if rec: rec.stop()
cap.release(); cv2.destroyAllWindows()
