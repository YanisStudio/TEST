"""
楓之谷 測謊器 全自動辨識
依賴：pip install mss pyautogui ddddocr Pillow opencv-python
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import io
import datetime

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw
import pyautogui
import mss
import ddddocr

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.05

OCR = ddddocr.DdddOcr(show_ad=False)


# ─── 偵測：藍色計時器 ────────────────────────────────────────────────────────

def find_timer(bgr):
    """
    找計時器的亮藍色數字區域。
    回傳 (x, y, w, h) 或 None。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 亮青藍色範圍（OpenCV H = 0~179）
    lower = np.array([85,  150, 150])
    upper = np.array([105, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # 膨脹讓分散的數字連起來
    kernel  = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 計時器：寬 > 高，寬度合理（40~300px）
        if 40 < w < 300 and 10 < h < 80 and w > h * 1.5:
            return x, y, w, h
    return None


# ─── 偵測：字元圖片框 ────────────────────────────────────────────────────────

def find_captcha_boxes(bgr, min_size=40, max_size=180):
    """
    在圖片中找高雜訊小方形（測謊字元圖片的特徵）。
    回傳 [(x, y, w, h, box_bgr), ...]，依 x 排序。
    """
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 25, 90)
    k     = np.ones((4, 4), np.uint8)
    dilated = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_size < w < max_size and min_size < h < max_size):
            continue
        if not (0.5 < w / h < 2.0):
            continue
        roi = gray[y:y+h, x:x+w]
        if roi.std() < 15:           # 雜訊夠高才算字元圖片
            continue
        candidates.append((x, y, w, h, bgr[y:y+h, x:x+w].copy()))

    # 去重疊
    candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
    kept = []
    for c in candidates:
        cx, cy, cw, ch, _ = c
        overlap = any(
            max(cx, kx) < min(cx+cw, kx+kw) and max(cy, ky) < min(cy+ch, ky+kh)
            for kx, ky, kw, kh, _ in kept
        )
        if not overlap:
            kept.append(c)

    kept.sort(key=lambda c: c[0])
    return kept


# ─── OCR 工具 ────────────────────────────────────────────────────────────────

def _to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    return buf.getvalue()


def _ocr_best(pil_img):
    """多種預處理輪流嘗試，回傳最佳結果。"""
    w, h = pil_img.size
    attempts = [pil_img]
    # 放大 3x
    attempts.append(pil_img.resize((w*3, h*3), Image.LANCZOS))
    # 灰階 + 對比
    gray = pil_img.convert("L")
    enh  = ImageEnhance.Contrast(gray).enhance(3.0)
    attempts.append(enh.resize((w*3, h*3), Image.LANCZOS))
    # 銳化
    attempts.append(enh.resize((w*3, h*3), Image.LANCZOS).filter(ImageFilter.SHARPEN))

    for img in attempts:
        r = OCR.classification(_to_bytes(img)).strip()
        if r:
            return r
    return "?"


def _find_corner_digit(pil_img):
    """四角找小數字，回傳 (digit, corner_name)。"""
    w, h = pil_img.size
    cw, ch = max(w//3, 12), max(h//3, 12)
    corners = {
        "TL": (0,    0,    cw,   ch),
        "TR": (w-cw, 0,    w,    ch),
        "BL": (0,    h-ch, cw,   h),
        "BR": (w-cw, h-ch, w,    h),
    }
    for name, box in corners.items():
        crop = pil_img.crop(box)
        big  = crop.resize((crop.width*5, crop.height*5), Image.LANCZOS)
        txt  = OCR.classification(_to_bytes(big)).strip()
        for c in txt:
            if c.isdigit() and c != "0":
                return c, name
    return "?", "?"


def recognize_box(box_bgr):
    """回傳 (order_str, char_str)。"""
    pil = Image.fromarray(cv2.cvtColor(box_bgr, cv2.COLOR_BGR2RGB))
    w, h = pil.size

    order, corner = _find_corner_digit(pil)

    # 遮蓋角落數字
    cw, ch = max(w//3, 12), max(h//3, 12)
    mask_map = {
        "TL": (0,    0,    cw,   ch),
        "TR": (w-cw, 0,    w,    ch),
        "BL": (0,    h-ch, cw,   h),
        "BR": (w-cw, h-ch, w,    h),
    }
    if corner in mask_map:
        pil = pil.copy()
        draw = ImageDraw.Draw(pil)
        draw.rectangle(mask_map[corner], fill=(255, 255, 255))

    char = _ocr_best(pil)
    return order, char


# ─── UI ──────────────────────────────────────────────────────────────────────

class CaptchaAutoApp:

    def __init__(self, root):
        self.root    = root
        self.root.title("測謊器全自動辨識")
        self.root.geometry("560x500")
        self.root.resizable(True, True)
        self.running      = False
        self._last_solved = 0
        self._build_ui()

    def _build_ui(self):
        ttk.Label(self.root, text="測謊器全自動辨識",
                  font=("Microsoft JhengHei UI", 14, "bold")).pack(pady=8)

        self.status_var = tk.StringVar(value="按「開始監控」即可，不需要任何設定")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="#7f8c8d").pack()

        # 控制列
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=12, pady=8)
        self.start_btn = ttk.Button(ctrl, text="▶ 開始監控",
                                    command=self._toggle, width=16)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="間隔(秒):").pack(side=tk.LEFT, padx=(12, 2))
        self.interval_var = tk.DoubleVar(value=0.4)
        ttk.Spinbox(ctrl, from_=0.1, to=2.0, increment=0.1,
                    textvariable=self.interval_var, width=5).pack(side=tk.LEFT)

        ttk.Label(ctrl, text="冷卻(秒):").pack(side=tk.LEFT, padx=(12, 2))
        self.cooldown_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(ctrl, from_=1.0, to=15.0, increment=0.5,
                    textvariable=self.cooldown_var, width=5).pack(side=tk.LEFT)

        # 預覽
        preview_frame = ttk.LabelFrame(self.root, text="偵測預覽（綠框=計時器  紅框=字元圖片）", padding=4)
        preview_frame.pack(fill=tk.X, padx=12, pady=4)
        self.preview_lbl = ttk.Label(preview_frame, text="監控開始後顯示")
        self.preview_lbl.pack()

        # Log
        log_frame = ttk.LabelFrame(self.root, text="執行記錄", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=10,
                                                  font=("Consolas", 9),
                                                  state="disabled")
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.tag_config("OK",   foreground="#27ae60")
        self.log_box.tag_config("WARN", foreground="#e67e22")
        self.log_box.tag_config("ERR",  foreground="#e74c3c")

        ttk.Label(self.root,
                  text="緊急停止：滑鼠移到螢幕左上角",
                  foreground="#e74c3c",
                  font=("Microsoft JhengHei UI", 8)).pack(pady=2)

    # ── 監控 ──────────────────────────────────────────────────────────────────

    def _toggle(self):
        if not self.running:
            self.running = True
            self.start_btn.config(text="■ 停止監控")
            self.status_var.set("監控中… 等待測謊出現")
            threading.Thread(target=self._loop, daemon=True).start()
        else:
            self.running = False
            self.start_btn.config(text="▶ 開始監控")
            self.status_var.set("已停止")

    def _loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[0]   # 全螢幕
            while self.running:
                try:
                    raw = sct.grab(monitor)
                    bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

                    timer = find_timer(bgr)

                    boxes = []
                    if timer:
                        # 只在計時器下方 ~300px 的區域內找字元圖片
                        tx, ty, tw, th = timer
                        H, W = bgr.shape[:2]
                        rx  = max(tx - 150, 0)
                        ry  = ty              # 從計時器頂部開始
                        rx2 = min(tx + tw + 150, W)
                        ry2 = min(ty + 350, H)  # 往下最多 350px
                        roi = bgr[ry:ry2, rx:rx2]
                        raw_boxes = find_captcha_boxes(roi)
                        # 座標轉換回全螢幕
                        boxes = [(rx+bx, ry+by, bw, bh, bimg)
                                 for bx, by, bw, bh, bimg in raw_boxes]

                    # 預覽更新
                    self._update_preview(bgr, timer, boxes)

                    # 冷卻中跳過
                    if time.time() - self._last_solved < self.cooldown_var.get():
                        time.sleep(self.interval_var.get())
                        continue

                    if not timer:
                        time.sleep(self.interval_var.get())
                        continue

                    if not boxes:
                        self.log("偵測到計時器，但找不到字元圖片", "WARN")
                        time.sleep(self.interval_var.get())
                        continue

                    # ── 辨識 ──
                    self.log(f"✓ 測謊偵測到！{len(boxes)} 個字元圖片", "OK")
                    results = []
                    for bx, by, bw, bh, box_img in boxes:
                        order, char = recognize_box(box_img)
                        self.log(f"   [{bx},{by}] 順序={order}  字元={char}")
                        results.append((order, char))

                    results.sort(key=lambda r: int(r[0]) if r[0].isdigit() else 99)
                    answer = "".join(c for _, c in results if c not in ("?", ""))

                    if not answer:
                        self.log("辨識失敗，跳過", "WARN")
                        time.sleep(self.interval_var.get())
                        continue

                    self.log(f"→ 輸入答案: {answer}", "OK")
                    self.root.after(0, lambda a=answer: self.status_var.set(f"輸入: {a}"))

                    pyautogui.typewrite(answer, interval=0.08)
                    pyautogui.press("enter")
                    self._last_solved = time.time()

                except Exception as e:
                    self.log(f"錯誤: {e}", "ERR")

                time.sleep(self.interval_var.get())

    # ── 預覽 ──────────────────────────────────────────────────────────────────

    def _update_preview(self, bgr, timer, boxes):
        h, w = bgr.shape[:2]
        scale = min(520 / w, 120 / h)
        small = cv2.resize(bgr, (int(w*scale), int(h*scale)))

        if timer:
            tx, ty, tw, th = timer
            cv2.rectangle(small,
                          (int(tx*scale), int(ty*scale)),
                          (int((tx+tw)*scale), int((ty+th)*scale)),
                          (0, 255, 0), 2)

        for bx, by, bw, bh, _ in boxes:
            cv2.rectangle(small,
                          (int(bx*scale), int(by*scale)),
                          (int((bx+bw)*scale), int((by+bh)*scale)),
                          (0, 80, 255), 2)

        pil    = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        tk_img = ImageTk.PhotoImage(pil)
        self.root.after(0, lambda img=tk_img: self._set_preview(img))

    def _set_preview(self, tk_img):
        self.preview_lbl.config(image=tk_img, text="")
        self.preview_lbl.image = tk_img

    # ── Log ───────────────────────────────────────────────────────────────────

    def log(self, msg, tag=""):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        def _w():
            self.log_box.config(state="normal")
            self.log_box.insert(tk.END, f"[{ts}] {msg}\n", tag)
            self.log_box.see(tk.END)
            self.log_box.config(state="disabled")
        self.root.after(0, _w)


if __name__ == "__main__":
    root = tk.Tk()
    CaptchaAutoApp(root)
    root.mainloop()
