"""
驗證碼辨識工具
依賴：pip install ddddocr Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import io

from PIL import Image, ImageTk, ImageGrab, ImageEnhance, ImageFilter
import ddddocr


OCR = ddddocr.DdddOcr(show_ad=False)


def _to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    return buf.getvalue()


def _ocr_with_fallback(img_bytes):
    """嘗試原圖 + 多種預處理，回傳最佳結果（不含空字串）。"""
    def _try(b):
        return OCR.classification(b).strip()

    result = _try(img_bytes)
    if result:
        return result

    orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = orig.size

    variants = []

    # 放大 3x
    big = orig.resize((w * 3, h * 3), Image.LANCZOS)
    variants.append(_to_bytes(big))

    # 灰階 + 對比增強 + 放大
    gray = orig.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(3.0)
    big_gray = enhanced.resize((w * 3, h * 3), Image.LANCZOS)
    variants.append(_to_bytes(big_gray))

    # 銳化
    sharp = big_gray.filter(ImageFilter.SHARPEN)
    variants.append(_to_bytes(sharp))

    # 二值化
    bw = gray.point(lambda x: 255 if x > 120 else 0).convert("RGB")
    big_bw = bw.resize((w * 3, h * 3), Image.LANCZOS)
    variants.append(_to_bytes(big_bw))

    for v in variants:
        r = _try(v)
        if r:
            return r

    return "?"


def _find_corner_number(img_bytes):
    """
    在圖片四個角落裁切，找出單一數字（1~9）作為輸入順序。
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    cw = max(w // 3, 15)
    ch = max(h // 3, 15)

    corners = {
        "TL": img.crop((0,    0,    cw,   ch)),
        "TR": img.crop((w-cw, 0,    w,    ch)),
        "BL": img.crop((0,    h-ch, cw,   h)),
        "BR": img.crop((w-cw, h-ch, w,    h)),
    }

    for pos, corner in corners.items():
        # 白底 + 放大 5x 讓小數字更清晰
        bg = Image.new("RGB", (corner.width * 5, corner.height * 5), (255, 255, 255))
        big = corner.resize((corner.width * 5, corner.height * 5), Image.LANCZOS)
        bg.paste(big, (0, 0))
        result = OCR.classification(_to_bytes(bg)).strip()
        for ch_char in result:
            if ch_char.isdigit() and ch_char != "0":
                return ch_char, pos

    return "?", "?"


def _recognize_image(img_bytes):
    """回傳 (order_str, char_str)"""
    order, corner_pos = _find_corner_number(img_bytes)

    # 裁掉角落數字區域再辨識主字元（避免干擾）
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    cw = max(w // 3, 15)
    ch = max(h // 3, 15)
    mask_areas = {
        "TL": (0,    0,    cw,   ch),
        "TR": (w-cw, 0,    w,    ch),
        "BL": (0,    h-ch, cw,   h),
        "BR": (w-cw, h-ch, w,    h),
    }
    if corner_pos in mask_areas:
        x0, y0, x1, y1 = mask_areas[corner_pos]
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

    char = _ocr_with_fallback(_to_bytes(img))
    return order, char


# ─── UI ──────────────────────────────────────────────────────────────────────

class CaptchaReaderApp:

    def __init__(self, root):
        self.root = root
        self.root.title("驗證碼辨識工具")
        self.root.geometry("560x560")
        self.root.resizable(True, True)

        self.items = []   # list of {bytes, tk_img, order, char}
        self._build_ui()

    def _build_ui(self):
        ttk.Label(self.root, text="驗證碼辨識工具",
                  font=("Microsoft JhengHei UI", 14, "bold")).pack(pady=8)

        self.status_var = tk.StringVar(value="新增圖片後按「辨識全部」")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="#7f8c8d").pack()

        # ── 按鈕列 ──
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=12, pady=6)
        ttk.Button(btn_frame, text="📂 新增圖片",
                   command=self._add_file, width=13).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="📋 貼上剪貼簿",
                   command=self._paste, width=13).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="✂ 框選螢幕",
                   command=self._snip, width=13).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="🗑 清除全部",
                   command=self._clear, width=13).pack(side=tk.LEFT, padx=3)
        self.recog_btn = ttk.Button(btn_frame, text="▶ 辨識全部",
                                    command=self._recognize_all, width=13)
        self.recog_btn.pack(side=tk.RIGHT, padx=3)

        # ── 圖片列表 ──
        list_frame = ttk.LabelFrame(self.root, text="已載入圖片", padding=6)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        cols = ("idx", "preview", "order", "char")
        self.tree = ttk.Treeview(list_frame, columns=cols,
                                  show="headings", height=6)
        self.tree.heading("idx",     text="#")
        self.tree.heading("preview", text="檔名 / 來源")
        self.tree.heading("order",   text="順序")
        self.tree.heading("char",    text="辨識字元")
        self.tree.column("idx",     width=30,  anchor="center")
        self.tree.column("preview", width=260)
        self.tree.column("order",   width=60,  anchor="center")
        self.tree.column("char",    width=100, anchor="center")
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── 結果 ──
        res_frame = ttk.LabelFrame(self.root, text="最終輸入答案（依順序排列）", padding=10)
        res_frame.pack(fill=tk.X, padx=12, pady=6)
        row = ttk.Frame(res_frame)
        row.pack(fill=tk.X)
        self.result_var = tk.StringVar(value="—")
        ttk.Label(row, textvariable=self.result_var,
                  font=("Consolas", 22, "bold"),
                  foreground="#2980b9").pack(side=tk.LEFT)
        ttk.Button(row, text="複製", command=self._copy, width=8).pack(side=tk.RIGHT)

    # ── 框選螢幕 ──────────────────────────────────────────────────────────────

    def _snip(self):
        """最小化工具視窗，顯示半透明全螢幕覆蓋層讓使用者框選區域。"""
        self.root.withdraw()
        self.root.after(150, self._show_snip_overlay)

    def _show_snip_overlay(self):
        screen_img = ImageGrab.grab()   # 全螢幕截圖

        overlay = tk.Toplevel(self.root)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.35)
        overlay.configure(bg="black")
        overlay.config(cursor="crosshair")

        canvas = tk.Canvas(overlay, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        state = {"x0": 0, "y0": 0, "rect": None}

        def on_press(e):
            state["x0"] = e.x
            state["y0"] = e.y
            if state["rect"]:
                canvas.delete(state["rect"])

        def on_drag(e):
            if state["rect"]:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                state["x0"], state["y0"], e.x, e.y,
                outline="#00ff88", width=2, fill=""
            )

        def on_release(e):
            x0, y0 = min(state["x0"], e.x), min(state["y0"], e.y)
            x1, y1 = max(state["x0"], e.x), max(state["y0"], e.y)
            overlay.destroy()
            self.root.deiconify()

            if x1 - x0 < 5 or y1 - y0 < 5:
                self.status_var.set("框選區域太小，請重試")
                return

            cropped = screen_img.crop((x0, y0, x1, y1))
            buf = io.BytesIO()
            cropped.save(buf, "PNG")
            self._add_item(buf.getvalue(), cropped, label=f"截圖 ({x0},{y0})-({x1},{y1})")

        def on_esc(_):
            overlay.destroy()
            self.root.deiconify()
            self.status_var.set("已取消框選")

        canvas.bind("<ButtonPress-1>",   on_press)
        canvas.bind("<B1-Motion>",       on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>",         on_esc)

    # ── 新增圖片 ──────────────────────────────────────────────────────────────

    def _add_file(self):
        paths = filedialog.askopenfilenames(
            title="選擇圖片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                       ("All files", "*.*")]
        )
        for path in paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                img = Image.open(io.BytesIO(data))
                import os
                self._add_item(data, img, label=os.path.basename(path))
            except Exception as e:
                messagebox.showerror("載入失敗", str(e))

    def _paste(self):
        try:
            clip = ImageGrab.grabclipboard()
            if clip is None:
                messagebox.showwarning("提示", "剪貼簿沒有圖片")
                return
            # grabclipboard 有時回傳 list of paths
            if isinstance(clip, list):
                for p in clip:
                    if isinstance(p, str):
                        with open(p, "rb") as f:
                            data = f.read()
                        img = Image.open(io.BytesIO(data))
                        import os
                        self._add_item(data, img, label=os.path.basename(p))
                return
            buf = io.BytesIO()
            clip.save(buf, "PNG")
            self._add_item(buf.getvalue(), clip, label=f"剪貼簿 {len(self.items)+1}")
        except Exception as e:
            messagebox.showerror("貼上失敗", str(e))

    def _add_item(self, img_bytes, pil_img, label=""):
        idx = len(self.items) + 1
        # 縮圖 (不顯示在 tree，只存著)
        thumb = pil_img.copy()
        thumb.thumbnail((48, 48))
        tk_img = ImageTk.PhotoImage(thumb)
        item = {"bytes": img_bytes, "tk_img": tk_img,
                "label": label, "order": "—", "char": "—"}
        self.items.append(item)
        self.tree.insert("", tk.END, iid=str(idx),
                         values=(idx, label, "—", "—"))
        self.status_var.set(f"已載入 {len(self.items)} 張，按「辨識全部」開始")

    # ── 辨識 ──────────────────────────────────────────────────────────────────

    def _recognize_all(self):
        if not self.items:
            messagebox.showwarning("提示", "請先新增圖片")
            return
        self.recog_btn.config(state="disabled")
        self.status_var.set("辨識中…")

        def _run():
            for i, item in enumerate(self.items):
                order, char = _recognize_image(item["bytes"])
                item["order"] = order
                item["char"]  = char
                iid = str(i + 1)
                self.root.after(0, lambda iid=iid, o=order, c=char:
                    self.tree.item(iid, values=(iid, self.items[int(iid)-1]["label"], o, c))
                )

            # 依 order 排列，? 排最後
            def sort_key(it):
                return int(it["order"]) if it["order"].isdigit() else 99
            sorted_items = sorted(self.items, key=sort_key)
            answer = "".join(it["char"] for it in sorted_items if it["char"] != "?")
            if not answer:
                answer = "（無法辨識）"
            self.root.after(0, lambda: self.result_var.set(answer))
            self.root.after(0, lambda: self.status_var.set("✓ 辨識完成"))
            self.root.after(0, lambda: self.recog_btn.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    # ── 其他 ──────────────────────────────────────────────────────────────────

    def _clear(self):
        self.items.clear()
        self.tree.delete(*self.tree.get_children())
        self.result_var.set("—")
        self.status_var.set("已清除，可重新載入圖片")

    def _copy(self):
        result = self.result_var.get()
        if result and result not in ("—", "（無法辨識）"):
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            self.status_var.set("已複製到剪貼簿")


if __name__ == "__main__":
    root = tk.Tk()
    CaptchaReaderApp(root)
    root.mainloop()
