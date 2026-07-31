import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import os
import sys
import json
import winsound
import keyboard
import pydirectinput
import pyautogui
import pyperclip
from pynput import mouse as pynput_mouse

from core.input import _send_batch, MOUSE_TOKENS, _BTN_TO_TOKEN
from gui.rows import ActionRow, SkillRow

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True
pydirectinput.PAUSE = 0

if getattr(sys, 'frozen', False):
    # PyInstaller 6.x 打包（onedir）後，datas 裡的隨附資源（assets、logo）會被放進
    # exe 旁邊的 _internal 子資料夾，不是直接跟 exe 同一層；sys._MEIPASS 是官方提供、
    # 專門用來取得「隨附資源實際所在位置」的正確路徑，onefile/onedir 都適用，
    # 不能再假設是 dirname(sys.executable)，否則打包版會完全找不到 assets 裡的樣板圖。
    BASE_DIR     = os.path.dirname(sys.executable)
    RESOURCE_DIR = getattr(sys, '_MEIPASS', BASE_DIR)
else:
    BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESOURCE_DIR = BASE_DIR
# presets 是使用者自己存的腳本，要放在 exe 旁邊（可寫入），不能放進唯讀的 _internal。
PRESETS_DIR = os.path.join(BASE_DIR, 'presets')
ASSETS_DIR  = os.path.join(RESOURCE_DIR, 'assets')
LOGO_ICON   = os.path.join(RESOURCE_DIR, 'logo', 'logo.ico')

# ── 視覺樣式 ─────────────────────────────────────────
FONT       = 'Segoe UI'
BG         = '#eef1f6'   # 主背景
CARD_BG    = '#ffffff'   # 區塊卡片背景
ACCENT     = '#2563eb'   # 主色（藍）
SUCCESS    = '#16a34a'   # 綠（開始）
SUCCESS_DK = '#15803d'
DANGER     = '#dc2626'   # 紅（停止/刪除）
DANGER_DK  = '#b91c1c'
WARN       = '#ea580c'   # 橘（錄製/暫停）
TEXT_MUTED = '#6b7280'


def _refine_match(cv2, np, screen, tmpl, mask, coarse_bbox, coarse_scale, coarse_step):
    """在粗略掃描（縮圖＋稀疏尺度）找到的位置附近，改用原始解析度＋密集尺度
    重新比對一次，取得更準確的信心分數與框選位置。

    粗略掃描為了效能把螢幕縮小、尺度只取 15 個點，真實命中的最佳尺度常常
    落在兩個測試點中間，加上縮圖本身會讓細筆畫模糊，兩者都會讓分數看起來
    偏低，即使那個位置其實才是對的。只在真的找到疑似命中時才花這筆額外
    成本重新比一次（比對範圍縮小到命中框附近，成本很低），不影響平常
    「還沒出現」時持續輪詢的速度。"""
    x, y, w, h = coarse_bbox
    th, tw = tmpl.shape[:2]
    has_mask = mask.min() < 255

    pad = int(max(w, h) * 0.6)
    rx1, ry1 = max(0, x - pad), max(0, y - pad)
    rx2, ry2 = min(screen.shape[1], x + w + pad), min(screen.shape[0], y + h + pad)
    roi = screen[ry1:ry2, rx1:rx2]

    best_score, best_bbox = -1.0, coarse_bbox
    lo, hi = max(0.1, coarse_scale - coarse_step), coarse_scale + coarse_step
    for scale in np.linspace(lo, hi, 21):
        rw, rh = int(tw * scale), int(th * scale)
        if rw < 8 or rh < 8 or rw > roi.shape[1] or rh > roi.shape[0]:
            continue
        resized = cv2.resize(tmpl, (rw, rh))
        if has_mask:
            resized_mask = cv2.resize(mask, (rw, rh))
            result = cv2.matchTemplate(roi, resized, cv2.TM_SQDIFF_NORMED, mask=resized_mask)
            min_val, _, min_loc, _ = cv2.minMaxLoc(result)
            score, loc = 1.0 - min_val, min_loc
        else:
            result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
        if score > best_score:
            best_score, best_bbox = score, (rx1 + loc[0], ry1 + loc[1], rw, rh)
    return best_score, best_bbox


def _scan_templates(cv2, np, screen, templates, masks, names, conf, full=False,
                     downscale=0.5, num_scales=15):
    """比對 screen（彩色 BGR）與所有 templates，回傳 (是否命中, 最佳信心分數, 命中框(x,y,w,h), 樣板檔名)。
    用彩色而非灰階比對：灰階會丟失顏色資訊（例如炸彈的橘色、時鐘的藍色），
    保留顏色能大幅提升對真實背景的鑑別度。
    有真正 alpha 透明通道的去背圖，比對時用遮罩忽略透明部分，方法用 TM_SQDIFF_NORMED
    （差異平方，分數為 0~1 且「越小越像」，換算成 confidence = 1 - 差異值）；
    這個方法搭配 mask 時，對真實背景雜訊的鑑別度遠優於 TM_CCORR_NORMED（後者對大面積
    平坦/明亮區域會嚴重灌水分數，實測對照真實桌面畫面可誤判到 0.96 的高分）。
    沒有透明通道的圖則沿用原本的 TM_CCOEFF_NORMED（不支援 mask，但本身有做平均值校正，
    對平坦區域一樣不會灌水）。
    full=False 時一找到達門檻的結果就提早返回（效能較佳）；
    full=True 時掃完全部取得最佳分數。

    downscale/num_scales：全螢幕多尺度比對原本要 3 秒以上一次（35 個尺度 × 全解析度），
    導致「連續命中 2 次才算偵測到」实际要等 6~7 秒。實測把螢幕先縮小到一半、尺度數從
    35 降到 15，單次掃描降到 0.3 秒左右，用 24 張真實樣本驗證過準確率沒有變化（22/24，
    跟原本一樣，沒過的 2 張是已知的特殊縮放比例邊緣案例），才敢把預設值改成這樣。
    bbox 座標最後會換算回原始解析度，供下游字元格定位使用不受影響。"""
    if downscale != 1.0:
        h, w = screen.shape[:2]
        small_screen = cv2.resize(screen, (int(w * downscale), int(h * downscale)))
    else:
        small_screen = screen
    inv = 1.0 / downscale

    scales = np.linspace(0.3, 2.0, num_scales)
    coarse_step = (scales[-1] - scales[0]) / (num_scales - 1)

    best_score, best_bbox, best_name, hit = -1.0, None, None, False
    for tmpl, mask, name in zip(templates, masks, names):
        th, tw = tmpl.shape[:2]
        has_mask = mask.min() < 255
        for scale in scales:
            rw, rh = int(tw * scale * downscale), int(th * scale * downscale)
            if rw < 8 or rh < 8 or rw > small_screen.shape[1] or rh > small_screen.shape[0]:
                continue
            resized = cv2.resize(tmpl, (rw, rh))
            if has_mask:
                resized_mask = cv2.resize(mask, (rw, rh))
                result = cv2.matchTemplate(small_screen, resized, cv2.TM_SQDIFF_NORMED, mask=resized_mask)
                min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                score, loc = 1.0 - min_val, min_loc
            else:
                result = cv2.matchTemplate(small_screen, resized, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(result)
            if score > best_score:
                best_score, best_name = score, name
                best_bbox = (int(loc[0] * inv), int(loc[1] * inv), int(rw * inv), int(rh * inv))
            if score >= conf:
                hit = True
                if not full:
                    r_score, r_bbox = _refine_match(cv2, np, screen, tmpl, mask,
                                                      best_bbox, scale, coarse_step)
                    if r_score > best_score:
                        return hit, r_score, r_bbox, best_name
                    return hit, best_score, best_bbox, best_name
    return hit, best_score, best_bbox, best_name


class MacroApp:
    def __init__(self, root):
        self.root      = root
        self.root.title('彥彥老字號自動化腳本')
        self.root.resizable(False, False)
        if os.path.exists(LOGO_ICON):
            try:
                # default= 才會連同工作列（下方 EXE 圖示）用的大尺寸圖示一併設定，
                # 只呼叫 iconbitmap(path) 有時只換掉左上角標題列的小圖示。
                self.root.iconbitmap(default=LOGO_ICON)
            except Exception:
                pass
        self.running   = False
        self.listening = None
        self.recording = False
        self.record_events  = []
        self.skill_timers   = {}
        self.action_rows    = []
        self.skill_rows     = []
        self._held_keys     = set()
        self._lock          = threading.Lock()
        self._combo_pending = []
        self._combo_timer   = None
        self._mouse_listener = None
        self._capture_mouse  = None
        self._pause_event     = threading.Event()
        self._pause_event.set()
        self._det_pause_event = threading.Event()
        self._det_pause_event.set()
        self._det_conf_var    = tk.StringVar(value='0.6')
        self._det_resume_var  = tk.StringVar(value='2.0')
        self._det_en_var      = tk.BooleanVar(value=True)
        self._det_mode_var    = tk.StringVar(value='auto')  # 'alert'=僅警報 / 'auto'=自動測謊
        self._detected        = False
        self._captcha_progress = {}
        self._captcha_done      = False
        self._captcha_solving   = False
        self._captcha_enter_bbox = None
        self._captcha_submitted_at = None
        self._captcha_alert_sent   = False
        self._log_box          = None
        self._build()
        keyboard.hook(self._on_key)
        keyboard.on_press_key('f12', lambda _: self.root.after(0, self.toggle_run))

    # ── Key hook ─────────────────────────────────────────
    def _on_key(self, ev):
        if ev.name == 'f12':
            return

        if ev.event_type == keyboard.KEY_UP:
            self._held_keys.discard(ev.name)
            if self.recording:
                self.record_events.append(('up', ev.name, time.time()))
            return

        if ev.event_type != keyboard.KEY_DOWN:
            return

        if self.listening is not None:
            if ev.name not in self._combo_pending:
                self._combo_pending.append(ev.name)
            if self._combo_timer:
                self._combo_timer.cancel()
            self._combo_timer = threading.Timer(0.2, self._finalize_capture)
            self._combo_timer.start()
            return

        if self.recording and ev.name not in self._held_keys:
            co_held = frozenset(self._held_keys)
            self._held_keys.add(ev.name)
            self.record_events.append(('down', ev.name, time.time(), co_held))

    def _start_combo_capture_mouse(self):
        def on_click(*a):
            if self.listening is None:
                return False
            token = _BTN_TO_TOKEN.get(a[2])
            if token and a[3]:
                if token not in self._combo_pending:
                    self._combo_pending.append(token)
                if self._combo_timer:
                    self._combo_timer.cancel()
                self._combo_timer = threading.Timer(0.2, self._finalize_capture)
                self._combo_timer.start()
        self._capture_mouse = pynput_mouse.Listener(on_click=on_click)
        self._capture_mouse.start()

    def _finalize_capture(self):
        if self._capture_mouse:
            self._capture_mouse.stop()
            self._capture_mouse = None
        if self.listening and self._combo_pending:
            row = self.listening
            combo = '+'.join(self._combo_pending)
            self._combo_pending = []
            self._combo_timer   = None
            self.root.after(0, lambda: row.set_key(combo))

    # ── UI ───────────────────────────────────────────────
    def _btn(self, parent, text, command, bg, fg='white', width=None,
              font=None, anchor=None):
        """統一風格的扁平按鈕。"""
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      activebackground=bg, activeforeground=fg,
                      relief='flat', bd=0, cursor='hand2', width=width,
                      padx=10, pady=5, font=font or (FONT, 9, 'bold'))
        if anchor:
            b.pack(anchor=anchor)
        return b

    def _build(self):
        style = ttk.Style()
        try:
            style.theme_use('vista')
        except tk.TclError:
            pass
        style.configure('TCombobox', font=(FONT, 9))

        self.root.configure(bg=BG)

        # 控制列
        bar = tk.Frame(self.root, bg=BG, pady=8)
        bar.pack(fill='x', padx=12)
        self.run_btn = self._btn(bar, '▶  開始 (F12)', self.toggle_run,
                                  bg=SUCCESS, width=14, font=(FONT, 10, 'bold'))
        self.run_btn.pack(side='left', padx=(0, 8))
        self.status = tk.Label(bar, text='停止中', fg=TEXT_MUTED, bg=BG,
                                font=(FONT, 10, 'bold'))
        self.status.pack(side='left', padx=4)
        self.rec_btn = self._btn(bar, '●  錄製', self.toggle_record, bg=DANGER, width=9)
        self.rec_btn.pack(side='right')
        self.kb_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text='僅鍵盤', variable=self.kb_only_var, bg=BG,
                       activebackground=BG, font=(FONT, 9)).pack(side='right', padx=8)

        # 腳本管理列
        sl = tk.Frame(self.root, bg=BG, pady=2)
        sl.pack(fill='x', padx=12)
        tk.Label(sl, text='腳本', fg=TEXT_MUTED, bg=BG, font=(FONT, 9)).pack(side='left')
        self._preset_var = tk.StringVar()
        self._preset_cb  = ttk.Combobox(sl, textvariable=self._preset_var,
                                         width=22, state='readonly')
        self._preset_cb.pack(side='left', padx=6)
        self._btn(sl, '載入', self._load_script, bg='#4b5563', width=5,
                   font=(FONT, 9)).pack(side='left', padx=2)
        self._btn(sl, '儲存', self._save_script, bg=ACCENT, width=5,
                   font=(FONT, 9)).pack(side='left', padx=2)
        self._btn(sl, '刪除', self._delete_script, bg=DANGER, width=5,
                   font=(FONT, 9)).pack(side='left', padx=2)

        # 主要動作序列
        sf_lf = tk.LabelFrame(self.root, text=' 動作序列 ', bg=CARD_BG, fg='#111827',
                               font=(FONT, 10, 'bold'), padx=8, pady=6, bd=1,
                               relief='solid', highlightbackground='#d1d5db')
        sf_lf.pack(fill='both', padx=12, pady=6)

        sf_sb = tk.Scrollbar(sf_lf, orient='vertical')
        sf_sb.pack(side='right', fill='y')
        sf_cv = tk.Canvas(sf_lf, height=200, bg=CARD_BG, highlightthickness=0,
                          yscrollcommand=sf_sb.set)
        sf_cv.pack(side='left', fill='both', expand=True)
        sf_sb.config(command=sf_cv.yview)

        sf = tk.Frame(sf_cv, bg=CARD_BG)
        sf_cv.create_window((0, 0), window=sf, anchor='nw')
        sf.bind('<Configure>',
                lambda _: sf_cv.configure(scrollregion=sf_cv.bbox('all')))
        sf_cv.bind('<MouseWheel>',
                   lambda e: sf_cv.yview_scroll(int(-1 * (e.delta / 120)), 'units'))

        for col, h in enumerate(['類型', '按鍵/按鈕', '按住(秒)', '等待(秒)', '次數(0=∞)', '']):
            tk.Label(sf, text=h, bg=CARD_BG, fg=TEXT_MUTED,
                     font=(FONT, 9, 'bold')).grid(row=0, column=col, padx=4, pady=2)
        self.sf = sf
        self._btn(self.root, '+ 新增動作', self.add_action, bg=ACCENT,
                   font=(FONT, 9)).pack(anchor='w', padx=16, pady=(0, 4))

        # 輔助技能
        kf = tk.LabelFrame(self.root, text=' 輔助技能 ', bg=CARD_BG, fg='#111827',
                            font=(FONT, 10, 'bold'), padx=8, pady=6, bd=1,
                            relief='solid', highlightbackground='#d1d5db')
        kf.pack(fill='both', padx=12, pady=6)

        pre_row = tk.Frame(kf, bg=CARD_BG)
        pre_row.grid(row=0, column=0, columnspan=5, sticky='w', pady=(0, 6))
        tk.Label(pre_row, text='施放前停頓(秒)：', bg=CARD_BG,
                 font=(FONT, 9)).pack(side='left')
        self.precst_var = tk.StringVar(value='1.5')
        tk.Entry(pre_row, textvariable=self.precst_var, width=5,
                 justify='center').pack(side='left')
        tk.Label(pre_row, text='  （腳本先靜止此時間再施放技能）', bg=CARD_BG,
                 font=(FONT, 8), fg=TEXT_MUTED).pack(side='left')

        for col, h in enumerate(['按鍵', '冷卻(秒)', '施放(秒)', '']):
            tk.Label(kf, text=h, bg=CARD_BG, fg=TEXT_MUTED,
                     font=(FONT, 9, 'bold')).grid(row=1, column=col, padx=4, pady=2)
        self.kf = kf
        self._btn(self.root, '+ 新增技能', self.add_skill, bg=ACCENT,
                   font=(FONT, 9)).pack(anchor='w', padx=16, pady=(0, 8))

        # 測謊偵測
        df = tk.LabelFrame(self.root, text=' 防測謊系統 ', bg=CARD_BG, fg='#111827',
                           font=(FONT, 10, 'bold'), padx=8, pady=6, bd=1,
                           relief='solid', highlightbackground='#d1d5db')
        df.pack(fill='both', padx=12, pady=(0, 10))

        dr0 = tk.Frame(df, bg=CARD_BG)
        dr0.pack(fill='x', pady=2)
        tk.Checkbutton(dr0, text='啟用偵測', variable=self._det_en_var, bg=CARD_BG,
                       activebackground=CARD_BG, font=(FONT, 9, 'bold')).pack(side='left')
        tk.Label(dr0, text='相似度：', bg=CARD_BG, font=(FONT, 9)).pack(side='left', padx=(14, 0))
        tk.Entry(dr0, textvariable=self._det_conf_var, width=5,
                 justify='center').pack(side='left', padx=(2, 10))

        # 模式選擇：僅警報 / 自動測謊
        mode_row = tk.Frame(df, bg=CARD_BG)
        mode_row.pack(fill='x', pady=(4, 2))
        tk.Label(mode_row, text='模式：', bg=CARD_BG, font=(FONT, 9, 'bold')).pack(side='left')
        tk.Radiobutton(mode_row, text='🔔 僅警報（暫停+提示音，不自動作答）',
                       variable=self._det_mode_var, value='alert', bg=CARD_BG,
                       activebackground=CARD_BG, font=(FONT, 9)).pack(side='left', padx=(4, 12))
        tk.Radiobutton(mode_row, text='🤖 自動測謊（自動辨識並輸入送出）',
                       variable=self._det_mode_var, value='auto', bg=CARD_BG,
                       activebackground=CARD_BG, font=(FONT, 9)).pack(side='left')

        dr1 = tk.Frame(df, bg=CARD_BG)
        dr1.pack(fill='x', pady=(0, 2))
        tk.Label(dr1, text='偵測到測謊視窗後自動暫停腳本，視窗關閉後延遲指定秒數繼續執行',
                 bg=CARD_BG, font=(FONT, 8), fg=TEXT_MUTED).pack(side='left')

        dr2 = tk.Frame(df, bg=CARD_BG)
        dr2.pack(fill='x', pady=(0, 4))
        tk.Label(dr2, text='測謊消失後延遲幾秒恢復執行：', bg=CARD_BG,
                 font=(FONT, 9)).pack(side='left')
        tk.Entry(dr2, textvariable=self._det_resume_var, width=5,
                 justify='center').pack(side='left', padx=2)
        tk.Label(dr2, text='秒', bg=CARD_BG, font=(FONT, 9)).pack(side='left')

        log_frame = tk.Frame(df, bg=CARD_BG)
        log_frame.pack(fill='both', expand=True, pady=(4, 0))
        tk.Label(log_frame, text='辨識紀錄：', bg=CARD_BG, font=(FONT, 9)).pack(anchor='w')
        log_sb = tk.Scrollbar(log_frame, orient='vertical')
        log_sb.pack(side='right', fill='y')
        self._log_box = tk.Text(log_frame, height=8, state='disabled', bg='#f9fafb',
                                 relief='solid', bd=1, highlightbackground='#d1d5db',
                                 font=('Consolas', 9), yscrollcommand=log_sb.set)
        self._log_box.pack(fill='both', expand=True)
        log_sb.config(command=self._log_box.yview)

        self._refresh_preset_list()

    # ── Script save / load ───────────────────────────────
    def _refresh_preset_list(self):
        os.makedirs(PRESETS_DIR, exist_ok=True)
        names = sorted(f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith('.json'))
        self._preset_cb['values'] = names
        if self._preset_var.get() not in names:
            self._preset_var.set(names[0] if names else '')

    def _save_script(self):
        name = simpledialog.askstring('儲存腳本', '輸入腳本名稱：', parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        try:    precst = float(self.precst_var.get())
        except: precst = 1.5
        data = {
            'name':    name,
            'precst':  precst,
            'actions': [r.get() for r in self.action_rows],
            'skills':  [r.get() for r in self.skill_rows],
            'detect': {
                'enabled':    self._det_en_var.get(),
                'mode':       self._det_mode_var.get(),
                'confidence': self._det_conf_var.get(),
                'resume':     self._det_resume_var.get(),
            },
        }
        os.makedirs(PRESETS_DIR, exist_ok=True)
        with open(os.path.join(PRESETS_DIR, f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._refresh_preset_list()
        self._preset_var.set(name)
        messagebox.showinfo('儲存成功', f'腳本「{name}」已儲存！', parent=self.root)

    def _load_script(self):
        name = self._preset_var.get()
        if not name:
            return
        path = os.path.join(PRESETS_DIR, f'{name}.json')
        if not os.path.exists(path):
            messagebox.showerror('錯誤', f'找不到腳本「{name}」', parent=self.root)
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for r in list(self.action_rows): self.del_action(r)
        for r in list(self.skill_rows):  self.del_skill(r)
        self.precst_var.set(str(data.get('precst', 1.5)))
        for a in data.get('actions', []):
            key_or_mouse = (a.get('mouse_btn', 'right')
                            if a['type'] in ('mouse_down', 'mouse_up')
                            else a.get('key', ''))
            self._new_action(a['type'], key_or_mouse,
                             a.get('hold', 0.05), a.get('delay', 0.0), a.get('repeat', 1))
        for s in data.get('skills', []):
            self._new_skill(s['key'], s['cooldown'], s.get('cast', 1.0))
        det = data.get('detect', {})
        self._det_en_var.set(det.get('enabled', True))
        self._det_mode_var.set(det.get('mode', 'auto'))
        self._det_conf_var.set(str(det.get('confidence', 0.6)))
        self._det_resume_var.set(str(det.get('resume', 2.0)))

    def _delete_script(self):
        name = self._preset_var.get()
        if not name:
            return
        if not messagebox.askyesno('確認刪除', f'確定要刪除腳本「{name}」？', parent=self.root):
            return
        path = os.path.join(PRESETS_DIR, f'{name}.json')
        if os.path.exists(path):
            os.remove(path)
        self._refresh_preset_list()

    # ── Row management ───────────────────────────────────
    def _new_action(self, type_key='key', key='', hold=0.05, delay=0.0, repeat=1):
        row = ActionRow(self, self.sf, self.del_action)
        row.init(type_key, key, hold, delay, repeat)
        row.place(len(self.action_rows) + 1)
        self.action_rows.append(row)

    def add_action(self):
        self._new_action()

    def del_action(self, row):
        row.destroy()
        self.action_rows.remove(row)
        for i, r in enumerate(self.action_rows):
            r.place(i + 1)

    def _new_skill(self, key='', cd=60, cast=1.0):
        row = SkillRow(self, self.kf, self.del_skill)
        if key:
            row.init(key, cd, cast)
        row.place(len(self.skill_rows) + 2)
        self.skill_rows.append(row)
        self.skill_timers[id(row)] = 0

    def add_skill(self):
        self._new_skill()

    def del_skill(self, row):
        self.skill_timers.pop(id(row), None)
        row.destroy()
        self.skill_rows.remove(row)
        for i, r in enumerate(self.skill_rows):
            r.place(i + 2)

    # ── Run loop ─────────────────────────────────────────
    def toggle_run(self):
        if self.recording:
            return
        (self._stop if self.running else self._start)()

    def _refresh_cache(self):
        self._action_snapshot = [row.get() for row in list(self.action_rows)]
        self._skill_snapshot  = [(id(r), r.get()) for r in list(self.skill_rows)]
        try:    self._precst_cache = float(self.precst_var.get())
        except: self._precst_cache = 1.5

    def _start(self):
        self.running = True
        self._pause_event.set()
        self._det_pause_event.set()
        self.skill_timers     = {id(r): 0 for r in self.skill_rows}
        self._action_snapshot = []
        self._skill_snapshot  = []
        self._precst_cache    = 1.5
        self._detected        = False
        self._captcha_progress = {}
        self._captcha_done      = False
        self._captcha_solving   = False
        self._captcha_enter_bbox = None
        self._captcha_submitted_at = None
        self._captcha_alert_sent   = False
        self._refresh_cache()
        self.run_btn.config(text='■ 停止 (F12)', bg='#c62828')
        self.status.config(text='▶ 執行中', fg='#2e7d32')
        threading.Thread(target=self._loop,        daemon=True).start()
        threading.Thread(target=self._skill_loop,  daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    def _stop(self):
        self.running = False
        self._pause_event.set()
        self._det_pause_event.set()
        with self._lock:
            pydirectinput.mouseUp(button='left')
            pydirectinput.mouseUp(button='right')
            pydirectinput.mouseUp(button='middle')
        self.run_btn.config(text='▶ 開始 (F12)', bg='#388e3c')
        self.status.config(text='停止中', fg='gray')

    # ── Detect ───────────────────────────────────────────
    _CONSEC_HITS_REQUIRED = 2  # 連續幾次偵測到才視為真的出現，濾掉單一畫面的巧合誤判
    # 偵測輪詢間隔：寫死成 0.2 秒——這已經是迴圈裡 sleep 的安全下限（見 _detect_loop
    # 最下面的 max(0.2, itvl)），再往下設也會被強制拉回 0.2，等於是能設到的最快值，
    # 不需要讓使用者自己煩惱怎麼調。
    _DETECT_INTERVAL = 0.2

    def _load_templates(self, cv2, np):
        """讀取 assets/ 內所有圖片作為樣板（保留彩色，不轉灰階，鑑別度較好）。
        有 alpha 透明通道的圖（去背圖）會取出透明遮罩，比對時忽略去背後的背景；
        沒有透明通道的圖則視為全不遮蔽。"""
        os.makedirs(ASSETS_DIR, exist_ok=True)
        templates, masks, names = [], [], []
        for f in os.listdir(ASSETS_DIR):
            if not f.lower().endswith(('.png', '.jpg', '.bmp')):
                continue
            # cv2.imread 在 Windows 上路徑含中文字時常直接讀失敗（回傳 None），
            # 因為打包後的資料夾名稱是中文「彥彥老字號」，改用 np.fromfile 讀
            # bytes 再用 cv2.imdecode 解碼，不受路徑編碼影響。
            try:
                data = np.fromfile(os.path.join(ASSETS_DIR, f), dtype=np.uint8)
            except (FileNotFoundError, OSError):
                continue
            if data.size == 0:
                continue
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                color = img[:, :, :3]
                mask  = img[:, :, 3]
            else:
                color = img
                mask  = np.full(img.shape[:2], 255, dtype=np.uint8)
            templates.append(color)
            masks.append(mask)
            names.append(f)
        return templates, masks, names

    def _detect_loop(self):
        import cv2
        import numpy as np

        consec_hits = 0
        while self.running:
            if not self._det_en_var.get():
                consec_hits = 0
                time.sleep(0.2)
                continue

            templates, masks, names = self._load_templates(cv2, np)
            if not templates:
                time.sleep(0.2)
                continue

            try:    conf = float(self._det_conf_var.get())
            except: conf = 0.6

            try:
                screenshot  = pyautogui.screenshot()
                screen      = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                on_screen, score, bbox, name = _scan_templates(cv2, np, screen, templates, masks, names, conf, full=False)
            except Exception:
                time.sleep(0.2)
                continue

            consec_hits = consec_hits + 1 if on_screen else 0

            if not self._detected and consec_hits >= self._CONSEC_HITS_REQUIRED:
                # 測謊出現 → 暫停腳本，重置這次的字元收集進度
                self._detected = True
                self._captcha_progress = {}
                self._captcha_done      = False
                self._captcha_enter_bbox = None
                self._captcha_submitted_at = None
                self._captcha_alert_sent   = False
                self._det_pause_event.clear()
                self.root.after(0, self._on_detected)
                self.root.after(0, self._log_line,
                                 f'✅ 偵測到測謊「{name}」(信心 {score:.2f} ≥ 門檻 {conf:.2f})')
                if self._det_mode_var.get() != 'auto':
                    self.root.after(0, self._log_line, '🔔 僅警報模式，不會自動辨識/作答')

            elif self._detected and not on_screen:
                # 測謊消失 → 等 delay 後恢復
                try:    resume = float(self._det_resume_var.get())
                except: resume = 2.0
                self.root.after(0, self._log_line, f'❌ 測謊消失，{resume:.1f} 秒後恢復執行')
                time.sleep(max(0.0, resume))
                self._detected = False
                self._captcha_progress = {}
                self._captcha_done      = False
                self._captcha_enter_bbox = None
                self._captcha_submitted_at = None
                self._captcha_alert_sent   = False
                self._det_pause_event.set()
                self.root.after(0, self._on_resumed)

            auto_mode = self._det_mode_var.get() == 'auto'

            # 測謊還在畫面上、還沒收集齊 3 個順序的字元 → 持續嘗試辨識
            # （這款測謊同一時間只顯示一格字元圖，順序 1/2/3 是輪流出現，
            # 所以要跨好幾次偵測累積，收滿 1、2、3 才算完成，不是抓到一格就結束）
            # 僅警報模式完全不辨識、不作答，只負責暫停腳本+提示音。
            if (auto_mode and on_screen and bbox and not self._captcha_done
                    and not self._captcha_solving):
                threading.Thread(target=self._solve_captcha,
                                  args=(screen, bbox), daemon=True).start()

            # 自動測謊模式下，送出答案後正常應該會讓測謊消失；如果送出後過了
            # 一段緩衝時間，畫面上還是偵測得到，很可能代表答案錯了，這時才需要
            # 警報提醒使用者介入，而不是像僅警報模式那樣一偵測到就響。
            if (auto_mode and self._captcha_submitted_at is not None
                    and not self._captcha_alert_sent and on_screen
                    and time.time() - self._captcha_submitted_at >= 2.0):
                self._captcha_alert_sent = True
                threading.Thread(target=self._alert_beep, daemon=True).start()
                self.root.after(0, self._log_line,
                                 '⚠ 送出答案後測謊仍未消失，可能答錯了，請確認')

            if auto_mode and self._detected and not self._captcha_done:
                # 還在收集 1/2/3 順序階段：字元格內容會一直變化，要盡快連續重試。
                time.sleep(0.1)
            else:
                # 還沒偵測到測謊、或已經收集完成：用寫死的最快間隔慢慢檢查即可。
                time.sleep(self._DETECT_INTERVAL)

    # ── 測謊自動辨識/作答 ─────────────────────────────────
    _CAPTCHA_ORDERS = ("1", "2", "3")

    def _solve_captcha(self, screen, anchor_bbox):
        self._captcha_solving = True
        try:
            try:
                from core.captcha import solve_captcha_region
            except Exception as e:
                self.root.after(0, lambda: self.status.config(
                    text=f'⚠ OCR 模組載入失敗: {e}', fg='#c62828'))
                self.root.after(0, self._log_line, f'⚠ OCR 模組載入失敗: {e}')
                return

            try:
                _, results, enter_bbox = solve_captcha_region(screen, anchor_bbox)
            except Exception as e:
                self.root.after(0, lambda: self.status.config(
                    text=f'⚠ 辨識錯誤: {e}', fg='#c62828'))
                self.root.after(0, self._log_line, f'⚠ 辨識錯誤: {e}')
                return

            if enter_bbox:
                self._captcha_enter_bbox = enter_bbox

            for order, char, *_ in results:
                if order in self._CAPTCHA_ORDERS and char not in ("", "?") \
                        and self._captcha_progress.get(order) != char:
                    self._captcha_progress[order] = char
                    self.root.after(0, self._log_line, f'   順序={order}　字元={char}')

            progress = self._captcha_progress
            progress_str = '  '.join(
                f'{o}={progress[o]}' if o in progress else f'{o}=?'
                for o in self._CAPTCHA_ORDERS)

            if all(o in progress for o in self._CAPTCHA_ORDERS):
                answer = "".join(progress[o] for o in self._CAPTCHA_ORDERS)
                self._captcha_done = True
                self.root.after(0, lambda: self.status.config(
                    text=f'⏸ 測謊辨識完成，輸入中: {answer}', fg='#1565c0'))
                self.root.after(0, self._log_line, f'→ 組合答案 (1+2+3): {answer}')
                time.sleep(0.15)
                self._submit_answer(self._captcha_enter_bbox, answer)
                self._captcha_submitted_at = time.time()
            else:
                self.root.after(0, lambda: self.status.config(
                    text=f'⏸ 測謊辨識中... {progress_str}', fg='#e65100'))
        finally:
            self._captcha_solving = False

    def _submit_answer(self, enter_bbox, answer):
        """先點擊輸入框讓輸入焦點正確落在測謊的文字框，再打字＋按 Enter。"""
        if enter_bbox:
            from core.captcha import enter_box_click_point
            cx, cy = enter_box_click_point(enter_bbox)
            with self._lock:
                pyautogui.click(cx, cy)
            self.root.after(0, self._log_line, '   已點擊輸入框')
            time.sleep(0.1)
        self._send_answer(answer)

    def _send_answer(self, answer):
        """用剪貼簿貼上而不是逐字元模擬按鍵——如果使用者當下輸入法是中文，
        模擬按鍵會被輸入法攔截去組字，貼上則是直接把文字塞進輸入框，不受
        目前輸入法狀態影響。"""
        prev_clip = None
        try:
            prev_clip = pyperclip.paste()
        except Exception:
            pass

        pyperclip.copy(answer)
        time.sleep(0.05)
        with self._lock:
            _send_batch(['ctrl', 'v'], [], release=False)
        time.sleep(0.05)
        with self._lock:
            _send_batch(['ctrl', 'v'], [], release=True)
        time.sleep(0.1)
        self.root.after(0, self._log_line, f'   已貼上答案: {answer}')

        if prev_clip is not None:
            try:
                pyperclip.copy(prev_clip)
            except Exception:
                pass

        with self._lock:
            _send_batch(['enter'], [], release=False)
        time.sleep(0.03)
        with self._lock:
            _send_batch(['enter'], [], release=True)
        self.root.after(0, self._log_line, '   已按下 Enter 送出')

    def _on_detected(self):
        # 自動測謊模式：先不響鈴，等送出答案後如果測謊沒消失才需要警報（見 _detect_loop）。
        # 僅警報模式：維持原本行為，一偵測到就立刻響鈴提示。
        if self._det_mode_var.get() != 'auto':
            threading.Thread(target=self._alert_beep, daemon=True).start()
        self.status.config(text='⏸ 測謊暫停中', fg='#e65100')

    def _on_resumed(self):
        self.status.config(text='▶ 執行中', fg='#2e7d32')

    def _alert_beep(self):
        # C5→E5→G5→C6 arpeggio，重複兩次
        melody = [(523,130),(659,130),(784,130),(1047,280),
                  (784,100),(880,100),(988,100),(1047,380)]
        for _ in range(2):
            for freq, dur in melody:
                winsound.Beep(freq, dur)
                time.sleep(0.03)
            time.sleep(0.25)

    # ── 辨識紀錄 LOG（主視窗，防測謊系統區塊內）──────────
    def _log_line(self, text):
        if not self._log_box:
            return
        ts = time.strftime('%H:%M:%S')
        self._log_box.config(state='normal')
        self._log_box.insert('end', f'[{ts}] {text}\n')
        self._log_box.see('end')
        self._log_box.config(state='disabled')

    # ── Main loops ───────────────────────────────────────
    def _loop(self):
        while self.running:
            actions = list(self._action_snapshot)
            self.root.after(0, self._refresh_cache)
            if not actions:
                time.sleep(0.1)
                continue
            for a in actions:
                if not self.running:
                    break
                if a['type'] == 'mouse_up':
                    self._exec(a)
                    continue
                repeat = a['repeat']
                if repeat == 0:
                    while self.running:
                        self._exec(a)
                else:
                    for _ in range(max(1, repeat)):
                        if not self.running:
                            break
                        self._exec(a)

    def _skill_loop(self):
        while self.running:
            self._det_pause_event.wait()
            now    = time.time()
            skills = list(self._skill_snapshot)
            ready  = [
                (rid, s) for rid, s in skills
                if s['key'] and now - self.skill_timers.get(rid, 0) >= s['cooldown']
            ]
            if ready:
                self._pause_event.clear()
                time.sleep(max(0.0, self._precst_cache))
                for rid, s in ready:
                    if not self.running:
                        break
                    self._det_pause_event.wait()
                    for _ in range(3):
                        with self._lock:
                            _send_batch([s['key']], [], release=False)
                        time.sleep(0.05)
                        with self._lock:
                            _send_batch([s['key']], [], release=True)
                        time.sleep(0.08)
                    self.skill_timers[rid] = time.time()
                    time.sleep(s['cast'])
                self._pause_event.set()
            time.sleep(0.3)

    def _isleep(self, secs):
        start = time.time()
        end   = start + secs
        while self.running and time.time() < end:
            if not self._pause_event.is_set() or not self._det_pause_event.is_set():
                break
            time.sleep(0.05)
        return time.time() - start

    def _exec(self, a):
        self._pause_event.wait()
        self._det_pause_event.wait()
        if not self.running:
            return
        t = a['type']

        if t == 'mouse_down':
            with self._lock:
                pydirectinput.mouseDown(button=a['mouse_btn'])
            remaining = a['hold']
            while self.running and remaining > 0.01:
                slept     = self._isleep(remaining)
                remaining -= slept
                if remaining > 0.01 and self.running:
                    self._pause_event.wait()
                    self._det_pause_event.wait()
            self._isleep(a['delay'])

        elif t == 'mouse_up':
            with self._lock:
                pydirectinput.mouseUp(button=a['mouse_btn'])
            self._isleep(a['delay'])

        else:
            parts      = a['key'].split('+') if a['key'] else []
            kb_keys    = [p for p in parts if p not in MOUSE_TOKENS]
            mouse_btns = [MOUSE_TOKENS[p] for p in parts if p in MOUSE_TOKENS]
            if parts:
                remaining = a['hold']
                while self.running:
                    with self._lock:
                        _send_batch(kb_keys, mouse_btns, release=False)
                    slept     = self._isleep(remaining)
                    remaining -= slept
                    with self._lock:
                        _send_batch(list(reversed(kb_keys)),
                                    list(reversed(mouse_btns)), release=True)
                    if remaining > 0.01 and self.running:
                        self._pause_event.wait()
                        self._det_pause_event.wait()
                    else:
                        break
            self._isleep(a['delay'])

    # ── Record ───────────────────────────────────────────
    def toggle_record(self):
        if self.running:
            return
        if not self.recording:
            self.recording = True
            self.record_events = []
            self._held_keys = set()
            if not self.kb_only_var.get():
                self._start_mouse_listener()
            self.rec_btn.config(text='■ 停止錄製', bg='#e65100')
            self.status.config(text='● 錄製中', fg='#c62828')
        else:
            self.recording = False
            self._stop_mouse_listener()
            self.rec_btn.config(text='● 錄製', bg='#c62828')
            self.status.config(text='停止中', fg='gray')
            self._apply_record()

    def _start_mouse_listener(self):
        _BTN = {
            pynput_mouse.Button.left:   'left',
            pynput_mouse.Button.right:  'right',
            pynput_mouse.Button.middle: 'middle',
        }
        def on_click(*a):
            btn = _BTN.get(a[2])
            if btn and self.recording:
                ev = 'mouse_down' if a[3] else 'mouse_up'
                self.record_events.append((ev, btn, time.time()))
        self._mouse_listener = pynput_mouse.Listener(on_click=on_click)
        self._mouse_listener.start()

    def _stop_mouse_listener(self):
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

    def _apply_record(self):
        if not self.record_events:
            return

        LONG_HOLD = 1.0

        kb_evs = sorted(
            [(item[0], item[1], item[2])
             for item in self.record_events if item[0] in ('down', 'up')],
            key=lambda x: x[2]
        )

        cur     = set()
        seg_t   = None
        kb_segs = []

        for ev, key, t in kb_evs:
            if cur and seg_t is not None:
                hold = round(max(0.01, t - seg_t), 3)
                kb_segs.append((seg_t, t, '+'.join(sorted(cur)), hold))
            if ev == 'down':
                cur.add(key)
            else:
                cur.discard(key)
            seg_t = t if cur else None

        _BTN_TOK = {'left': 'MOUSE_L', 'right': 'MOUSE_R', 'middle': 'MOUSE_M'}
        m_dn = {}
        completed_mouse = []
        for item in self.record_events:
            if item[0] == 'mouse_down':
                m_dn[item[1]] = item[2]
            elif item[0] == 'mouse_up' and item[1] in m_dn:
                completed_mouse.append([item[1], m_dn.pop(item[1]), item[2], False])
        for btn, t in m_dn.items():
            completed_mouse.append([btn, t, t, False])

        timeline = []

        for start_t, end_t, combo, hold in kb_segs:
            merged = []
            for entry in completed_mouse:
                if not entry[3]:
                    dn_t, up_t = entry[1], entry[2]
                    if abs(dn_t - start_t) < 0.05 and (up_t - dn_t) < LONG_HOLD:
                        merged.append(entry)
                        entry[3] = True
            if merged:
                end_t = max([end_t] + [e[2] for e in merged])
                hold  = round(max(0.01, end_t - start_t), 3)
                combo = combo + '+' + '+'.join(_BTN_TOK[e[0]] for e in merged)
            timeline.append((start_t, end_t, 'key', combo, hold))

        for btn, dn_t, up_t, consumed in completed_mouse:
            if not consumed:
                hold = round(max(0.0, up_t - dn_t), 3)
                timeline.append((dn_t, up_t, 'mouse_down', btn, hold))
                if up_t > dn_t:
                    timeline.append((up_t, up_t, 'mouse_up', btn, 0.0))

        if not timeline:
            return

        timeline.sort(key=lambda x: x[0])

        for row in list(self.action_rows):
            self.del_action(row)

        for i, (_, end_t, type_, key_or_btn, hold) in enumerate(timeline):
            delay = round(max(0.0, timeline[i+1][0] - end_t), 3) if i + 1 < len(timeline) else 0.0
            self._new_action(type_, key_or_btn, hold, delay, 1)
