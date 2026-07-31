import tkinter as tk
from tkinter import ttk
from core.input import (TYPE_OPTIONS, TYPE_KEYS, MOUSE_BTNS, MOUSE_TOKEN_DISP)


class ActionRow:
    def __init__(self, app, frame, on_delete):
        self.app = app
        self.frame = frame
        self.key = ''
        self.mouse_btn = 'right'
        self._type = 'key'

        self.type_var = tk.StringVar(value='鍵盤按鍵')
        self.type_cb = ttk.Combobox(frame, textvariable=self.type_var,
                                     values=TYPE_OPTIONS, width=9, state='readonly')
        self.type_cb.bind('<<ComboboxSelected>>', self._on_type)

        self.key_btn = tk.Button(frame, text='(點擊設定)', width=10,
                                  command=self._capture, relief='groove', font=('Segoe UI', 9))

        self.hold_var   = tk.StringVar(value='0.05')
        self.delay_var  = tk.StringVar(value='0.0')
        self.repeat_var = tk.StringVar(value='1')
        self.hold_e   = tk.Entry(frame, textvariable=self.hold_var,   width=6, justify='center')
        self.delay_e  = tk.Entry(frame, textvariable=self.delay_var,  width=6, justify='center')
        self.repeat_e = tk.Entry(frame, textvariable=self.repeat_var, width=5, justify='center')
        self.del_btn  = tk.Button(frame, text='✕', fg='red',
                                   command=lambda: on_delete(self), relief='flat')

        self.all_w = [self.type_cb, self.key_btn,
                      self.hold_e, self.delay_e, self.repeat_e, self.del_btn]

    def _on_type(self, _=None):
        self._type = TYPE_KEYS[TYPE_OPTIONS.index(self.type_var.get())]
        if self._type in ('mouse_down', 'mouse_up'):
            self.key_btn.config(text=MOUSE_BTNS.get(self.mouse_btn, '右鍵'))
        else:
            self.key_btn.config(text=self.key.upper() if self.key else '(點擊設定)')
        state = 'disabled' if self._type == 'mouse_up' else 'normal'
        self.hold_e.config(state=state)
        self.repeat_e.config(state=state)

    def _capture(self):
        if self._type in ('mouse_down', 'mouse_up'):
            m = tk.Menu(self.frame, tearoff=0)
            for btn, label in MOUSE_BTNS.items():
                m.add_command(label=label, command=lambda b=btn: self._set_mouse(b))
            try:
                m.tk_popup(self.key_btn.winfo_rootx(),
                           self.key_btn.winfo_rooty() + self.key_btn.winfo_height())
            finally:
                m.grab_release()
        else:
            self.key_btn.config(text='按鍵或點滑鼠...', bg='#fff3cd')
            self.app.listening = self
            self.app._start_combo_capture_mouse()

    def _set_mouse(self, btn):
        self.mouse_btn = btn
        self.key_btn.config(text=MOUSE_BTNS[btn], bg='SystemButtonFace')

    def set_key(self, key):
        self.key = key
        parts = key.split('+')
        display = '+'.join(MOUSE_TOKEN_DISP.get(p, p.upper()) for p in parts)
        self.key_btn.config(text=display, bg='SystemButtonFace')
        self.app.listening = None

    def init(self, type_key, key_or_mouse, hold=0.05, delay=0.0, repeat=1):
        self._type = type_key
        self.type_var.set(TYPE_OPTIONS[TYPE_KEYS.index(type_key)])
        if type_key in ('mouse_down', 'mouse_up'):
            self.mouse_btn = key_or_mouse
            self.key_btn.config(text=MOUSE_BTNS.get(key_or_mouse, key_or_mouse))
            if type_key == 'mouse_up':
                self.hold_e.config(state='disabled')
                self.repeat_e.config(state='disabled')
        else:
            self.key = key_or_mouse
            parts = key_or_mouse.split('+')
            display = '+'.join(MOUSE_TOKEN_DISP.get(p, p.upper()) for p in parts)
            self.key_btn.config(text=display)
        self.hold_var.set(str(hold))
        self.delay_var.set(str(delay))
        self.repeat_var.set(str(repeat))
        return self

    def get(self):
        def sf(v, d):
            try: return float(v.get())
            except: return d
        def si(v, d):
            try: return int(v.get())
            except: return d
        return {
            'type':      self._type,
            'key':       self.key,
            'mouse_btn': self.mouse_btn,
            'hold':      sf(self.hold_var, 0.05),
            'delay':     sf(self.delay_var, 0.0),
            'repeat':    si(self.repeat_var, 1),
        }

    def place(self, row):
        for col, w in enumerate(self.all_w):
            w.grid(row=row, column=col, padx=3, pady=2)

    def destroy(self):
        for w in self.all_w:
            w.destroy()


class SkillRow:
    def __init__(self, app, frame, on_delete):
        self.app   = app
        self.frame = frame
        self.key   = ''

        self.key_btn  = tk.Button(frame, text='(點擊設定)', width=10,
                                   command=self._capture, relief='groove', font=('Segoe UI', 9))
        self.cd_var   = tk.StringVar(value='60')
        self.cast_var = tk.StringVar(value='1.0')
        cd_e    = tk.Entry(frame, textvariable=self.cd_var,   width=7, justify='center')
        cast_e  = tk.Entry(frame, textvariable=self.cast_var, width=6, justify='center')
        del_btn = tk.Button(frame, text='✕', fg='red',
                             command=lambda: on_delete(self), relief='flat')
        self.all_w = [self.key_btn, cd_e, cast_e, del_btn]

    def _capture(self):
        self.key_btn.config(text='按下按鍵...', bg='#fff3cd')
        self.app.listening = self

    def set_key(self, key):
        self.key = key.split('+')[0]
        self.key_btn.config(text=self.key.upper(), bg='SystemButtonFace')
        self.app.listening = None

    def init(self, key, cooldown, cast=1.0):
        self.key = key
        self.key_btn.config(text=key.upper())
        self.cd_var.set(str(cooldown))
        self.cast_var.set(str(cast))
        return self

    def get(self):
        try: cd = float(self.cd_var.get())
        except: cd = 60
        try: cast = float(self.cast_var.get())
        except: cast = 1.0
        return {'key': self.key, 'cooldown': cd, 'cast': cast}

    def place(self, row):
        for col, w in enumerate(self.all_w):
            w.grid(row=row, column=col, padx=3, pady=2)

    def destroy(self):
        for w in self.all_w:
            w.destroy()
