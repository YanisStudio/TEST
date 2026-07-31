# 安裝需求：pip install pyautogui pydirectinput keyboard pynput opencv-python
import sys
import tkinter as tk
from gui.app import MacroApp

if __name__ == '__main__':
    if sys.platform == 'win32':
        # 沒有這行，Windows 工作列會把視窗跟 python.exe 分到同一群組，直接沿用
        # python.exe 內建的預設圖示，即使視窗本身已經用 iconbitmap 設過 LOGO 也沒用；
        # 給一個獨立的 AppUserModelID 讓工作列改成用視窗自己的圖示。
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('YenYen.GameMacro.AutoScript')
        except Exception:
            pass
    root = tk.Tk()
    app  = MacroApp(root)
    root.mainloop()
