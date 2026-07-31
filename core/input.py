import ctypes
from ctypes import wintypes
from pynput import mouse as pynput_mouse

class _KI(ctypes.Structure):
    _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD),
                ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                ('dwExtraInfo', ctypes.c_size_t)]

class _MI(ctypes.Structure):
    _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG),
                ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]

class _IU(ctypes.Union):
    _fields_ = [('ki', _KI), ('mi', _MI)]

class _INPUT(ctypes.Structure):
    _fields_ = [('type', wintypes.DWORD), ('_u', _IU)]

_MOUSE_DN = {'left': 0x0002, 'right': 0x0008, 'middle': 0x0020}
_MOUSE_UP = {'left': 0x0004, 'right': 0x0010, 'middle': 0x0040}
_VK = {
    'a':0x41,'b':0x42,'c':0x43,'d':0x44,'e':0x45,'f':0x46,'g':0x47,'h':0x48,
    'i':0x49,'j':0x4A,'k':0x4B,'l':0x4C,'m':0x4D,'n':0x4E,'o':0x4F,'p':0x50,
    'q':0x51,'r':0x52,'s':0x53,'t':0x54,'u':0x55,'v':0x56,'w':0x57,'x':0x58,
    'y':0x59,'z':0x5A,
    '0':0x30,'1':0x31,'2':0x32,'3':0x33,'4':0x34,
    '5':0x35,'6':0x36,'7':0x37,'8':0x38,'9':0x39,
    'space':0x20,'enter':0x0D,'shift':0x10,'ctrl':0x11,'alt':0x12,
    'tab':0x09,'escape':0x1B,'backspace':0x08,
    'f1':0x70,'f2':0x71,'f3':0x72,'f4':0x73,'f5':0x74,
    'f6':0x75,'f7':0x76,'f8':0x77,'f9':0x78,'f10':0x79,'f11':0x7A,
    'up':0x26,'down':0x28,'left':0x25,'right':0x27,
}

TYPE_OPTIONS     = ['鍵盤按鍵', '滑鼠按下', '滑鼠放開']
TYPE_KEYS        = ['key',      'mouse_down', 'mouse_up']
MOUSE_BTNS       = {'left': '左鍵', 'right': '右鍵', 'middle': '中鍵'}
MOUSE_TOKENS     = {'MOUSE_L': 'left', 'MOUSE_R': 'right', 'MOUSE_M': 'middle'}
MOUSE_TOKEN_DISP = {'MOUSE_L': '左鍵', 'MOUSE_R': '右鍵',  'MOUSE_M': '中鍵'}
_BTN_TO_TOKEN    = {
    pynput_mouse.Button.left:   'MOUSE_L',
    pynput_mouse.Button.right:  'MOUSE_R',
    pynput_mouse.Button.middle: 'MOUSE_M',
}

def _send_batch(kb_keys, mouse_btns, release=False):
    items = []
    kb_flag = 0x0002 if release else 0
    for k in kb_keys:
        vk = _VK.get(k.lower(), 0)
        if vk:
            inp = _INPUT(type=1)
            inp._u.ki.wVk = vk
            inp._u.ki.dwFlags = kb_flag
            items.append(inp)
    for btn in mouse_btns:
        inp = _INPUT(type=0)
        inp._u.mi.dwFlags = _MOUSE_UP[btn] if release else _MOUSE_DN[btn]
        items.append(inp)
    if items:
        arr = (_INPUT * len(items))(*items)
        ctypes.windll.user32.SendInput(len(items), arr, ctypes.sizeof(_INPUT))
