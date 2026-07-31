# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

# ddddocr 的 OCR 模型（.onnx）是套件內的資料檔，不是 Python 程式碼，
# 不會被 PyInstaller 的 import 分析自動抓到，要用 collect_data_files 明確收集。
ddddocr_datas = collect_data_files('ddddocr')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), ('logo', 'logo')] + ddddocr_datas,
    hiddenimports=['cv2', 'numpy', 'pynput.keyboard._win32', 'pynput.mouse._win32',
                   'ddddocr', 'onnxruntime', 'pyperclip'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='彥彥老字號',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo\\logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='彥彥老字號',
)
