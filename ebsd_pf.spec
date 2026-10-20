# -*- mode: python ; coding: utf-8 -*-
# Спецификация PyInstaller: сборка в папку dist\EBSD_PoleFigures (один каталог — быстрый запуск).
# Использование: pyinstaller --noconfirm ebsd_pf.spec
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    collect_submodules("skimage.color")
    + ["PySide6.QtSvg", "PySide6.QtPrintSupport"]
)

excludes = [
    "tkinter", "matplotlib", "IPython", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml",
    "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EBSD_PoleFigures",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EBSD_PoleFigures",
)
