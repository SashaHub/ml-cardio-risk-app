# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app.py', '.'),
        ('src', 'src'),
        ('models', 'models'),
        ('config', 'config')
    ],
    hiddenimports=[
        'streamlit.web',
        'streamlit.web.cli',
        'streamlit.runtime',
        'streamlit.runtime.caching',
        'streamlit.runtime.legacy_caching',
        'plotly',
        'altair',
        'pandas',
        'numpy',
        'shap',
        'catboost'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CardioRiskApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CardioRiskApp',
)

app = BUNDLE(
    coll,
    name='CardioRiskApp.app',
    icon=None,
    bundle_identifier=None,
)
