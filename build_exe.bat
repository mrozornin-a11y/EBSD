@echo off
rem ======================================================================
rem  Сборка Windows EXE (PyInstaller). Требуется Python 3.11+ в PATH.
rem  Результат: dist\EBSD_PoleFigures\EBSD_PoleFigures.exe
rem ======================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Создание виртуального окружения .venv ...
    python -m venv .venv
    if errorlevel 1 goto :error
)

echo [2/4] Установка зависимостей ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 goto :error

echo [3/4] Проверка (автотесты) ...
".venv\Scripts\python.exe" -m pytest -q tests
if errorlevel 1 (
    echo ВНИМАНИЕ: тесты не пройдены. Сборка продолжится, но проверьте вывод выше.
)

echo [4/4] Сборка EXE ...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean ebsd_pf.spec
if errorlevel 1 goto :error

echo.
echo Готово: dist\EBSD_PoleFigures\EBSD_PoleFigures.exe
echo Для переноса на другой компьютер скопируйте всю папку dist\EBSD_PoleFigures.
endlocal
exit /b 0

:error
echo.
echo ОШИБКА сборки. Подробности см. выше.
endlocal
exit /b 1
