@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo.
python main.py dashboard
echo.
echo Fake Review Detector has stopped.
pause
