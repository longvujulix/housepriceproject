@echo off
chcp 65001 >nul
echo [NHÀ INSIGHT] Đang kiểm tra môi trường Python trên Windows...

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [LỖI] Không tìm thấy Python trên hệ thống.
    echo Vui lòng cài đặt Python (3.10 - 3.12) và tick chọn "Add Python to PATH".
    pause
    exit /b 1
)

python -c "import streamlit, pandas, scipy, sklearn, pyarrow, numpy" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Đang cài đặt các thư viện cần thiết từ requirements.txt...
    python -m pip install -r requirements.txt
) else (
    echo ✓ Tất cả thư viện đã có sẵn trên máy.
)

echo.
echo ✓ Khởi động ứng dụng Streamlit...
python -m streamlit run app.py
