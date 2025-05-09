@echo off
setlocal

:: === CONFIGURATION ===
set GITHUB_OWNER=FratosVR
set GITHUB_REPO=Models
set RELEASE_TAG=latest
set MODEL_ASSET_NAME=model.zip
set MODEL_NAME=rigardu

:: === GET LATEST RELEASE INFO ===
echo Fetching release info from GitHub...
curl -s https://api.github.com/repos/%GITHUB_OWNER%/%GITHUB_REPO%/releases/%RELEASE_TAG% > release.json

:: === PARSE DOWNLOAD URL ===
for /f "tokens=2 delims=:" %%A in ('findstr /i "browser_download_url.*%MODEL_ASSET_NAME%" release.json') do (
    set "DOWNLOAD_URL=%%A"
)
set DOWNLOAD_URL=%DOWNLOAD_URL:~2,-2%

:: === DOWNLOAD MODEL ===
echo Downloading model from: %DOWNLOAD_URL%
curl -L -o %MODEL_ASSET_NAME% "%DOWNLOAD_URL%"

:: === EXTRACT MODEL ZIP ===
echo Extracting model...
powershell -Command "Expand-Archive -Force '%MODEL_ASSET_NAME%' ."

:: === SERVE MODEL WITH TENSORFLOW SERVING VIA DOCKER ===
echo Launching TensorFlow Serving...
docker run -p 8501:8501 --mount type=bind,source=%cd%\%MODEL_NAME%,target=/models/%MODEL_NAME% -e MODEL_NAME=%MODEL_NAME% -t tensorflow/serving

endlocal
