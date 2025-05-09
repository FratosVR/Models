
@echo off
setlocal

:: === CONFIGURATION ===
set GITHUB_OWNER=FratosVR
set GITHUB_REPO=Models
set RELEASE_TAG=v1
set MODEL_ASSET_NAME=model.zip
set MODEL_NAME=rigardu

:: === GET RELEASE INFO ===
echo Fetching release info from GitHub...
curl -s https://api.github.com/repos/%GITHUB_OWNER%/%GITHUB_REPO%/releases/tags/%RELEASE_TAG% > release.json

:: === EXTRACT DOWNLOAD URL AND VERSION TAG ===
for /f %%A in ('powershell -Command "(Get-Content release.json | ConvertFrom-Json).assets | Where-Object { $_.name -eq '%MODEL_ASSET_NAME%' } | Select-Object -ExpandProperty browser_download_url"') do (
    set "DOWNLOAD_URL=%%A"
)

for /f %%B in ('powershell -Command "(Get-Content release.json | ConvertFrom-Json).tag_name -replace '[^0-9.]',''"') do (
    set "VERSION_NUMBER=%%B"
)

:: === CLEANUP RELEASE FILE ===
del release.json >nul 2>&1

:: === VALIDATE DOWNLOAD URL ===
if "%DOWNLOAD_URL%"=="" (
    echo Failed to extract download URL for %MODEL_ASSET_NAME%.
    exit /b 1
)

:: === DOWNLOAD MODEL ===
echo Downloading model from: %DOWNLOAD_URL%
curl -L -o %MODEL_ASSET_NAME% "%DOWNLOAD_URL%"
if %errorlevel% neq 0 (
    echo Download failed.
    exit /b 1
)

:: === DEFINE EXTRACTION PATH ===
set "EXTRACT_PATH=%MODEL_NAME%\%VERSION_NUMBER%"

:: === CREATE TARGET DIRECTORY ===
mkdir "%EXTRACT_PATH%" >nul 2>&1

:: === EXTRACT MODEL ZIP TO VERSIONED FOLDER ===
echo Extracting model to %EXTRACT_PATH%...
powershell -Command "Expand-Archive -Force '%MODEL_ASSET_NAME%' '%EXTRACT_PATH%'"

:: === OPTIONAL: FLATTEN NESTED DIRECTORY IF ZIP CONTAINS EXTRA LAYER ===
for /d %%D in ("%EXTRACT_PATH%\*") do (
    if exist "%%D\saved_model.pb" (
        echo Flattening nested directory...
        move "%%D\*" "%EXTRACT_PATH%\" >nul
        rd /s /q "%%D"
        goto :flattened
    )
)
:flattened

:: === PREPARE PATH FOR DOCKER (mount base model folder only) ===
set "MODEL_PATH=%cd:\=/%/%MODEL_NAME%"

:: === SERVE MODEL WITH TENSORFLOW SERVING ===
echo Launching TensorFlow Serving...
docker run -p 8501:8501 --mount type=bind,source=%MODEL_PATH%,target=/models/%MODEL_NAME% -e MODEL_NAME=%MODEL_NAME% -t tensorflow/serving

endlocal

