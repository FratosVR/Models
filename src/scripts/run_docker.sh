#!/bin/bash
set -e

# === CONFIGURATION ===
GITHUB_OWNER="FratosVR"
GITHUB_REPO="Models"
RELEASE_TAG="v1"
MODEL_ASSET_NAME="model.zip"
MODEL_NAME="rigardu"

# === GET RELEASE INFO ===
echo "Fetching release info from GitHub..."
curl -s "https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases/tags/${RELEASE_TAG}" -o release.json

# === EXTRACT DOWNLOAD URL AND VERSION TAG ===
DOWNLOAD_URL=$(jq -r --arg name "$MODEL_ASSET_NAME" '.assets[] | select(.name == $name) | .browser_download_url' release.json)
VERSION_NUMBER=$(jq -r '.tag_name' release.json | sed 's/[^0-9.]//g')

# === CLEANUP RELEASE FILE ===
rm -f release.json

# === VALIDATE DOWNLOAD URL ===
if [[ -z "$DOWNLOAD_URL" ]]; then
    echo "Failed to extract download URL for $MODEL_ASSET_NAME."
    exit 1
fi

# === DOWNLOAD MODEL ===
echo "Downloading model from: $DOWNLOAD_URL"
curl -L -o "$MODEL_ASSET_NAME" "$DOWNLOAD_URL"

# === DEFINE EXTRACTION PATH ===
EXTRACT_PATH="${MODEL_NAME}/${VERSION_NUMBER}"

# === CREATE TARGET DIRECTORY ===
mkdir -p "$EXTRACT_PATH"

# === EXTRACT MODEL ZIP TO VERSIONED FOLDER ===
echo "Extracting model to $EXTRACT_PATH..."
unzip -o "$MODEL_ASSET_NAME" -d "$EXTRACT_PATH"

# === OPTIONAL: FLATTEN NESTED DIRECTORY IF ZIP CONTAINS EXTRA LAYER ===
for d in "$EXTRACT_PATH"/*/; do
    if [[ -f "$d/saved_model.pb" ]]; then
        echo "Flattening nested directory..."
        mv "$d"* "$EXTRACT_PATH"/
        rmdir "$d"
        break
    fi
done

# === PREPARE PATH FOR DOCKER ===
MODEL_PATH="$(pwd)/$MODEL_NAME"

# === SERVE MODEL WITH TENSORFLOW SERVING ===
echo "Launching TensorFlow Serving..."
docker run -p 8501:8501 --mount type=bind,source="$MODEL_PATH",target="/models/$MODEL_NAME" -e MODEL_NAME="$MODEL_NAME" -t tensorflow/serving
