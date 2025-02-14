#!/bin/bash

NDK_URL="https://dl.google.com/android/repository/android-ndk-r25c-linux.zip"
DOWNLOAD_PATH="android-ndk.zip"

echo "Downloading NDK from $NDK_URL ..."
curl -L "$NDK_URL" -o "$DOWNLOAD_PATH"

if ! command -v unzip &> /dev/null; then
    echo "Error: unzip is not installed."
    exit 1
fi

echo "Unzipping NDK..."
unzip "$DOWNLOAD_PATH"

rm "$DOWNLOAD_PATH"
echo "NDK successfully installed!"