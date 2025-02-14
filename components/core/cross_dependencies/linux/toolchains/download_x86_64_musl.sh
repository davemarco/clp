#!/bin/bash

toolchain_URL="https://toolchains.bootlin.com/downloads/releases/toolchains/x86-64/tarballs/x86-64--musl--stable-2023.08-1.tar.bz2"
DOWNLOAD_PATH="x86-64--musl.tar.bz2"

echo "Downloading toolchain from $toolchain_URL ..."
curl -L "$toolchain_URL" -o "$DOWNLOAD_PATH"

if ! command -v tar &> /dev/null; then
    echo "Error: tar is not installed."
    exit 1
fi

echo "Extracting toolchain..."
tar -xjf "$DOWNLOAD_PATH"

#rm "$DOWNLOAD_PATH"
echo "Toolchain successfully installed!"
