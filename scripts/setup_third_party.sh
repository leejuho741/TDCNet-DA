#!/usr/bin/env bash
# Clone the ReMake repository (MIT). It ships the TDCNet model code, the official TransCG loader used for
# depth-model training, and the ReMake baseline. Pinned to the commit used for every result in this repo.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REMAKE_REPO:-$ROOT/third_party/ReMake}"
if [ ! -d "$DEST" ]; then
  git clone -q https://github.com/ChengYaofeng/ReMake.git "$DEST"
fi
git -C "$DEST" checkout -q 4f56814
echo "ReMake at $DEST ($(git -C "$DEST" log -1 --format='%h %cd'))"
# packages its modules import at import time (some are only needed for training / the ReMake baseline)
pip -q install -r "$ROOT/requirements-third-party.txt"
