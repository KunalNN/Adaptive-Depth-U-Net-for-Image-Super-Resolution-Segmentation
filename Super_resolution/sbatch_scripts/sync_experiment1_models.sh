#!/usr/bin/env bash

# Synchronise Experiment 1 checkpoints from a remote scratch volume (e.g. cn48)
# into the repository checkout on the current node.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cn48}"
REMOTE_USER="${REMOTE_USER:-knarwani}"
REMOTE_BASE="${REMOTE_BASE:-/scratch/knarwani/Final_data/Super_resolution/experiments/experiment_1_constant_depth_3/models}"

REPO_DIR="${REPO_DIR:-/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation}"
LOCAL_BASE="${LOCAL_BASE:-$REPO_DIR/Super_resolution/models/Experiment_1}"

SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519}"
SSH_OPTIONS="-o IdentitiesOnly=yes -o IdentityAgent=none -o PreferredAuthentications=publickey -o PubkeyAuthentication=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no"
if [[ -f "$SSH_IDENTITY" ]]; then
  SSH_OPTIONS+=" -i $SSH_IDENTITY"
else
  echo "[warn] SSH identity not found at $SSH_IDENTITY; falling back to ssh defaults." >&2
fi

RSYNC_BIN="${RSYNC_BIN:-rsync}"
if ! command -v "$RSYNC_BIN" >/dev/null 2>&1; then
  echo "[error] rsync not available on PATH." >&2
  exit 1
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"

echo "[sync] Remote: $REMOTE"
echo "[sync] Local:  $LOCAL_BASE"

mkdir -p "$LOCAL_BASE"

unset SSH_AUTH_SOCK

set -x
"$RSYNC_BIN" -av --delete --info=progress2 \
  -e "ssh $SSH_OPTIONS" \
  "$REMOTE" \
  "$LOCAL_BASE/"
set +x

echo "[sync] Completed at $(date --iso-8601=seconds)"
