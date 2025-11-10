#!/usr/bin/env bash

#SBATCH -A cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --nodelist=cn47,cn48
#SBATCH -J stage-isic
#SBATCH --output=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out
#SBATCH --error=/home/knarwani/thesis/git/Adaptive-Depth-U-Net-for-Image-Super-Resolution-Segmentation/Segmenation/logs/slurm-%x-%j.out

set -euo pipefail

SOURCE_SPEC="${SOURCE_SPEC:-${SOURCE_DIR:-}}"
if [[ -z "$SOURCE_SPEC" ]]; then
  echo "[error] SOURCE_SPEC (or legacy SOURCE_DIR) environment variable is required." >&2
  exit 1
fi

if [[ "$SOURCE_SPEC" == *:* ]]; then
  SOURCE_KIND="remote"
else
  SOURCE_KIND="local"
  if [[ ! -d "$SOURCE_SPEC" ]]; then
    echo "[error] SOURCE_DIR does not exist: $SOURCE_SPEC" >&2
    exit 1
  fi
fi

TARGET_ROOT="/scratch/knarwani/Final_data/Segmenation/ISIC-2017"
NODES=(cn47 cn48)

export SOURCE_SPEC SOURCE_KIND TARGET_ROOT

sync_node() {
  local node="$1"
  echo "[stage] Syncing dataset to $node:$TARGET_ROOT"
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 --exclusive -w "$node" bash -c '
    set -euo pipefail
    if [[ "$SOURCE_KIND" == "local" ]] && [[ ! -d "$SOURCE_SPEC" ]]; then
      echo "[error] SOURCE_DIR missing inside task: $SOURCE_SPEC" >&2
      exit 1
    fi
    mkdir -p "$TARGET_ROOT"
    rsync -aL --delete --info=progress2 \
      -e "ssh -o IdentitiesOnly=yes -o PubkeyAuthentication=no -o PreferredAuthentications=password" \
      "${SOURCE_SPEC%/}"/ "$TARGET_ROOT"/
  '
}

for node in "${NODES[@]}"; do
  sync_node "$node"
done

echo "Finished staging at: $(date)"
