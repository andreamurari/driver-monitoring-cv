#!/usr/bin/env bash
# Scarica State Farm Distracted Driver Detection con la Kaggle CLI.
# Prerequisiti (una tantum, a mano):
#   1. account Kaggle e token API in ~/.kaggle/kaggle.json (Settings > API)
#   2. aver accettato le regole su
#      https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules
#      e averle LETTE: copia le condizioni d'uso dei dati in DATA.md
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

kaggle competitions download -c state-farm-distracted-driver-detection -p "$DEST"

cd "$DEST"
# lo zip principale contiene driver_imgs_list.csv e le cartelle imgs/
unzip -q -n state-farm-distracted-driver-detection.zip
# alcune versioni hanno ancora zip annidati: estrai se presenti
for z in driver_imgs_list.csv.zip imgs.zip; do
  [ -f "$z" ] && unzip -q -n "$z"
done

echo "Controlla che esistano: $DEST/driver_imgs_list.csv e $DEST/imgs/train/c0 ... c9"
ls "$DEST" 2>/dev/null || ls .
