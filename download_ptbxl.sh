#!/bin/bash
# Parallel download of PTB-XL 500Hz records from PhysioNet
# Uses xargs with 10 parallel curl processes

BASE_URL="https://physionet.org/files/ptb-xl/1.0.3"
OUT_DIR="ptb-xl"

cd /Users/teo/Desktop/research/heart

# Generate list of all record folders (00000 through 21000, step 1000)
for folder in $(seq -w 0 1000 21000); do
    folder_fmt=$(printf "%05d" $folder)
    mkdir -p "$OUT_DIR/records500/$folder_fmt"
done

# Generate file list: each record has _hr.dat and _hr.hea
# Record IDs go from 00001 to 21837
python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/ptbxl_database.csv', index_col='ecg_id')
for _, row in df.iterrows():
    path = row['filename_hr']  # e.g., records500/00000/00001_hr
    print(f'{path}.dat')
    print(f'{path}.hea')
" > /tmp/ptbxl_filelist.txt

TOTAL=$(wc -l < /tmp/ptbxl_filelist.txt)
echo "Downloading $TOTAL files with 10 parallel connections..."

# Download with xargs parallel
cat /tmp/ptbxl_filelist.txt | xargs -P 10 -I {} bash -c '
    URL="'"$BASE_URL"'/{}"
    OUT="'"$OUT_DIR"'/{}"
    if [ ! -f "$OUT" ]; then
        curl -s -o "$OUT" --create-dirs "$URL"
    fi
'

echo "Download complete!"
find "$OUT_DIR/records500" -name "*.dat" | wc -l
echo "records downloaded."
