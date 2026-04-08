#!/bin/bash
# Continuously process newly downloaded records until all are done
cd /Users/teo/Desktop/research/heart

while true; do
    TOTAL_DAT=$(find ptb-xl/records500 -name "*.dat" 2>/dev/null | wc -l | tr -d ' ')
    PROCESSED=$(wc -l < results/beta_features_partial.csv 2>/dev/null | tr -d ' ')
    PROCESSED=$((PROCESSED - 1))  # subtract header

    echo "$(date '+%H:%M:%S') | Downloaded: $TOTAL_DAT | Processed: $PROCESSED"

    if [ "$TOTAL_DAT" -ge 21799 ] && [ "$PROCESSED" -ge 21700 ]; then
        echo "All records downloaded and processed!"
        # Rename to final
        cp results/beta_features_partial.csv results/beta_features.csv
        echo "Final results saved to results/beta_features.csv"
        break
    fi

    python3 process_incremental.py 2>&1 | tail -3

    # Wait for more downloads
    sleep 60
done
