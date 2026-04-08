"""
Incremental processing: compute β for newly downloaded records,
merge with existing partial results.
"""
import os, ast, numpy as np, pandas as pd, wfdb
from scipy import signal, stats
from neurodsp.aperiodic import compute_irasa
from specparam import SpectralModel
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'ptb-xl'
RESULTS_DIR = 'results'
FS = 500
IRASA_HSET = np.arange(1.1, 1.95, 0.05)
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
N_JOBS = 6

def preprocess(s):
    sos = signal.butter(4, [0.5, 100], btype='bandpass', fs=FS, output='sos')
    out = signal.sosfiltfilt(sos, s)
    b, a = signal.iirnotch(50, Q=30, fs=FS)
    return signal.filtfilt(b, a, out)

def beta_irasa(s):
    try:
        f, pa, pp = compute_irasa(s, fs=FS, f_range=(0.5, 50), hset=IRASA_HSET)
        m = (f >= 2) & (f <= 45) & (pa > 0)
        if m.sum() < 5: return np.nan, np.nan
        sl, ic, rv, _, _ = stats.linregress(np.log10(f[m]), np.log10(pa[m]))
        return -sl, rv**2
    except: return np.nan, np.nan

def beta_sp(s):
    try:
        f, p = signal.welch(s, fs=FS, nperseg=1000, noverlap=500)
        sm = SpectralModel(peak_width_limits=[1,8], max_n_peaks=6,
                          min_peak_height=0.1, aperiodic_mode='fixed')
        sm.fit(f, p, freq_range=[1, 50])
        ap = sm.get_params('aperiodic')
        return ap[-1]
    except: return np.nan

def process_one(ecg_id, path):
    try:
        fp = os.path.join(DATA_DIR, path)
        if not os.path.exists(fp + '.dat'):
            return None
        rec = wfdb.rdrecord(fp)
        ecg = rec.p_signal
        if ecg is None or ecg.shape[0] < 5000:
            return None
        result = {'ecg_id': ecg_id}
        betas_ir, r2s, betas_sp = [], [], []
        for i, lead in enumerate(LEAD_NAMES):
            s = preprocess(ecg[:, i])
            b_ir, r2 = beta_irasa(s)
            b_s = beta_sp(s)
            result[f'beta_ir_{lead}'] = b_ir
            result[f'r2_ir_{lead}'] = r2
            result[f'beta_sp_{lead}'] = b_s
            betas_ir.append(b_ir); r2s.append(r2); betas_sp.append(b_s)
        arr = np.array(betas_ir)
        if (~np.isnan(arr)).sum() < 6: return None
        result['beta_mean'] = np.nanmean(arr)
        result['beta_std'] = np.nanstd(arr)
        result['beta_median'] = np.nanmedian(arr)
        result['delta'] = abs(np.nanmean(arr) - 1.0)
        result['r2_mean'] = np.nanmean(r2s)
        result['n_valid'] = int((~np.isnan(arr)).sum())
        result['beta_sp_mean'] = np.nanmean(betas_sp)
        return result
    except: return None


if __name__ == '__main__':
    df = pd.read_csv(f'{DATA_DIR}/ptbxl_database.csv', index_col='ecg_id')
    records = [(idx, row.filename_hr) for idx, row in df.iterrows()
               if pd.notna(row.filename_hr)]

    # Load existing results
    cache_path = f'{RESULTS_DIR}/beta_features_partial.csv'
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path, index_col='ecg_id')
        done_ids = set(existing.index)
        print(f"Existing: {len(existing)} records")
    else:
        existing = pd.DataFrame()
        done_ids = set()

    # Find new available records
    new_records = [(eid, p) for eid, p in records
                   if eid not in done_ids and os.path.exists(os.path.join(DATA_DIR, p + '.dat'))]
    print(f"New records to process: {len(new_records)}")

    if len(new_records) == 0:
        print("Nothing new to process.")
    else:
        results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(process_one)(eid, p) for eid, p in new_records)
        results = [r for r in results if r is not None]
        new_df = pd.DataFrame(results).set_index('ecg_id')
        print(f"Successfully processed: {len(new_df)}")

        # Merge with existing
        combined = pd.concat([existing, new_df])
        combined.to_csv(cache_path)
        print(f"Total saved: {len(combined)} records → {cache_path}")
        print(f"β_mean: {combined.beta_mean.mean():.3f} ± {combined.beta_mean.std():.3f}")
