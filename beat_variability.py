#!/usr/bin/env python3
"""
Block D: Beat-to-Beat Morphological Variability
=================================================
For each ECG recording:
1. R-peak detection via scipy.signal.find_peaks
2. Segment into individual beats
3. Compute mean template
4. Beat-to-beat correlation with template → morphological variability
5. Plot vs age, vs sigma_beta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal, stats
from pathlib import Path
import wfdb
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

DATA_DIR = Path('ptb-xl')
FS = 500  # Sampling rate


def detect_rpeaks(ecg_lead, fs=500):
    """Detect R-peaks using bandpass + find_peaks."""
    # Bandpass 5-30 Hz to isolate QRS
    sos = signal.butter(4, [5, 30], btype='band', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, ecg_lead)

    # Square to emphasize QRS
    squared = filtered ** 2

    # Moving average (150ms window)
    window = int(0.15 * fs)
    ma = np.convolve(squared, np.ones(window)/window, mode='same')

    # Find peaks with minimum distance of 300ms
    min_dist = int(0.3 * fs)
    height_thresh = np.percentile(ma, 70)
    peaks, _ = signal.find_peaks(ma, distance=min_dist, height=height_thresh)

    return peaks


def compute_beat_variability(ecg_lead, fs=500, lead_name='II'):
    """
    Compute beat-to-beat morphological variability for one lead.
    Returns: dict with metrics or None if failed.
    """
    rpeaks = detect_rpeaks(ecg_lead, fs)

    if len(rpeaks) < 5:
        return None

    # Segment beats: window = [-200ms, +400ms] around R-peak
    pre = int(0.2 * fs)   # 100 samples
    post = int(0.4 * fs)  # 200 samples
    beat_len = pre + post

    beats = []
    for rp in rpeaks:
        start = rp - pre
        end = rp + post
        if start < 0 or end > len(ecg_lead):
            continue
        beat = ecg_lead[start:end]
        # Normalize each beat to zero mean, unit std
        if np.std(beat) > 0:
            beat = (beat - np.mean(beat)) / np.std(beat)
            beats.append(beat)

    if len(beats) < 5:
        return None

    beats = np.array(beats)

    # Template = median beat (robust to outliers)
    template = np.median(beats, axis=0)

    # Correlation of each beat with template
    corrs = np.array([np.corrcoef(b, template)[0, 1] for b in beats])

    # Morphological variability = 1 - mean(correlation)
    morph_var = 1 - np.mean(corrs)

    # Also compute beat-to-beat RR intervals
    rr_intervals = np.diff(rpeaks) / fs  # in seconds
    rr_cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else np.nan

    # Cosine similarity between consecutive beats PSD
    psd_similarities = []
    for i in range(len(beats) - 1):
        f1, p1 = signal.welch(beats[i], fs=fs, nperseg=min(128, len(beats[i])))
        f2, p2 = signal.welch(beats[i+1], fs=fs, nperseg=min(128, len(beats[i+1])))
        cos_sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10)
        psd_similarities.append(cos_sim)
    spectral_var = 1 - np.mean(psd_similarities) if psd_similarities else np.nan

    return {
        'n_beats': len(beats),
        'morph_var': morph_var,
        'mean_corr': np.mean(corrs),
        'std_corr': np.std(corrs),
        'min_corr': np.min(corrs),
        'spectral_var': spectral_var,
        'rr_mean': np.mean(rr_intervals),
        'rr_cv': rr_cv,
        'hr': 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else np.nan,
    }


def process_records(n_sample=3000, seed=42):
    """Process a sample of NORM records for beat variability."""
    # Load metadata
    meta = pd.read_csv(DATA_DIR / 'ptbxl_database.csv', index_col=0)
    beta_df = pd.read_csv('results/beta_features.csv', index_col=0)

    # Filter NORM
    norm_mask = meta.scp_codes.str.contains('NORM', na=False)
    age_mask = meta.age.between(18, 95)
    valid = meta[norm_mask & age_mask].index

    # Sample
    np.random.seed(seed)
    if len(valid) > n_sample:
        sample_ids = np.random.choice(valid, n_sample, replace=False)
    else:
        sample_ids = valid.values
    print(f"Processing {len(sample_ids)} NORM records...")

    results = []
    for i, ecg_id in enumerate(sample_ids):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(sample_ids)}...")

        # Load record
        row = meta.loc[ecg_id]
        fname = row.filename_hr
        fpath = str(DATA_DIR / fname)

        try:
            record = wfdb.rdrecord(fpath)
            ecg = record.p_signal  # (5000, 12)
        except Exception:
            continue

        # Process Lead II (index 1)
        lead_ii = ecg[:, 1]
        result = compute_beat_variability(lead_ii, FS, 'II')
        if result is None:
            continue

        # Also process V1 (index 6) for regional comparison
        lead_v1 = ecg[:, 6]
        result_v1 = compute_beat_variability(lead_v1, FS, 'V1')

        # Merge with metadata
        entry = {
            'ecg_id': ecg_id,
            'age': row.age,
            'sex': row.sex,
        }
        # Add lead II metrics
        for k, v in result.items():
            entry[f'II_{k}'] = v
        # Add V1 metrics
        if result_v1:
            for k, v in result_v1.items():
                entry[f'V1_{k}'] = v

        # Add beta features if available
        if ecg_id in beta_df.index:
            entry['beta_mean'] = beta_df.loc[ecg_id, 'beta_mean']
            entry['beta_std'] = beta_df.loc[ecg_id, 'beta_std']

        results.append(entry)

    df = pd.DataFrame(results)
    df.to_csv('results/block_D_beat_variability.csv', index=False)
    print(f"Processed {len(df)} records successfully")
    return df


def make_figures(df):
    """Generate Block D figures."""
    print("\n--- Generating figures ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Panel A: Morphological variability vs age ──
    ax = axes[0, 0]
    valid = df.dropna(subset=['age', 'II_morph_var'])
    ax.scatter(valid.age, valid.II_morph_var, alpha=0.15, s=8, c='#1565C0')

    # LOWESS trend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw = lowess(valid.II_morph_var.values, valid.age.values, frac=0.3)
    ax.plot(lw[:, 0], lw[:, 1], 'r-', linewidth=2.5, label='LOWESS')

    rho, p = stats.spearmanr(valid.age, valid.II_morph_var)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Morphological Variability\n(1 - mean template corr)')
    ax.set_title(f'A. Beat Variability vs Age (Lead II)\nρ={rho:.3f}, p={p:.1e}',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel B: Morphological variability vs sigma_beta ──
    ax = axes[0, 1]
    valid2 = df.dropna(subset=['II_morph_var', 'beta_std'])
    ax.scatter(valid2.beta_std, valid2.II_morph_var, alpha=0.15, s=8, c='#E53935')
    rho2, p2 = stats.spearmanr(valid2.beta_std, valid2.II_morph_var)
    lw2 = lowess(valid2.II_morph_var.values, valid2.beta_std.values, frac=0.3)
    ax.plot(lw2[:, 0], lw2[:, 1], 'k-', linewidth=2.5, label='LOWESS')
    ax.set_xlabel('σ_β (inter-lead SD)')
    ax.set_ylabel('Morphological Variability')
    ax.set_title(f'B. Beat Var. vs σ_β\nρ={rho2:.3f}, p={p2:.1e}',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel C: By sex ──
    ax = axes[0, 2]
    for sex, color, label in [(0, '#1565C0', 'Male'), (1, '#E91E63', 'Female')]:
        sub = df[df.sex == sex].dropna(subset=['age', 'II_morph_var'])
        lw_s = lowess(sub.II_morph_var.values, sub.age.values, frac=0.3)
        ax.plot(lw_s[:, 0], lw_s[:, 1], '-', color=color, linewidth=2.5, label=label)
        ax.scatter(sub.age, sub.II_morph_var, alpha=0.08, s=6, c=color)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Morphological Variability')
    ax.set_title('C. Sex Dimorphism', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel D: Spectral variability vs age ──
    ax = axes[1, 0]
    valid3 = df.dropna(subset=['age', 'II_spectral_var'])
    ax.scatter(valid3.age, valid3.II_spectral_var, alpha=0.15, s=8, c='#43A047')
    lw3 = lowess(valid3.II_spectral_var.values, valid3.age.values, frac=0.3)
    ax.plot(lw3[:, 0], lw3[:, 1], 'r-', linewidth=2.5, label='LOWESS')
    rho3, p3 = stats.spearmanr(valid3.age, valid3.II_spectral_var)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Spectral Variability\n(1 - mean PSD cosine sim)')
    ax.set_title(f'D. Beat Spectral Var. vs Age\nρ={rho3:.3f}, p={p3:.1e}',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel E: Regional comparison (Lead II vs V1) ──
    ax = axes[1, 1]
    valid4 = df.dropna(subset=['II_morph_var', 'V1_morph_var'])
    ax.scatter(valid4.II_morph_var, valid4.V1_morph_var, alpha=0.15, s=8, c='#FF9800')
    rho4, p4 = stats.spearmanr(valid4.II_morph_var, valid4.V1_morph_var)
    ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Morph. Var. Lead II')
    ax.set_ylabel('Morph. Var. Lead V1')
    ax.set_title(f'E. Regional: Lead II vs V1\nρ={rho4:.3f}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel F: Decade boxplot ──
    ax = axes[1, 2]
    valid5 = df.dropna(subset=['age', 'II_morph_var']).copy()
    valid5['decade'] = (valid5.age // 10) * 10
    decades = sorted(valid5.decade.unique())
    bp_data = [valid5[valid5.decade == d]['II_morph_var'].values for d in decades]

    bp = ax.boxplot(bp_data, positions=range(len(decades)), widths=0.6,
                    patch_artist=True, showfliers=False)
    for patch, d in zip(bp['boxes'], decades):
        c = plt.cm.RdYlBu_r((d - 10) / 80)
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels([f'{int(d)}s' for d in decades])
    ax.set_xlabel('Age Decade')
    ax.set_ylabel('Morphological Variability')
    ax.set_title('F. Decade Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add N per decade
    for i, d in enumerate(decades):
        n = len(valid5[valid5.decade == d])
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=8)

    plt.suptitle('Fig D-1: Beat-to-Beat Morphological Variability — A New Temporal Scale',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/fig_D1_beat_variability.png')
    plt.close()
    print("  Saved fig_D1_beat_variability.png")


def print_stats(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("BLOCK D: SUMMARY STATISTICS")
    print("="*70)

    v = df.dropna(subset=['age', 'II_morph_var'])
    rho, p = stats.spearmanr(v.age, v.II_morph_var)
    print(f"\n  Lead II morph. var. vs age: ρ={rho:.4f}, p={p:.2e}")

    v2 = df.dropna(subset=['beta_std', 'II_morph_var'])
    rho2, p2 = stats.spearmanr(v2.beta_std, v2.II_morph_var)
    print(f"  Lead II morph. var. vs σ_β: ρ={rho2:.4f}, p={p2:.2e}")

    v3 = df.dropna(subset=['age', 'II_spectral_var'])
    rho3, p3 = stats.spearmanr(v3.age, v3.II_spectral_var)
    print(f"  Lead II spectral var. vs age: ρ={rho3:.4f}, p={p3:.2e}")

    # By decade
    v['decade'] = (v.age // 10) * 10
    print("\n  Decade means:")
    for d in sorted(v.decade.unique()):
        sub = v[v.decade == d]
        print(f"    {int(d)}s: {sub.II_morph_var.mean():.4f} ± {sub.II_morph_var.std():.4f} "
              f"(n={len(sub)})")

    # Partial correlation: morph_var vs age, controlling for HR
    v_hr = df.dropna(subset=['age', 'II_morph_var', 'II_hr'])
    from scipy.stats import pearsonr
    # Simple partial: regress out HR
    residuals_mv = v_hr.II_morph_var - np.polyval(np.polyfit(v_hr.II_hr, v_hr.II_morph_var, 1), v_hr.II_hr)
    residuals_age = v_hr.age - np.polyval(np.polyfit(v_hr.II_hr, v_hr.age, 1), v_hr.II_hr)
    rho_partial, p_partial = stats.spearmanr(residuals_age, residuals_mv)
    print(f"\n  Partial corr (age vs morph_var | HR): ρ={rho_partial:.4f}, p={p_partial:.2e}")


def main():
    print("="*70)
    print("BLOCK D: Beat-to-Beat Morphological Variability")
    print("="*70)

    # Check if cached results exist
    cache_path = Path('results/block_D_beat_variability.csv')
    if cache_path.exists():
        print("Loading cached results...")
        df = pd.read_csv(cache_path)
        print(f"  Loaded {len(df)} records")
    else:
        df = process_records(n_sample=3000)

    make_figures(df)
    print_stats(df)


if __name__ == '__main__':
    main()
