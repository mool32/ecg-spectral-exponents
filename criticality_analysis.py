"""
The Heartbeat at the Edge of Chaos: Criticality Analysis of 21,000 ECGs
=========================================================================
PTB-XL dataset analysis through the lens of criticality theory.

Central thesis: A healthy heart operates near a critical state (β ≈ 1, pink noise).
Disease systematically shifts β away from criticality in diagnosis-specific patterns.
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import wfdb
from neurodsp.aperiodic import compute_irasa
from specparam import SpectralModel
import pingouin as pg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ptb-xl')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FS = 500
DURATION = 10
N_SAMPLES = FS * DURATION  # 5000

# IRASA / fitting parameters
F_MIN = 2.0
F_MAX = 45.0
F_MIN_CONSERVATIVE = 20.0
IRASA_HSET = np.arange(1.1, 1.95, 0.05)

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SC_COLORS = {
    'NORM': '#2ecc71', 'MI': '#e74c3c', 'STTC': '#3498db',
    'CD': '#9b59b6', 'HYP': '#f39c12'
}

N_JOBS = 6  # parallel workers


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_metadata():
    """Load PTB-XL metadata and parse diagnostic labels."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)
    scp = scp[scp.diagnostic == 1]

    def get_superclasses(codes):
        result = set()
        for code, lk in codes.items():
            if code in scp.index and lk > 0:
                sc = scp.loc[code, 'diagnostic_class']
                if pd.notna(sc):
                    result.add(sc)
        return result

    def get_primary_superclass(codes):
        best_sc, best_lk = None, 0
        for code, lk in codes.items():
            if code in scp.index and lk > best_lk:
                sc = scp.loc[code, 'diagnostic_class']
                if pd.notna(sc):
                    best_sc, best_lk = sc, lk
        return best_sc

    def get_subclasses(codes):
        result = {}
        for code, lk in codes.items():
            if code in scp.index and lk > 0:
                result[code] = lk
        return result

    df['superclasses'] = df.scp_codes.apply(get_superclasses)
    df['n_superclasses'] = df.superclasses.apply(len)
    df['primary_superclass'] = df.scp_codes.apply(get_primary_superclass)
    df['max_likelihood'] = df.scp_codes.apply(lambda c: max(c.values()) if c else 0)
    df['is_clean'] = df.n_superclasses == 1
    df['clean_superclass'] = df.apply(
        lambda r: list(r.superclasses)[0] if r.is_clean else None, axis=1)
    df['subclasses'] = df.scp_codes.apply(get_subclasses)

    print(f"Loaded {len(df)} records, {df.patient_id.nunique()} patients")
    print(f"Clean records: {df.is_clean.sum()}")
    print(df[df.is_clean].clean_superclass.value_counts().to_string())
    return df, scp


# ============================================================================
# 2. SIGNAL PROCESSING & β COMPUTATION
# ============================================================================
def preprocess(sig_1d, fs=FS):
    """Bandpass 0.5–100 Hz + 50 Hz notch."""
    sos = signal.butter(4, [0.5, 100], btype='bandpass', fs=fs, output='sos')
    out = signal.sosfiltfilt(sos, sig_1d)
    b, a = signal.iirnotch(50, Q=30, fs=fs)
    out = signal.filtfilt(b, a, out)
    return out


def compute_beta_irasa(sig_1d, fs=FS):
    """Approach A: IRASA → log-log linear fit [2–45 Hz]."""
    try:
        freqs, pa, pp = compute_irasa(
            sig_1d, fs=fs, f_range=(0.5, 50), hset=IRASA_HSET)
        mask = (freqs >= F_MIN) & (freqs <= F_MAX) & (pa > 0)
        if mask.sum() < 5:
            return np.nan, np.nan
        log_f = np.log10(freqs[mask])
        log_p = np.log10(pa[mask])
        sl, ic, rv, pv, se = stats.linregress(log_f, log_p)
        return -sl, rv**2
    except Exception:
        return np.nan, np.nan


def compute_beta_specparam(sig_1d, fs=FS):
    """Approach B: specparam (FOOOF) parametric fit."""
    try:
        freqs, psd = signal.welch(sig_1d, fs=fs, nperseg=fs*2, noverlap=fs)
        sm = SpectralModel(
            peak_width_limits=[1, 8], max_n_peaks=6,
            min_peak_height=0.1, aperiodic_mode='fixed')
        sm.fit(freqs, psd, freq_range=[1, 50])
        ap = sm.get_params('aperiodic')
        return ap[-1], sm.r_squared_ if hasattr(sm, 'r_squared_') else np.nan
    except Exception:
        return np.nan, np.nan


def compute_beta_aboveknee(sig_1d, fs=FS):
    """Approach C: raw PSD log-log fit above knee [20–45 Hz]."""
    try:
        freqs, psd = signal.welch(sig_1d, fs=fs, nperseg=fs*2, noverlap=fs)
        mask = (freqs >= F_MIN_CONSERVATIVE) & (freqs <= F_MAX) & (psd > 0)
        if mask.sum() < 5:
            return np.nan, np.nan
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask])
        sl, ic, rv, pv, se = stats.linregress(log_f, log_p)
        return -sl, rv**2
    except Exception:
        return np.nan, np.nan


def process_record(ecg_id, record_path):
    """Process one ECG: compute β for all 12 leads with all 3 methods."""
    try:
        full_path = os.path.join(DATA_DIR, record_path)
        if not os.path.exists(full_path + '.dat'):
            return None
        rec = wfdb.rdrecord(full_path)
        ecg = rec.p_signal
        if ecg is None or ecg.shape[0] < N_SAMPLES:
            return None

        result = {'ecg_id': ecg_id}
        betas_ir, r2s_ir, betas_sp, betas_ak = [], [], [], []

        for i, lead in enumerate(LEAD_NAMES):
            sig = preprocess(ecg[:, i])

            b_ir, r2_ir = compute_beta_irasa(sig)
            b_sp, _ = compute_beta_specparam(sig)
            b_ak, _ = compute_beta_aboveknee(sig)

            result[f'beta_ir_{lead}'] = b_ir
            result[f'r2_ir_{lead}'] = r2_ir
            result[f'beta_sp_{lead}'] = b_sp
            result[f'beta_ak_{lead}'] = b_ak

            betas_ir.append(b_ir)
            r2s_ir.append(r2_ir)
            betas_sp.append(b_sp)
            betas_ak.append(b_ak)

        arr = np.array(betas_ir)
        valid = ~np.isnan(arr)
        if valid.sum() < 6:
            return None

        result['beta_mean'] = np.nanmean(arr)
        result['beta_std'] = np.nanstd(arr)
        result['beta_median'] = np.nanmedian(arr)
        result['delta'] = abs(np.nanmean(arr) - 1.0)
        result['r2_mean'] = np.nanmean(r2s_ir)
        result['n_valid'] = int(valid.sum())
        result['beta_sp_mean'] = np.nanmean(betas_sp)
        result['beta_ak_mean'] = np.nanmean(betas_ak)

        return result
    except Exception:
        return None


# ============================================================================
# 3. BATCH PROCESSING
# ============================================================================
def process_all(df):
    """Process all available records in parallel. Caches results."""
    cache = os.path.join(RESULTS_DIR, 'beta_features.csv')
    if os.path.exists(cache):
        print(f"Loading cached results from {cache}")
        return pd.read_csv(cache, index_col='ecg_id')

    records = [(idx, row.filename_hr) for idx, row in df.iterrows()
               if pd.notna(row.filename_hr)]

    print(f"Processing {len(records)} records ({N_JOBS} workers)...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_record)(eid, path) for eid, path in records)

    results = [r for r in results if r is not None]
    beta_df = pd.DataFrame(results).set_index('ecg_id')
    beta_df.to_csv(cache)
    print(f"Done: {len(beta_df)} records → {cache}")
    return beta_df


# ============================================================================
# 4. PART 1 — β LANDSCAPE BY DIAGNOSTIC CATEGORY
# ============================================================================
def part1_beta_landscape(dm):
    """Violin plots + statistics for β across superclasses."""
    clean = dm[dm.is_clean & dm.beta_mean.notna()].copy()

    # Compute β_NORM reference
    beta_norm_ref = clean[clean.clean_superclass == 'NORM'].beta_mean.median()
    print(f"  β_NORM reference (median): {beta_norm_ref:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Violin plot ---
    ax = axes[0]
    data_groups = []
    labels = []
    for sc in SUPERCLASSES:
        vals = clean[clean.clean_superclass == sc].beta_mean.dropna()
        data_groups.append(vals.values)
        labels.append(f'{sc}\n(n={len(vals)})')

    parts = ax.violinplot(data_groups, positions=range(len(SUPERCLASSES)),
                          showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(SC_COLORS.values())[i])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('blue')

    ax.axhline(y=beta_norm_ref, color='green', ls='--', lw=2, alpha=0.7,
               label=f'β_NORM = {beta_norm_ref:.2f}')
    ax.axhline(y=1.0, color='red', ls=':', lw=1, alpha=0.5, label='β = 1 (1/f pink noise)')
    ax.set_xticks(range(len(SUPERCLASSES)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('β (spectral exponent)', fontsize=13)
    ax.set_title('Figure 1: Criticality Landscape', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # --- Statistics ---
    ax2 = axes[1]
    stat_kw, p_kw = stats.kruskal(*data_groups)

    txt = f"Kruskal-Wallis H = {stat_kw:.1f}, p = {p_kw:.2e}\n\n"
    txt += "Group statistics (median [IQR]):\n"
    for sc in SUPERCLASSES:
        vals = clean[clean.clean_superclass == sc].beta_mean
        q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
        txt += f"  {sc:5s}: {q50:.3f} [{q25:.3f}–{q75:.3f}] n={len(vals)}\n"

    txt += "\nPairwise Cohen's d (Bonferroni adjusted):\n"
    pairs = []
    for i in range(len(SUPERCLASSES)):
        for j in range(i+1, len(SUPERCLASSES)):
            d = pg.compute_effsize(data_groups[i], data_groups[j], eftype='cohen')
            _, p = stats.mannwhitneyu(data_groups[i], data_groups[j])
            pairs.append((SUPERCLASSES[i], SUPERCLASSES[j], d, p))
    n_comp = len(pairs)
    for g1, g2, d, p in pairs:
        p_adj = min(p * n_comp, 1.0)
        sig = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
        txt += f"  {g1:4s} vs {g2:4s}: d={d:+.3f} {sig}\n"

    ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, fontsize=9.5,
             va='top', fontfamily='monospace')
    ax2.axis('off')
    ax2.set_title('Statistical Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig1_criticality_landscape.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_criticality_landscape.png")
    return pairs


# ============================================================================
# 5. PART 2 — β BY SUBCLASSES (HEATMAP)
# ============================================================================
def part2_subclass_heatmap(dm, scp_df):
    """Heatmap: rows = subclasses, cols = leads, color = median β."""
    beta_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]

    rows = []
    for idx, rec in dm.iterrows():
        if pd.isna(rec.beta_mean):
            continue
        for code, lk in rec.subclasses.items():
            if code in scp_df.index and lk >= 80:
                row = {'subclass': code,
                       'superclass': scp_df.loc[code, 'diagnostic_class']}
                for c in beta_cols:
                    if c in rec.index:
                        row[c] = rec[c]
                rows.append(row)

    sub = pd.DataFrame(rows)
    if sub.empty:
        print("  No subclass data; skipping.")
        return

    counts = sub.subclass.value_counts()
    valid_subs = counts[counts >= 30].index
    sub = sub[sub.subclass.isin(valid_subs)]

    hm = sub.groupby('subclass')[beta_cols].median()
    hm.columns = LEAD_NAMES
    hm = hm.loc[hm.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(14, max(6, len(hm) * 0.45)))
    # Center heatmap on NORM median β
    beta_norm_ref = dm[dm.is_clean & (dm.clean_superclass == 'NORM')].beta_mean.median()
    sns.heatmap(hm, center=beta_norm_ref, cmap='RdBu_r', annot=True, fmt='.2f',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Median β'})
    ax.set_title('Figure 2: Spatial β Fingerprint by Subclass',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Diagnostic Subclass')
    ax.set_xlabel('ECG Lead')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig2_spatial_fingerprint.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_spatial_fingerprint.png")


# ============================================================================
# 6. PART 3 — CROSS-LEAD β COHERENCE
# ============================================================================
def part3_cross_lead_coherence(dm):
    """Correlation matrices of β across leads for each superclass."""
    beta_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]
    clean = dm[dm.is_clean & dm.beta_mean.notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, sc in enumerate(SUPERCLASSES):
        ax = axes.flat[i]
        grp = clean[clean.clean_superclass == sc][beta_cols].dropna()
        if len(grp) < 10:
            ax.set_title(f'{sc} (n<10)')
            continue
        corr = grp.corr()
        corr.index = corr.columns = LEAD_NAMES
        sns.heatmap(corr, vmin=0, vmax=1, cmap='YlOrRd', annot=True, fmt='.2f',
                    square=True, ax=ax, cbar_kws={'shrink': 0.7})
        ax.set_title(f'{sc} (n={len(grp)})', fontsize=13, fontweight='bold')

    # Difference NORM − pathology
    ax = axes.flat[5]
    norm_corr = clean[clean.clean_superclass == 'NORM'][beta_cols].dropna().corr()
    path_corr = clean[clean.clean_superclass != 'NORM'][beta_cols].dropna().corr()
    diff = norm_corr - path_corr
    diff.index = diff.columns = LEAD_NAMES
    sns.heatmap(diff, center=0, cmap='RdBu_r', annot=True, fmt='.2f',
                square=True, ax=ax, cbar_kws={'shrink': 0.7})
    ax.set_title('NORM − Pathology', fontsize=13, fontweight='bold')

    plt.suptitle('Figure 3: Cross-Lead β Coherence', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, 'fig3_cross_lead_coherence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # σ_β histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    for sc in SUPERCLASSES:
        vals = clean[clean.clean_superclass == sc].beta_std.dropna()
        ax.hist(vals, bins=50, alpha=0.5, label=f'{sc} (n={len(vals)})',
                color=SC_COLORS[sc], density=True)
    ax.set_xlabel('σ_β (cross-lead dispersion)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Cross-Lead β Dispersion', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig3b_beta_dispersion.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_cross_lead_coherence.png, fig3b_beta_dispersion.png")


# ============================================================================
# 7. PART 4 — |β - 1| AS SEVERITY MARKER
# ============================================================================
def part4_distance_criticality(dm):
    """Scatter: |β − β_NORM| vs diagnostic likelihood."""
    v = dm[dm.beta_mean.notna()].copy()
    # Use NORM median as reference
    beta_norm_ref = dm[dm.is_clean & (dm.clean_superclass == 'NORM')].beta_mean.median()
    v['delta'] = np.abs(v.beta_mean - beta_norm_ref)
    v['delta_from_1'] = np.abs(v.beta_mean - 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for sc in SUPERCLASSES:
        s = v[v.primary_superclass == sc]
        if len(s) > 0:
            ax.scatter(s.max_likelihood, s.delta, alpha=0.12, s=8,
                       color=SC_COLORS[sc], label=sc)
    ax.set_xlabel('Diagnostic Likelihood (%)', fontsize=12)
    ax.set_ylabel('|β − 1|  (distance from criticality)', fontsize=12)
    ax.set_title('Figure 4: Distance from Criticality vs Confidence',
                 fontsize=14, fontweight='bold')
    ax.legend(markerscale=4)

    ax2 = axes[1]
    txt = "Spearman ρ  (δ vs likelihood):\n\n"
    for sc in SUPERCLASSES:
        s = v[v.primary_superclass == sc].dropna(subset=['delta', 'max_likelihood'])
        if len(s) > 20:
            rho, p = stats.spearmanr(s.max_likelihood, s.delta)
            txt += f"  {sc:5s}: ρ={rho:+.3f}  p={p:.2e}  n={len(s)}\n"

    rho_all, p_all = stats.spearmanr(v.max_likelihood, v.delta)
    txt += f"\n  ALL  : ρ={rho_all:+.3f}  p={p_all:.2e}  n={len(v)}\n"

    # Combined marker
    v['combined'] = v.delta + v.beta_std
    rho_c, p_c = stats.spearmanr(v.max_likelihood.dropna(), v.combined.dropna())
    txt += f"\n  Combined (δ+σ_β): ρ={rho_c:+.3f}  p={p_c:.2e}\n"

    ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, fontsize=11,
             va='top', fontfamily='monospace')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig4_distance_criticality.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_distance_criticality.png")


# ============================================================================
# 8. PART 5 — PREDICTIVE VALIDATION
# ============================================================================
def part5_prediction(dm):
    """Logistic regression: NORM vs pathology with incremental β features."""
    beta_lead_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]
    clean = dm[dm.is_clean & dm.beta_mean.notna()].copy()
    clean['is_norm'] = (clean.clean_superclass == 'NORM').astype(int)
    clean['sex_num'] = clean.sex.astype(float)

    train = clean[clean.strat_fold.isin(range(1, 9))]
    test = clean[clean.strat_fold.isin([9, 10])]
    print(f"  Train n={len(train)} (NORM {train.is_norm.sum()}), "
          f"Test n={len(test)} (NORM {test.is_norm.sum()})")

    models = {
        'Baseline (age+sex)':     ['age', 'sex_num'],
        '+ β_mean':               ['age', 'sex_num', 'beta_mean'],
        '+ β_mean + δ':           ['age', 'sex_num', 'beta_mean', 'delta'],
        '+ β_mean + δ + σ_β':    ['age', 'sex_num', 'beta_mean', 'delta', 'beta_std'],
        '+ all β leads':          ['age', 'sex_num', 'beta_mean', 'delta', 'beta_std'] + beta_lead_cols,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_roc, ax_txt = axes
    txt = "AUC-ROC (NORM vs Pathology):\n\n"

    for name, feats in models.items():
        avail = [f for f in feats if f in train.columns]
        tr = train[avail + ['is_norm']].dropna()
        te = test[avail + ['is_norm']].dropna()
        if len(tr) < 50 or len(te) < 20:
            continue

        X_tr, y_tr = tr[avail], tr.is_norm
        X_te, y_te = te[avail], te.is_norm

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr_s, y_tr)
        y_prob = clf.predict_proba(X_te_s)[:, 1]
        auc_val = roc_auc_score(y_te, y_prob)
        fpr, tpr, _ = roc_curve(y_te, y_prob)

        ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc_val:.3f})')
        txt += f"  {name:25s}: {auc_val:.3f}  (n={len(te)})\n"

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax_roc.set_xlabel('FPR', fontsize=12)
    ax_roc.set_ylabel('TPR', fontsize=12)
    ax_roc.set_title('Figure 6: Predictive Validation', fontsize=14, fontweight='bold')
    ax_roc.legend(fontsize=9, loc='lower right')

    ax_txt.text(0.02, 0.98, txt, transform=ax_txt.transAxes, fontsize=11,
                va='top', fontfamily='monospace')
    ax_txt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig6_predictive_validation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig6_predictive_validation.png")


# ============================================================================
# 9. CONFOUND ANALYSIS
# ============================================================================
def confound_analysis(dm):
    """β vs age/sex + ANCOVA."""
    clean = dm[dm.is_clean & dm.beta_mean.notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # β vs age
    ax = axes[0]
    for sc in SUPERCLASSES:
        s = clean[clean.clean_superclass == sc]
        ax.scatter(s.age, s.beta_mean, alpha=0.1, s=5, color=SC_COLORS[sc], label=sc)
    ax.axhline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Age'); ax.set_ylabel('β_mean')
    ax.set_title('β vs Age', fontweight='bold')
    ax.legend(markerscale=5)

    # β distributions by sex per diagnosis
    ax = axes[1]
    positions = []
    violins_data = []
    tick_positions = []
    tick_labels = []
    for j, sc in enumerate(SUPERCLASSES):
        for sex_val in [0, 1]:
            s = clean[(clean.clean_superclass == sc) & (clean.sex == sex_val)].beta_mean.dropna()
            if len(s) > 5:
                pos = j * 2.5 + sex_val * 0.8
                violins_data.append(s.values)
                positions.append(pos)
        tick_positions.append(j * 2.5 + 0.4)
        tick_labels.append(sc)
    if violins_data:
        ax.violinplot(violins_data, positions=positions, showmeans=True, widths=0.6)
    ax.axhline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('β_mean')
    ax.set_title('β by Diagnosis × Sex', fontweight='bold')

    # ANCOVA
    ax = axes[2]
    cv = clean.dropna(subset=['beta_mean', 'age', 'sex', 'clean_superclass'])
    if len(cv) > 100:
        res = pg.ancova(data=cv, dv='beta_mean',
                        between='clean_superclass', covar=['age', 'sex'])
        ax.text(0.02, 0.98, f"ANCOVA: β ~ diagnosis + age + sex\n\n{res.to_string()}",
                transform=ax.transAxes, fontsize=8.5, va='top', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Confound Control', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confound_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved confound_analysis.png")


# ============================================================================
# 10. FIGURE 5 — REGIONAL β
# ============================================================================
def fig5_regional_beta(dm):
    """Bar chart: regional β by diagnosis."""
    clean = dm[dm.is_clean & dm.beta_mean.notna()].copy()
    regions = {
        'Anterior': ['V1', 'V2', 'V3', 'V4'],
        'Lateral':  ['I', 'aVL', 'V5', 'V6'],
        'Inferior': ['II', 'III', 'aVF'],
        'Right':    ['aVR'],
    }

    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(4*len(SUPERCLASSES), 5),
                             sharey=True)
    for i, sc in enumerate(SUPERCLASSES):
        ax = axes[i]
        sub = clean[clean.clean_superclass == sc]
        reg_vals = {}
        for rname, leads in regions.items():
            cols = [f'beta_ir_{l}' for l in leads]
            reg_vals[rname] = sub[cols].median().mean()
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(0.5, 2.0)
        bars = ax.barh(list(reg_vals.keys()), list(reg_vals.values()),
                       color=[cmap(norm(v)) for v in reg_vals.values()])
        ax.axvline(1.0, color='red', ls='--', alpha=0.7)
        ax.set_xlim(0.5, 2.0)
        ax.set_title(f'{sc}\n(n={len(sub)})', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Region')
        ax.set_xlabel('Median β')

    plt.suptitle('Figure 5: Regional β by Diagnosis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(RESULTS_DIR, 'fig5_regional_beta.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_regional_beta.png")


# ============================================================================
# 11. METHOD COMPARISON
# ============================================================================
def method_comparison(dm):
    """IRASA vs specparam vs above-knee."""
    v = dm.dropna(subset=['beta_mean', 'beta_sp_mean']).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IRASA vs specparam
    ax = axes[0]
    ax.scatter(v.beta_mean, v.beta_sp_mean, alpha=0.08, s=5)
    lims = [min(v.beta_mean.min(), v.beta_sp_mean.min()),
            max(v.beta_mean.max(), v.beta_sp_mean.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7)
    rho, p = stats.spearmanr(v.beta_mean, v.beta_sp_mean)
    ax.set_xlabel('β (IRASA)')
    ax.set_ylabel('β (specparam)')
    ax.set_title(f'IRASA vs specparam (ρ={rho:.3f})', fontweight='bold')

    # IRASA vs above-knee
    v2 = dm.dropna(subset=['beta_mean', 'beta_ak_mean'])
    ax = axes[1]
    ax.scatter(v2.beta_mean, v2.beta_ak_mean, alpha=0.08, s=5)
    rho2, p2 = stats.spearmanr(v2.beta_mean, v2.beta_ak_mean)
    lims2 = [min(v2.beta_mean.min(), v2.beta_ak_mean.min()),
             max(v2.beta_mean.max(), v2.beta_ak_mean.max())]
    ax.plot(lims2, lims2, 'r--', alpha=0.7)
    ax.set_xlabel('β (IRASA)')
    ax.set_ylabel('β (above-knee)')
    ax.set_title(f'IRASA vs above-knee (ρ={rho2:.3f})', fontweight='bold')

    plt.suptitle('Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(RESULTS_DIR, 'method_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved method_comparison.png")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("The Heartbeat at the Edge of Chaos")
    print("Criticality Analysis of PTB-XL")
    print("=" * 70)

    print("\n[1] Loading metadata...")
    df, scp = load_metadata()

    print("\n[2] Computing β features...")
    beta_df = process_all(df)

    print(f"\n[3] Merging: {len(beta_df)} records with β")
    dm = df.join(beta_df, how='inner')

    # Quality info
    if 'r2_mean' in dm.columns:
        good = dm[dm.r2_mean >= 0.7]
        print(f"    R² ≥ 0.7: {len(good)} ({100*len(good)/len(dm):.1f}%)")

    print("\n[4] Part 1: β landscape...")
    part1_beta_landscape(dm)

    print("\n[5] Part 2: Subclass heatmap...")
    part2_subclass_heatmap(dm, scp)

    print("\n[6] Part 3: Cross-lead coherence...")
    part3_cross_lead_coherence(dm)

    print("\n[7] Part 4: Distance from criticality...")
    part4_distance_criticality(dm)

    print("\n[8] Part 5: Prediction...")
    part5_prediction(dm)

    print("\n[9] Confounds...")
    confound_analysis(dm)

    print("\n[10] Regional β (Fig 5)...")
    fig5_regional_beta(dm)

    print("\n[11] Method comparison...")
    method_comparison(dm)

    print("\n" + "=" * 70)
    print("DONE. Figures in:", RESULTS_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
