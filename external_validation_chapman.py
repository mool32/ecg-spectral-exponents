"""
External Validation on Chapman-Shaoxing (45,152 ECGs, China)
============================================================
Replicates the PTB-XL β-analysis pipeline on an independent dataset:
1. IRASA → β per 12 leads
2. β distributions by diagnostic group (NORM / LBBB / RBBB / etc.)
3. LBBB vs RBBB classification → AUC
4. Spatial fingerprints comparison

SNOMED-CT codes for Chapman-Shaoxing:
  - Sinus rhythm (NORM): 426783006
  - Normal ECG: 284470004
  - LBBB: 164909002
  - CLBBB: 733534002       (treated as equivalent to LBBB)
  - RBBB: 59118001
  - CRBBB: 713427006       (treated as equivalent to RBBB)
  - LAFB: 445118002
  - 1AVB: 270492004
  - IRBBB: 713426002
  - IVCD: 698252002
"""
import os, sys, glob, re, warnings
import numpy as np, pandas as pd, wfdb
from scipy import signal, stats
from neurodsp.aperiodic import compute_irasa
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = 'chapman-shaoxing'
RESULTS_DIR = 'results'
CACHE_CSV  = f'{RESULTS_DIR}/chapman_beta_features.csv'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
FS = 500
IRASA_HSET = np.arange(1.1, 1.95, 0.05)
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# SNOMED-CT code mapping
SNOMED = {
    'NORM':  {'426783006', '284470004'},          # Sinus rhythm, Normal ECG
    'LBBB':  {'164909002', '733534002'},          # LBBB, CLBBB
    'RBBB':  {'59118001', '713427006'},           # RBBB, CRBBB
    'LAFB':  {'445118002'},                        # Left anterior fascicular block
    '1AVB':  {'270492004'},                        # First degree AV block
    'IRBBB': {'713426002'},                        # Incomplete RBBB
    'AF':    {'164889003'},                        # Atrial fibrillation
    'MI':    {'164865005', '54329005'},            # Myocardial infarction
    'STTC':  {'164930006', '164931005', '164934002', '59931005'},  # ST/T changes
    'HYP':   {'164873001', '89792004'},            # LVH, RVH
}


# ── Signal Processing ─────────────────────────────────────────────────
def preprocess(s, fs=FS):
    """Bandpass 0.5-100 Hz + 50 Hz notch (same as PTB-XL pipeline)."""
    sos = signal.butter(4, [0.5, 100], btype='bandpass', fs=fs, output='sos')
    out = signal.sosfiltfilt(sos, s)
    b, a = signal.iirnotch(50, Q=30, fs=fs)
    return signal.filtfilt(b, a, out)


def beta_irasa(s, fs=FS):
    """Compute spectral exponent via IRASA (same parameters as PTB-XL)."""
    try:
        f, pa, pp = compute_irasa(s, fs=fs, f_range=(0.5, 50), hset=IRASA_HSET)
        m = (f >= 2) & (f <= 45) & (pa > 0)
        if m.sum() < 5:
            return np.nan, np.nan
        sl, ic, rv, _, _ = stats.linregress(np.log10(f[m]), np.log10(pa[m]))
        return -sl, rv**2
    except:
        return np.nan, np.nan


# ── Record Discovery ──────────────────────────────────────────────────
def discover_records(data_dir):
    """Find all .hea files and parse record paths + SNOMED labels."""
    hea_files = sorted(glob.glob(os.path.join(data_dir, 'WFDBRecords', '**', '*.hea'),
                                 recursive=True))
    print(f"Found {len(hea_files)} header files")

    records = []
    for hea_path in hea_files:
        rec_id = os.path.splitext(os.path.basename(hea_path))[0]
        rec_path = os.path.splitext(hea_path)[0]  # path without extension

        # Parse header for SNOMED codes and demographics
        dx_codes = set()
        age = np.nan
        sex = np.nan

        with open(hea_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#Dx:'):
                    codes = line.replace('#Dx:', '').strip().split(',')
                    dx_codes = set(c.strip() for c in codes)
                elif line.startswith('#Age:'):
                    try:
                        age = float(line.replace('#Age:', '').strip())
                    except:
                        pass
                elif line.startswith('#Sex:'):
                    sex_str = line.replace('#Sex:', '').strip().lower()
                    if sex_str in ('male', 'm'):
                        sex = 0.0
                    elif sex_str in ('female', 'f'):
                        sex = 1.0

        records.append({
            'rec_id': rec_id,
            'rec_path': rec_path,
            'dx_codes': dx_codes,
            'age': age,
            'sex': sex
        })

    return records


def assign_labels(records):
    """Map SNOMED codes to diagnostic labels."""
    for rec in records:
        codes = rec['dx_codes']
        # Assign superclass labels
        rec['is_NORM'] = bool(codes & SNOMED['NORM'])
        rec['is_LBBB'] = bool(codes & SNOMED['LBBB'])
        rec['is_RBBB'] = bool(codes & SNOMED['RBBB'])
        rec['is_LAFB'] = bool(codes & SNOMED['LAFB'])
        rec['is_1AVB'] = bool(codes & SNOMED['1AVB'])
        rec['is_IRBBB'] = bool(codes & SNOMED['IRBBB'])
        rec['is_AF']   = bool(codes & SNOMED['AF'])
        rec['is_MI']   = bool(codes & SNOMED['MI'])
        rec['is_STTC'] = bool(codes & SNOMED['STTC'])
        rec['is_HYP']  = bool(codes & SNOMED['HYP'])

        # Assign CD subtype (priority: LBBB > RBBB > LAFB > 1AVB > IRBBB)
        if rec['is_LBBB']:
            rec['cd_subtype'] = 'LBBB'
        elif rec['is_RBBB']:
            rec['cd_subtype'] = 'RBBB'
        elif rec['is_LAFB']:
            rec['cd_subtype'] = 'LAFB'
        elif rec['is_1AVB']:
            rec['cd_subtype'] = '1AVB'
        elif rec['is_IRBBB']:
            rec['cd_subtype'] = 'IRBBB'
        else:
            rec['cd_subtype'] = None

    return records


# ── Feature Extraction ────────────────────────────────────────────────
def process_one(rec):
    """Process a single record: read WFDB → preprocess → IRASA → features."""
    try:
        rec_path = rec['rec_path']
        record = wfdb.rdrecord(rec_path)
        ecg = record.p_signal
        fs = record.fs

        if ecg is None or ecg.shape[0] < fs * 5:  # at least 5 seconds
            return None

        # Ensure 12 leads
        n_leads = min(ecg.shape[1], 12)

        result = {
            'rec_id': rec['rec_id'],
            'age': rec['age'],
            'sex': rec['sex'],
        }

        betas_ir = []
        r2s = []
        for i in range(n_leads):
            lead = LEAD_NAMES[i]
            s = preprocess(ecg[:, i], fs=fs)
            b_ir, r2 = beta_irasa(s, fs=fs)
            result[f'beta_ir_{lead}'] = b_ir
            result[f'r2_ir_{lead}'] = r2
            betas_ir.append(b_ir)
            r2s.append(r2)

        arr = np.array(betas_ir)
        valid = (~np.isnan(arr)).sum()
        if valid < 6:
            return None

        result['beta_mean'] = np.nanmean(arr)
        result['beta_std'] = np.nanstd(arr)
        result['beta_median'] = np.nanmedian(arr)
        result['delta'] = abs(np.nanmean(arr) - 1.0)
        result['r2_mean'] = np.nanmean(r2s)
        result['n_valid'] = int(valid)

        # Copy labels
        for key in ['is_NORM','is_LBBB','is_RBBB','is_LAFB','is_1AVB','is_IRBBB',
                     'is_AF','is_MI','is_STTC','is_HYP','cd_subtype']:
            result[key] = rec[key]

        return result
    except Exception as e:
        return None


# ── Derived Features ──────────────────────────────────────────────────
def add_derived_features(df):
    """Add same derived features as PTB-XL analysis."""
    anterior = ['V1','V2','V3','V4']
    lateral  = ['I','aVL','V5','V6']
    inferior = ['II','III','aVF']

    df['beta_anterior'] = df[[f'beta_ir_{l}' for l in anterior]].mean(axis=1)
    df['beta_lateral']  = df[[f'beta_ir_{l}' for l in lateral]].mean(axis=1)
    df['beta_inferior'] = df[[f'beta_ir_{l}' for l in inferior]].mean(axis=1)

    regions = df[['beta_anterior','beta_lateral','beta_inferior']]
    df['beta_regional_div'] = regions.max(axis=1) - regions.min(axis=1)

    # Per-row IQR, CV, skew, range from the 12 lead betas
    lead_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]
    lead_arr = df[lead_cols].values
    df['beta_iqr']   = np.nanpercentile(lead_arr, 75, axis=1) - np.nanpercentile(lead_arr, 25, axis=1)
    df['beta_cv']    = df['beta_std'] / df['beta_mean'].abs()
    df['beta_range'] = np.nanmax(lead_arr, axis=1) - np.nanmin(lead_arr, axis=1)
    df['r2_std']     = df[[f'r2_ir_{l}' for l in LEAD_NAMES]].std(axis=1)
    df['sex_num']    = df['sex'].fillna(0.0)
    df['age_x_beta'] = df['age'] * df['beta_mean']

    # Skewness per row
    from scipy.stats import skew as spskew
    df['beta_skew'] = df[lead_cols].apply(lambda row: spskew(row.dropna()), axis=1)

    return df


def get_feature_cols():
    """Same 54-feature set as PTB-XL analysis (minus beta_sp_*)."""
    base = ['age', 'sex_num', 'beta_mean', 'beta_std', 'beta_median', 'delta',
            'beta_anterior', 'beta_lateral', 'beta_inferior', 'beta_regional_div',
            'r2_mean', 'r2_std', 'beta_iqr', 'beta_cv', 'beta_skew', 'beta_range',
            'age_x_beta']
    leads_beta = [f'beta_ir_{l}' for l in LEAD_NAMES]
    leads_r2   = [f'r2_ir_{l}' for l in LEAD_NAMES]
    return base + leads_beta + leads_r2  # 17 + 12 + 12 = 41 features


# ── Analysis Functions ────────────────────────────────────────────────
def lbbb_vs_rbbb_classification(df):
    """Binary LBBB vs RBBB classification (headline replication)."""
    lbbb = df[df['is_LBBB'] == True].copy()
    rbbb = df[df['is_RBBB'] == True].copy()

    lbbb['label'] = 1
    rbbb['label'] = 0
    data = pd.concat([lbbb, rbbb]).dropna(subset=['beta_mean'])

    print(f"\n{'='*60}")
    print(f"LBBB vs RBBB Classification")
    print(f"{'='*60}")
    print(f"LBBB: {len(lbbb)}, RBBB: {len(rbbb)}")

    feat_cols = get_feature_cols()
    # Only keep available columns
    feat_cols = [c for c in feat_cols if c in data.columns]

    X = data[feat_cols].fillna(0).values
    y = data['label'].values

    # Random 80/20 split (no strat_fold in Chapman-Shaoxing)
    np.random.seed(42)
    n = len(data)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # GBM (same hyperparameters as PTB-XL)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.1, subsample=0.8,
                                      random_state=42)
    gbm.fit(X_tr, y_tr)
    y_prob = gbm.predict_proba(X_te)[:, 1]
    y_pred = gbm.predict(X_te)

    auc = roc_auc_score(y_te, y_prob)
    f1 = f1_score(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    # Also try Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)[:, 1]
    lr_auc = roc_auc_score(y_te, lr_prob)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                 min_samples_leaf=3, class_weight='balanced',
                                 random_state=42)
    rf.fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    rf_auc = roc_auc_score(y_te, rf_prob)

    print(f"GBM  AUC: {auc:.4f}  F1: {f1:.4f}")
    print(f"LR   AUC: {lr_auc:.4f}")
    print(f"RF   AUC: {rf_auc:.4f}")
    best_auc = max(auc, lr_auc, rf_auc)
    print(f"Best AUC: {best_auc:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': gbm.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'auc_gbm': auc, 'auc_lr': lr_auc, 'auc_rf': rf_auc,
        'f1': f1, 'fpr': fpr, 'tpr': tpr,
        'n_lbbb': len(lbbb), 'n_rbbb': len(rbbb),
        'importance': importance,
        'y_te': y_te, 'y_prob': y_prob
    }


def beta_distributions(df):
    """Compare β distributions: NORM vs LBBB vs RBBB (replication of Result 1)."""
    print(f"\n{'='*60}")
    print(f"β Distributions by Diagnostic Group")
    print(f"{'='*60}")

    groups = {
        'NORM': df[df['is_NORM'] == True],
        'LBBB': df[df['is_LBBB'] == True],
        'RBBB': df[df['is_RBBB'] == True],
        'LAFB': df[df['is_LAFB'] == True],
        '1AVB': df[df['is_1AVB'] == True],
        'AF':   df[df['is_AF'] == True],
    }

    # Only include groups with enough samples
    groups = {k: v for k, v in groups.items() if len(v) >= 20}

    for name, grp in groups.items():
        bm = grp['beta_mean'].dropna()
        print(f"  {name:6s}: N={len(grp):5d}  β_mean = {bm.mean():.4f} ± {bm.std():.4f}")

    # Cohen's d: LBBB vs NORM
    if 'LBBB' in groups and 'NORM' in groups:
        b_norm = groups['NORM']['beta_mean'].dropna()
        b_lbbb = groups['LBBB']['beta_mean'].dropna()
        n1, n2 = len(b_norm), len(b_lbbb)
        pooled_std = np.sqrt(((n1-1)*b_norm.std()**2 + (n2-1)*b_lbbb.std()**2) / (n1+n2-2))
        d = (b_lbbb.mean() - b_norm.mean()) / pooled_std
        print(f"\n  Cohen's d (LBBB vs NORM): {d:.3f}")

    if 'RBBB' in groups and 'NORM' in groups:
        b_norm = groups['NORM']['beta_mean'].dropna()
        b_rbbb = groups['RBBB']['beta_mean'].dropna()
        n1, n2 = len(b_norm), len(b_rbbb)
        pooled_std = np.sqrt(((n1-1)*b_norm.std()**2 + (n2-1)*b_rbbb.std()**2) / (n1+n2-2))
        d = (b_rbbb.mean() - b_norm.mean()) / pooled_std
        print(f"  Cohen's d (RBBB vs NORM): {d:.3f}")

    return groups


def spatial_fingerprints(df):
    """Compare spatial fingerprints: LBBB vs RBBB vs NORM across 12 leads."""
    print(f"\n{'='*60}")
    print(f"Spatial Fingerprints (β per lead)")
    print(f"{'='*60}")

    groups = {}
    for name, mask_col in [('NORM', 'is_NORM'), ('LBBB', 'is_LBBB'), ('RBBB', 'is_RBBB')]:
        grp = df[df[mask_col] == True]
        if len(grp) < 20:
            continue
        profile = []
        for lead in LEAD_NAMES:
            col = f'beta_ir_{lead}'
            vals = grp[col].dropna()
            profile.append(vals.mean())
        groups[name] = np.array(profile)
        print(f"  {name}: " + "  ".join(f"{lead}={v:.3f}" for lead, v in zip(LEAD_NAMES, profile)))

    return groups


def aging_trajectory(df):
    """β vs age correlation (replication of Result 3)."""
    norm = df[(df['is_NORM'] == True)].dropna(subset=['age', 'beta_mean'])

    if len(norm) < 100:
        print("Not enough NORM records with age for aging analysis")
        return None

    rho, pval = stats.spearmanr(norm['age'], norm['beta_mean'])
    print(f"\n{'='*60}")
    print(f"Aging Trajectory (NORM only)")
    print(f"{'='*60}")
    print(f"  N = {len(norm)}")
    print(f"  β vs age: ρ = {rho:.4f}, p = {pval:.2e}")
    print(f"  β_mean overall: {norm['beta_mean'].mean():.4f} ± {norm['beta_mean'].std():.4f}")

    return {'rho': rho, 'pval': pval, 'norm_data': norm}


# ── Visualization ─────────────────────────────────────────────────────
def plot_validation_results(df, lbbb_rbbb_res, groups, fingerprints, aging_res):
    """Generate comprehensive validation figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    palette = {'NORM': '#2ecc71', 'LBBB': '#e74c3c', 'RBBB': '#3498db',
               'LAFB': '#f39c12', '1AVB': '#9b59b6', 'IRBBB': '#1abc9c', 'AF': '#95a5a6'}

    # ── Panel A: β distributions violin ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    grp_names = [g for g in ['NORM','LBBB','RBBB','LAFB','1AVB','AF'] if g in groups]
    data_viol = [groups[g]['beta_mean'].dropna().values for g in grp_names]

    parts = ax1.violinplot(data_viol, showmeans=True, showmedians=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette.get(grp_names[i], '#999'))
        pc.set_alpha(0.7)
    ax1.set_xticks(range(1, len(grp_names)+1))
    ax1.set_xticklabels(grp_names, fontsize=10)
    ax1.set_ylabel('β (spectral exponent)', fontsize=11)
    ax1.set_title('A  β by Diagnostic Group\n(Chapman-Shaoxing, China)', fontsize=12, fontweight='bold')
    ax1.axhline(y=groups['NORM']['beta_mean'].mean() if 'NORM' in groups else 1.76,
                color='gray', ls='--', alpha=0.5)

    # ── Panel B: LBBB vs RBBB ROC ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lbbb_rbbb_res['fpr'], lbbb_rbbb_res['tpr'], 'b-', lw=2,
             label=f"GBM AUC = {lbbb_rbbb_res['auc_gbm']:.3f}")
    ax2.plot([0,1], [0,1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('B  LBBB vs RBBB Classification\n(External Validation)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.text(0.05, 0.85, f"N = {lbbb_rbbb_res['n_lbbb']} + {lbbb_rbbb_res['n_rbbb']}",
             transform=ax2.transAxes, fontsize=10)

    # ── Panel C: Feature importance ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    imp = lbbb_rbbb_res['importance'].head(15)
    colors = ['#e74c3c' if 'beta' in f else '#3498db' if 'r2' in f else '#95a5a6'
              for f in imp['feature']]
    ax3.barh(range(len(imp)), imp['importance'].values, color=colors)
    ax3.set_yticks(range(len(imp)))
    ax3.set_yticklabels(imp['feature'].values, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('C  Top Features (LBBB vs RBBB)', fontsize=12, fontweight='bold')

    # ── Panel D: Spatial fingerprints ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    x = np.arange(12)
    for name, profile in fingerprints.items():
        ax4.plot(x, profile, 'o-', color=palette.get(name, '#999'),
                 label=name, lw=2, markersize=6)
    ax4.set_xticks(x)
    ax4.set_xticklabels(LEAD_NAMES, fontsize=10)
    ax4.set_ylabel('β (spectral exponent)', fontsize=11)
    ax4.set_title('D  Spatial Fingerprints by Lead (Chapman-Shaoxing)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ── Panel E: PTB-XL vs Chapman comparison (radar) ─────────────────
    ax5 = fig.add_subplot(gs[1, 2], projection='polar')
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False).tolist()
    angles += angles[:1]

    for name, profile in fingerprints.items():
        vals = list(profile) + [profile[0]]
        ax5.plot(angles, vals, '-', color=palette.get(name, '#999'),
                 label=name, lw=2)
        ax5.fill(angles, vals, color=palette.get(name, '#999'), alpha=0.1)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(LEAD_NAMES, fontsize=8)
    ax5.set_title('E  Radar: Spatial Fingerprints', fontsize=12, fontweight='bold', pad=20)

    # ── Panel F: β vs age (if NORM + age available) ───────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    if aging_res is not None:
        norm = aging_res['norm_data']
        ax6.scatter(norm['age'], norm['beta_mean'], s=2, alpha=0.15, color='#2ecc71')
        # Binned means
        bins = np.arange(20, 90, 5)
        for i in range(len(bins)-1):
            mask = (norm['age'] >= bins[i]) & (norm['age'] < bins[i+1])
            if mask.sum() > 10:
                ax6.errorbar(bins[i]+2.5, norm.loc[mask, 'beta_mean'].mean(),
                             yerr=norm.loc[mask, 'beta_mean'].sem(),
                             fmt='ko', markersize=5, capsize=3)
        ax6.set_xlabel('Age (years)')
        ax6.set_ylabel('β (spectral exponent)')
        ax6.set_title(f'F  β vs Age (NORM, ρ = {aging_res["rho"]:.3f})',
                       fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax6.transAxes)
        ax6.set_title('F  β vs Age', fontsize=12, fontweight='bold')

    # ── Panel G: Cross-dataset β_NORM comparison ──────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    # PTB-XL reference value
    ptbxl_beta_norm = 1.76  # from previous analysis
    chapman_beta_norm = groups['NORM']['beta_mean'].mean() if 'NORM' in groups else np.nan

    datasets = ['PTB-XL\n(Germany)', 'Chapman\n(China)']
    betas = [ptbxl_beta_norm, chapman_beta_norm]
    colors_bar = ['#3498db', '#e74c3c']
    bars = ax7.bar(datasets, betas, color=colors_bar, alpha=0.8, edgecolor='black')
    ax7.set_ylabel('β_NORM (mean)')
    ax7.set_title('G  β_NORM Cross-Population', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, betas):
        ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax7.set_ylim(0, max(betas)*1.15)

    # ── Panel H: Summary stats table ──────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    summary_data = [
        ['Metric', 'PTB-XL', 'Chapman'],
        ['Population', 'Germany', 'China'],
        ['N (total)', '21,799', f'{len(df):,}'],
        ['β_NORM', '1.760', f'{chapman_beta_norm:.3f}'],
        ['LBBB vs RBBB AUC', '0.982', f'{lbbb_rbbb_res["auc_gbm"]:.3f}'],
        ['LBBB N', '536', f'{lbbb_rbbb_res["n_lbbb"]}'],
        ['RBBB N', '537', f'{lbbb_rbbb_res["n_rbbb"]}'],
    ]

    if aging_res is not None:
        summary_data.append(['β vs age ρ', '-0.181', f'{aging_res["rho"]:.3f}'])

    table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0],
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Color header
    for j in range(3):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax8.set_title('H  Cross-Dataset Summary', fontsize=12, fontweight='bold')

    plt.suptitle('External Validation: Chapman-Shaoxing (N = {:,}, China)'.format(len(df)),
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(f'{RESULTS_DIR}/fig_external_validation_chapman.png', dpi=200, bbox_inches='tight')
    print(f"\n→ Saved: {RESULTS_DIR}/fig_external_validation_chapman.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    N_JOBS = 6
    # Set: process only relevant groups (LBBB, RBBB, NORM, LAFB, 1AVB, IRBBB, AF, HYP)
    # and a NORM subsample for speed. Set MAX_NORM=0 to include all NORM.
    MAX_NORM = 3000  # subsample NORM for speed; 0 = all

    # Step 1: Check for cached features
    if os.path.exists(CACHE_CSV):
        print(f"Loading cached features from {CACHE_CSV}")
        df = pd.read_csv(CACHE_CSV)
        print(f"Loaded {len(df)} records")
    else:
        # Step 2: Discover records
        print("Discovering records...")
        records = discover_records(DATA_DIR)
        print(f"Total records: {len(records)}")

        # Assign labels
        records = assign_labels(records)

        # Count labels
        n_lbbb = sum(1 for r in records if r['is_LBBB'])
        n_rbbb = sum(1 for r in records if r['is_RBBB'])
        n_norm = sum(1 for r in records if r['is_NORM'])
        n_lafb = sum(1 for r in records if r['is_LAFB'])
        n_1avb = sum(1 for r in records if r['is_1AVB'])
        n_irbbb = sum(1 for r in records if r['is_IRBBB'])
        print(f"NORM: {n_norm}, LBBB: {n_lbbb}, RBBB: {n_rbbb}, "
              f"LAFB: {n_lafb}, 1AVB: {n_1avb}, IRBBB: {n_irbbb}")

        # Filter: keep only records in relevant diagnostic groups
        relevant = [r for r in records if (
            r['is_LBBB'] or r['is_RBBB'] or r['is_NORM'] or
            r['is_LAFB'] or r['is_1AVB'] or r['is_IRBBB'] or
            r['is_AF'] or r['is_HYP']
        )]
        print(f"Relevant records: {len(relevant)}")

        # Subsample NORM if needed
        if MAX_NORM > 0:
            norm_recs = [r for r in relevant if r['is_NORM'] and not (
                r['is_LBBB'] or r['is_RBBB'] or r['is_LAFB'] or
                r['is_1AVB'] or r['is_IRBBB'])]
            non_norm = [r for r in relevant if not (r['is_NORM'] and not (
                r['is_LBBB'] or r['is_RBBB'] or r['is_LAFB'] or
                r['is_1AVB'] or r['is_IRBBB']))]
            np.random.seed(42)
            if len(norm_recs) > MAX_NORM:
                norm_recs = list(np.random.choice(norm_recs, MAX_NORM, replace=False))
            relevant = non_norm + norm_recs
            print(f"After NORM subsampling: {len(relevant)} records")

        # Step 3: Process all records with IRASA
        print(f"\nProcessing {len(relevant)} records with IRASA ({N_JOBS} workers)...")
        results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(process_one)(rec) for rec in relevant
        )
        results = [r for r in results if r is not None]
        print(f"Successfully processed: {len(results)}")

        df = pd.DataFrame(results)
        df.to_csv(CACHE_CSV, index=False)
        print(f"Saved to {CACHE_CSV}")

    # Step 4: Add derived features
    df = add_derived_features(df)

    # Step 5: Run analyses
    print(f"\n{'#'*60}")
    print(f"# EXTERNAL VALIDATION RESULTS")
    print(f"# Dataset: Chapman-Shaoxing (China)")
    print(f"# N = {len(df):,}")
    print(f"{'#'*60}")

    # 5a: β distributions
    groups = beta_distributions(df)

    # 5b: LBBB vs RBBB classification
    lbbb_rbbb_res = lbbb_vs_rbbb_classification(df)

    # 5c: Spatial fingerprints
    fingerprints = spatial_fingerprints(df)

    # 5d: Aging trajectory
    aging_res = aging_trajectory(df)

    # Step 6: Generate figure
    plot_validation_results(df, lbbb_rbbb_res, groups, fingerprints, aging_res)

    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset:          Chapman-Shaoxing (China)")
    print(f"N processed:      {len(df):,}")
    print(f"LBBB vs RBBB AUC: {lbbb_rbbb_res['auc_gbm']:.4f} (GBM)")
    print(f"                   {lbbb_rbbb_res['auc_lr']:.4f} (LR)")
    print(f"                   {lbbb_rbbb_res['auc_rf']:.4f} (RF)")
    if 'NORM' in groups:
        print(f"β_NORM (Chapman): {groups['NORM']['beta_mean'].mean():.4f}")
        print(f"β_NORM (PTB-XL):  1.7600 (reference)")
    if aging_res:
        print(f"β vs age ρ:       {aging_res['rho']:.4f}")
    print(f"\nDone!")
