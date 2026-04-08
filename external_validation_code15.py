"""
External Validation on CODE-15% (345,779 ECGs, Brazil)
======================================================
Three analyses:
1. LBBB vs RBBB classification (third population)
2. β vs all-cause mortality (Cox regression + Kaplan-Meier)
3. β biological age vs DNN-predicted age

Key differences from PTB-XL/Chapman:
- 400 Hz (not 500 Hz) → IRASA parameters adapted
- HDF5 format (not WFDB)
- 4096 samples per record (zero-padded, ~10.24s)
- Mortality follow-up: death + timey columns
- DNN-predicted age: nn_predicted_age column
"""
import os, sys, warnings
import numpy as np, pandas as pd, h5py
from scipy import signal, stats
from neurodsp.aperiodic import compute_irasa
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR    = 'code15'
RESULTS_DIR = 'results'
CACHE_CSV   = f'{RESULTS_DIR}/code15_beta_features.csv'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
FS_CODE = 400  # CODE-15% sampling rate
IRASA_HSET = np.arange(1.1, 1.95, 0.05)
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
N_JOBS = 6

# ── Signal Processing (adapted for 400 Hz) ────────────────────────────
def preprocess(s, fs=FS_CODE):
    """Bandpass 0.5-100 Hz + 50 Hz notch."""
    nyq = fs / 2
    high = min(100, nyq - 1)  # ensure < Nyquist
    sos = signal.butter(4, [0.5, high], btype='bandpass', fs=fs, output='sos')
    out = signal.sosfiltfilt(sos, s)
    # 60 Hz notch for Brazil (not 50 Hz like Europe)
    b, a = signal.iirnotch(60, Q=30, fs=fs)
    return signal.filtfilt(b, a, out)


def beta_irasa(s, fs=FS_CODE):
    """Compute spectral exponent via IRASA (adapted for 400 Hz)."""
    try:
        f, pa, pp = compute_irasa(s, fs=fs, f_range=(0.5, 50), hset=IRASA_HSET)
        m = (f >= 2) & (f <= 45) & (pa > 0)
        if m.sum() < 5:
            return np.nan, np.nan
        sl, ic, rv, _, _ = stats.linregress(np.log10(f[m]), np.log10(pa[m]))
        return -sl, rv**2
    except:
        return np.nan, np.nan


def process_batch_from_hdf5(hdf5_path, meta_df, max_records=None):
    """Process all records from an HDF5 partition."""
    print(f"  Loading {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        exam_ids = np.array(f['exam_id'])
        tracings = f['tracings']  # shape: (N, 4096, 12)
        n_total = tracings.shape[0]

        if max_records and max_records < n_total:
            indices = np.random.choice(n_total, max_records, replace=False)
            indices = sorted(indices)
        else:
            indices = range(n_total)

        print(f"  Processing {len(indices)} of {n_total} records...")

        # Process in chunks to avoid memory issues
        chunk_size = 500
        all_results = []
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_idx = list(indices[chunk_start:chunk_start + chunk_size])
            # Load chunk into memory
            chunk_data = np.array(tracings[chunk_idx, :, :])
            chunk_ids = exam_ids[chunk_idx]

            # Process each record in chunk
            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_one_signal)(
                    chunk_ids[i], chunk_data[i], meta_df
                ) for i in range(len(chunk_idx))
            )
            results = [r for r in results if r is not None]
            all_results.extend(results)

            if (chunk_start + chunk_size) % 2000 == 0 or chunk_start + chunk_size >= len(indices):
                print(f"    {min(chunk_start + chunk_size, len(indices))}/{len(indices)} done "
                      f"({len(all_results)} valid)")

    return all_results


def process_one_signal(exam_id, ecg_data, meta_df):
    """Process a single record: ecg_data shape (4096, 12)."""
    try:
        # Raw values are already in mV-scale (range ~±3)
        # β is scale-invariant (log-log slope), so scaling doesn't matter
        ecg = ecg_data.astype(np.float64)

        # Remove zero-padding: find actual signal extent
        # (signals are symmetrically zero-padded to 4096)
        signal_energy = np.sum(ecg**2, axis=1)
        nonzero = np.where(signal_energy > 1e-6)[0]
        if len(nonzero) < FS_CODE * 3:  # need at least 3 seconds
            return None
        start, end = nonzero[0], nonzero[-1] + 1
        ecg = ecg[start:end, :]

        if ecg.shape[0] < FS_CODE * 3:
            return None

        result = {'exam_id': int(exam_id)}

        betas_ir = []
        r2s = []
        for i, lead in enumerate(LEAD_NAMES):
            s = preprocess(ecg[:, i], fs=FS_CODE)
            b_ir, r2 = beta_irasa(s, fs=FS_CODE)
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

        return result
    except:
        return None


def add_derived_features(df):
    """Add derived features (same as PTB-XL/Chapman)."""
    anterior = ['V1','V2','V3','V4']
    lateral  = ['I','aVL','V5','V6']
    inferior = ['II','III','aVF']

    df['beta_anterior'] = df[[f'beta_ir_{l}' for l in anterior]].mean(axis=1)
    df['beta_lateral']  = df[[f'beta_ir_{l}' for l in lateral]].mean(axis=1)
    df['beta_inferior'] = df[[f'beta_ir_{l}' for l in inferior]].mean(axis=1)

    regions = df[['beta_anterior','beta_lateral','beta_inferior']]
    df['beta_regional_div'] = regions.max(axis=1) - regions.min(axis=1)

    lead_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]
    lead_arr = df[lead_cols].values
    df['beta_iqr']   = np.nanpercentile(lead_arr, 75, axis=1) - np.nanpercentile(lead_arr, 25, axis=1)
    df['beta_cv']    = df['beta_std'] / df['beta_mean'].abs()
    df['beta_range'] = np.nanmax(lead_arr, axis=1) - np.nanmin(lead_arr, axis=1)
    df['r2_std']     = df[[f'r2_ir_{l}' for l in LEAD_NAMES]].std(axis=1)
    df['sex_num']    = df['is_male'].astype(float)
    df['age_x_beta'] = df['age'] * df['beta_mean']

    from scipy.stats import skew as spskew
    df['beta_skew'] = df[lead_cols].apply(lambda row: spskew(row.dropna()), axis=1)

    return df


def get_feature_cols():
    """Feature set for classification."""
    base = ['age', 'sex_num', 'beta_mean', 'beta_std', 'beta_median', 'delta',
            'beta_anterior', 'beta_lateral', 'beta_inferior', 'beta_regional_div',
            'r2_mean', 'r2_std', 'beta_iqr', 'beta_cv', 'beta_skew', 'beta_range',
            'age_x_beta']
    leads_beta = [f'beta_ir_{l}' for l in LEAD_NAMES]
    leads_r2   = [f'r2_ir_{l}' for l in LEAD_NAMES]
    return base + leads_beta + leads_r2


# ── Analysis 1: LBBB vs RBBB ─────────────────────────────────────────
def analysis_lbbb_rbbb(df):
    """LBBB vs RBBB classification — third population."""
    lbbb = df[df['LBBB'] == True].copy()
    rbbb = df[df['RBBB'] == True].copy()

    if len(lbbb) < 20 or len(rbbb) < 20:
        print("Not enough LBBB/RBBB records for classification")
        return None

    lbbb['label'] = 1
    rbbb['label'] = 0
    data = pd.concat([lbbb, rbbb]).dropna(subset=['beta_mean'])

    print(f"\n{'='*60}")
    print(f"Analysis 1: LBBB vs RBBB Classification")
    print(f"{'='*60}")
    print(f"LBBB: {len(lbbb)}, RBBB: {len(rbbb)}")

    feat_cols = [c for c in get_feature_cols() if c in data.columns]
    X = data[feat_cols].fillna(0).values
    y = data['label'].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    results = {}
    for name, model in [
        ('GBM', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.1, subsample=0.8, random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=300, max_depth=10,
                                       min_samples_leaf=3, class_weight='balanced', random_state=42)),
        ('LR', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ]:
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
        print(f"  {name}: AUC = {auc:.4f}")
        if name == 'GBM':
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            results['fpr'] = fpr
            results['tpr'] = tpr
            results['importance'] = pd.DataFrame({
                'feature': feat_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        results[f'auc_{name.lower()}'] = auc

    # 5-fold CV
    cv_scores = cross_val_score(
        GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, random_state=42),
        scaler.fit_transform(X), y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    print(f"  GBM 5-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()
    results['n_lbbb'] = len(lbbb)
    results['n_rbbb'] = len(rbbb)

    return results


# ── Analysis 2: Mortality ─────────────────────────────────────────────
def analysis_mortality(df):
    """β vs all-cause mortality: Cox regression + Kaplan-Meier."""
    # Only first exams with mortality data
    mort = df.dropna(subset=['death', 'timey']).copy()
    mort = mort[mort['timey'] > 0]

    if len(mort) < 100:
        print("Not enough mortality data")
        return None

    print(f"\n{'='*60}")
    print(f"Analysis 2: β vs All-Cause Mortality")
    print(f"{'='*60}")
    print(f"N with mortality data: {len(mort)}")
    print(f"Deaths: {mort['death'].sum()} ({mort['death'].mean()*100:.1f}%)")
    print(f"Follow-up: median {mort['timey'].median():.1f}y")

    # β quartiles
    mort['beta_quartile'] = pd.qcut(mort['beta_mean'], 4, labels=['Q1','Q2','Q3','Q4'])
    for q in ['Q1','Q2','Q3','Q4']:
        qdf = mort[mort['beta_quartile'] == q]
        print(f"  {q}: N={len(qdf):5d}, β={qdf['beta_mean'].mean():.3f}, "
              f"deaths={qdf['death'].sum():4d} ({qdf['death'].mean()*100:.1f}%)")

    # Cox Proportional Hazards
    print(f"\n  Cox PH regression:")

    # Model 1: β_mean only
    cox_data = mort[['timey', 'death', 'beta_mean']].dropna().copy()
    cox_data['death'] = cox_data['death'].astype(int)
    cph1 = CoxPHFitter()
    cph1.fit(cox_data, duration_col='timey', event_col='death')
    hr1 = np.exp(cph1.params_['beta_mean'])
    ci1 = np.exp(cph1.confidence_intervals_.values[0])
    p1 = cph1.summary['p'].values[0]
    print(f"    Model 1 (β only):     HR = {hr1:.3f} [{ci1[0]:.3f}-{ci1[1]:.3f}], p = {p1:.2e}")

    # Model 2: β_mean + age + sex
    cox_cols = ['timey', 'death', 'beta_mean', 'age', 'sex_num']
    cox_data2 = mort[cox_cols].dropna().copy()
    cox_data2['death'] = cox_data2['death'].astype(int)
    cph2 = CoxPHFitter()
    cph2.fit(cox_data2, duration_col='timey', event_col='death')
    hr2 = np.exp(cph2.params_['beta_mean'])
    ci2_raw = cph2.confidence_intervals_.loc['beta_mean'].values
    ci2 = np.exp(ci2_raw)
    p2 = cph2.summary.loc['beta_mean', 'p']
    print(f"    Model 2 (β+age+sex):  HR = {hr2:.3f} [{ci2[0]:.3f}-{ci2[1]:.3f}], p = {p2:.2e}")

    # Model 3: β_mean + age + sex + diagnostic labels
    cox_cols3 = ['timey', 'death', 'beta_mean', 'age', 'sex_num',
                 'LBBB', 'RBBB', '1dAVb', 'AF']
    cox_data3 = mort[cox_cols3].dropna().copy()
    cox_data3['death'] = cox_data3['death'].astype(int)
    for col in ['LBBB','RBBB','1dAVb','AF']:
        cox_data3[col] = cox_data3[col].astype(int)
    cph3 = CoxPHFitter()
    cph3.fit(cox_data3, duration_col='timey', event_col='death')
    hr3 = np.exp(cph3.params_['beta_mean'])
    ci3_raw = cph3.confidence_intervals_.loc['beta_mean'].values
    ci3 = np.exp(ci3_raw)
    p3 = cph3.summary.loc['beta_mean', 'p']
    print(f"    Model 3 (β+age+sex+dx): HR = {hr3:.3f} [{ci3[0]:.3f}-{ci3[1]:.3f}], p = {p3:.2e}")

    # Kaplan-Meier by β quartiles
    kmf = KaplanMeierFitter()
    km_data = {}
    for q in ['Q1','Q2','Q3','Q4']:
        qdf = mort[mort['beta_quartile'] == q]
        kmf.fit(qdf['timey'], event_observed=qdf['death'].astype(int), label=q)
        km_data[q] = {
            'timeline': kmf.survival_function_.index.values,
            'survival': kmf.survival_function_.values.flatten()
        }

    # Log-rank test: Q1 vs Q4
    q1 = mort[mort['beta_quartile'] == 'Q1']
    q4 = mort[mort['beta_quartile'] == 'Q4']
    lr_result = logrank_test(q1['timey'], q4['timey'],
                              event_observed_A=q1['death'].astype(int),
                              event_observed_B=q4['death'].astype(int))
    print(f"\n  Log-rank Q1 vs Q4: χ² = {lr_result.test_statistic:.1f}, p = {lr_result.p_value:.2e}")

    return {
        'mort_df': mort,
        'km_data': km_data,
        'hr_unadj': hr1, 'ci_unadj': ci1, 'p_unadj': p1,
        'hr_adj': hr2, 'ci_adj': ci2, 'p_adj': p2,
        'hr_full': hr3, 'ci_full': ci3, 'p_full': p3,
        'logrank_p': lr_result.p_value,
        'cph_summary2': cph2.summary,
        'cph_summary3': cph3.summary,
    }


# ── Analysis 3: β Age vs DNN Age ─────────────────────────────────────
def analysis_bio_age(df):
    """Compare β-derived biological age proxy with DNN-predicted age."""
    age_data = df.dropna(subset=['age', 'nn_predicted_age', 'beta_mean']).copy()

    if len(age_data) < 100:
        print("Not enough data for age analysis")
        return None

    print(f"\n{'='*60}")
    print(f"Analysis 3: β Biological Age vs DNN-Predicted Age")
    print(f"{'='*60}")
    print(f"N: {len(age_data)}")

    # DNN age gap
    age_data['dnn_age_gap'] = age_data['nn_predicted_age'] - age_data['age']
    print(f"DNN age gap: {age_data['dnn_age_gap'].mean():.1f} ± {age_data['dnn_age_gap'].std():.1f} years")

    # β vs chronological age
    rho_beta_age, p_beta = stats.spearmanr(age_data['age'], age_data['beta_mean'])
    print(f"β vs chrono age: ρ = {rho_beta_age:.4f}, p = {p_beta:.2e}")

    # DNN age vs chrono age
    rho_dnn_age, p_dnn = stats.spearmanr(age_data['age'], age_data['nn_predicted_age'])
    print(f"DNN age vs chrono age: ρ = {rho_dnn_age:.4f}")

    # β vs DNN age
    rho_beta_dnn, p_bd = stats.spearmanr(age_data['beta_mean'], age_data['nn_predicted_age'])
    print(f"β vs DNN age: ρ = {rho_beta_dnn:.4f}, p = {p_bd:.2e}")

    # β vs DNN age gap (the interesting one: does β capture "accelerated aging"?)
    rho_beta_gap, p_bg = stats.spearmanr(age_data['beta_mean'], age_data['dnn_age_gap'])
    print(f"β vs DNN age gap: ρ = {rho_beta_gap:.4f}, p = {p_bg:.2e}")

    # If mortality data available, compare β and DNN age as mortality predictors
    mort_data = age_data.dropna(subset=['death', 'timey'])
    mort_data = mort_data[mort_data['timey'] > 0].copy()

    if len(mort_data) > 100:
        print(f"\n  Comparing β vs DNN age as mortality predictors (N={len(mort_data)}):")
        mort_data['death_int'] = mort_data['death'].astype(int)

        # C-index for β
        from lifelines.utils import concordance_index
        c_beta = concordance_index(mort_data['timey'], -mort_data['beta_mean'], mort_data['death_int'])
        c_dnn = concordance_index(mort_data['timey'], -mort_data['nn_predicted_age'], mort_data['death_int'])
        c_age = concordance_index(mort_data['timey'], -mort_data['age'], mort_data['death_int'])
        c_gap = concordance_index(mort_data['timey'], -mort_data['dnn_age_gap'], mort_data['death_int'])

        print(f"    C-index (chrono age):    {c_age:.4f}")
        print(f"    C-index (DNN age):       {c_dnn:.4f}")
        print(f"    C-index (DNN age gap):   {c_gap:.4f}")
        print(f"    C-index (β):             {c_beta:.4f}")

        return {
            'age_data': age_data,
            'rho_beta_age': rho_beta_age,
            'rho_dnn_age': rho_dnn_age,
            'rho_beta_dnn': rho_beta_dnn,
            'rho_beta_gap': rho_beta_gap,
            'c_age': c_age, 'c_dnn': c_dnn, 'c_gap': c_gap, 'c_beta': c_beta,
        }
    return {
        'age_data': age_data,
        'rho_beta_age': rho_beta_age,
        'rho_dnn_age': rho_dnn_age,
        'rho_beta_dnn': rho_beta_dnn,
        'rho_beta_gap': rho_beta_gap,
    }


# ── Visualization ─────────────────────────────────────────────────────
def plot_code15_results(df, lbbb_res, mort_res, age_res):
    """Generate comprehensive CODE-15% validation figure."""
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.30)
    palette = {'NORM':'#2ecc71','LBBB':'#e74c3c','RBBB':'#3498db','AF':'#9b59b6'}

    # ── Row 1: LBBB vs RBBB ──────────────────────────────────────────
    # A: β distributions
    ax1 = fig.add_subplot(gs[0, 0])
    groups_list = []
    labels_list = []
    for name, col in [('NORM','normal_ecg'), ('LBBB','LBBB'), ('RBBB','RBBB'), ('AF','AF')]:
        grp = df[df[col] == True]['beta_mean'].dropna()
        if len(grp) > 10:
            groups_list.append(grp.values)
            labels_list.append(name)
    parts = ax1.violinplot(groups_list, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette.get(labels_list[i],'#999')); pc.set_alpha(0.7)
    ax1.set_xticks(range(1, len(labels_list)+1))
    ax1.set_xticklabels([f'{l}\n(N={len(g)})' for l,g in zip(labels_list, groups_list)], fontsize=9)
    ax1.set_ylabel('β', fontsize=11)
    ax1.set_title('A  β by Diagnosis (CODE-15%, Brazil)', fontsize=12, fontweight='bold')

    # B: LBBB vs RBBB ROC
    ax2 = fig.add_subplot(gs[0, 1])
    if lbbb_res:
        ax2.plot(lbbb_res['fpr'], lbbb_res['tpr'], 'b-', lw=2.5,
                 label=f"GBM AUC = {lbbb_res['auc_gbm']:.3f}")
        ax2.fill_between(lbbb_res['fpr'], lbbb_res['tpr'], alpha=0.15, color='blue')
        ax2.plot([0,1],[0,1],'k--',alpha=0.3)
        ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
        ax2.set_title('B  LBBB vs RBBB (3rd Population)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11, loc='lower right')
        ax2.text(0.05, 0.88, f"LBBB={lbbb_res['n_lbbb']}  RBBB={lbbb_res['n_rbbb']}",
                 transform=ax2.transAxes, fontsize=10)
        ax2.text(0.05, 0.80, f"CV: {lbbb_res['cv_mean']:.3f}±{lbbb_res['cv_std']:.3f}",
                 transform=ax2.transAxes, fontsize=10, color='#555')

    # C: Three-continent AUC comparison
    ax3 = fig.add_subplot(gs[0, 2])
    datasets = ['PTB-XL\n(Germany)', 'Chapman\n(China)', 'CODE-15%\n(Brazil)']
    aucs = [0.982, 0.982, lbbb_res['auc_gbm'] if lbbb_res else 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax3.bar(datasets, aucs, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, aucs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.85, 1.02)
    ax3.set_ylabel('AUC (LBBB vs RBBB)', fontsize=11)
    ax3.set_title('C  Cross-Continental Replication', fontsize=12, fontweight='bold')
    ax3.axhline(y=0.95, color='gray', ls=':', alpha=0.3)

    # ── Row 2: Mortality ──────────────────────────────────────────────
    if mort_res:
        # D: Kaplan-Meier by β quartiles
        ax4 = fig.add_subplot(gs[1, 0:2])
        km_colors = {'Q1':'#2ecc71', 'Q2':'#f1c40f', 'Q3':'#e67e22', 'Q4':'#e74c3c'}
        mort = mort_res['mort_df']
        for q in ['Q1','Q2','Q3','Q4']:
            kd = mort_res['km_data'][q]
            q_beta = mort[mort['beta_quartile']==q]['beta_mean'].mean()
            ax4.step(kd['timeline'], kd['survival'], where='post',
                     color=km_colors[q], lw=2.5,
                     label=f"{q} (β={q_beta:.2f})")
        ax4.set_xlabel('Time (years)', fontsize=11)
        ax4.set_ylabel('Survival Probability', fontsize=11)
        ax4.set_title(f'D  Kaplan-Meier by β Quartiles  (log-rank p = {mort_res["logrank_p"]:.2e})',
                       fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.set_xlim(0, mort['timey'].quantile(0.95))
        ax4.grid(True, alpha=0.2)

        # E: Cox HR forest plot
        ax5 = fig.add_subplot(gs[1, 2])
        models = ['β only', 'β + age + sex', 'β + age + sex + dx']
        hrs = [mort_res['hr_unadj'], mort_res['hr_adj'], mort_res['hr_full']]
        cis_lo = [mort_res['ci_unadj'][0], mort_res['ci_adj'][0], mort_res['ci_full'][0]]
        cis_hi = [mort_res['ci_unadj'][1], mort_res['ci_adj'][1], mort_res['ci_full'][1]]

        y_pos = range(len(models))
        ax5.errorbar(hrs, y_pos,
                     xerr=[[h-l for h,l in zip(hrs, cis_lo)],
                           [h-l for h,l in zip(cis_hi, hrs)]],
                     fmt='ko', capsize=5, markersize=8)
        ax5.axvline(x=1.0, color='red', ls='--', alpha=0.5)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(models, fontsize=10)
        ax5.set_xlabel('Hazard Ratio (per unit β)', fontsize=11)
        ax5.set_title('E  Cox Regression: β → Mortality', fontsize=12, fontweight='bold')
        for i, (hr, lo, hi) in enumerate(zip(hrs, cis_lo, cis_hi)):
            ax5.text(max(hi, hr) + 0.05, i,
                     f'HR={hr:.2f}\n[{lo:.2f}-{hi:.2f}]', fontsize=9, va='center')

    # ── Row 3: Bio Age ────────────────────────────────────────────────
    if age_res:
        ad = age_res['age_data']

        # F: β vs chrono age
        ax6 = fig.add_subplot(gs[2, 0])
        # Subsample for plotting
        plot_idx = np.random.choice(len(ad), min(5000, len(ad)), replace=False)
        ax6.scatter(ad.iloc[plot_idx]['age'], ad.iloc[plot_idx]['beta_mean'],
                    s=1, alpha=0.1, color='#3498db')
        bins = np.arange(20, 95, 5)
        for i in range(len(bins)-1):
            mask = (ad['age'] >= bins[i]) & (ad['age'] < bins[i+1])
            if mask.sum() > 10:
                ax6.errorbar(bins[i]+2.5, ad.loc[mask, 'beta_mean'].mean(),
                             yerr=ad.loc[mask, 'beta_mean'].sem(),
                             fmt='ko', markersize=5, capsize=3)
        ax6.set_xlabel('Chronological Age', fontsize=11)
        ax6.set_ylabel('β', fontsize=11)
        ax6.set_title(f'F  β vs Age (ρ = {age_res["rho_beta_age"]:.3f})', fontsize=12, fontweight='bold')

        # G: β vs DNN age gap
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(ad.iloc[plot_idx]['dnn_age_gap'], ad.iloc[plot_idx]['beta_mean'],
                    s=1, alpha=0.1, color='#e74c3c')
        ax7.set_xlabel('DNN Age Gap (DNN age − chrono age)', fontsize=11)
        ax7.set_ylabel('β', fontsize=11)
        ax7.set_title(f'G  β vs DNN Age Gap (ρ = {age_res["rho_beta_gap"]:.3f})',
                       fontsize=12, fontweight='bold')
        ax7.axvline(x=0, color='gray', ls='--', alpha=0.3)

        # H: C-index comparison
        if 'c_beta' in age_res:
            ax8 = fig.add_subplot(gs[2, 2])
            predictors = ['Chrono\nAge', 'DNN\nAge', 'DNN Age\nGap', 'β']
            c_vals = [age_res['c_age'], age_res['c_dnn'], age_res['c_gap'], age_res['c_beta']]
            colors_c = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71']
            bars = ax8.bar(predictors, c_vals, color=colors_c, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, c_vals):
                ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                         f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
            ax8.set_ylim(0.45, 0.75)
            ax8.set_ylabel('C-index (mortality)', fontsize=11)
            ax8.set_title('H  Mortality Prediction: C-index', fontsize=12, fontweight='bold')
            ax8.axhline(y=0.5, color='gray', ls=':', alpha=0.3)

    plt.suptitle('CODE-15% Validation (N = {:,}, Brazil): Mortality + Three-Continent Replication'.format(len(df)),
                 fontsize=15, fontweight='bold', y=0.99)

    plt.savefig(f'{RESULTS_DIR}/fig_external_validation_code15.png', dpi=200, bbox_inches='tight')
    print(f"\n→ Saved: {RESULTS_DIR}/fig_external_validation_code15.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load metadata
    meta = pd.read_csv(f'{DATA_DIR}/exams.csv')
    print(f"Metadata: {len(meta)} records")

    # Step 1: Check for cached features
    if os.path.exists(CACHE_CSV):
        print(f"Loading cached features from {CACHE_CSV}")
        df = pd.read_csv(CACHE_CSV)
        print(f"Loaded {len(df)} records")
    else:
        # Step 2: Process HDF5 partitions
        all_results = []
        for part in range(3):  # partitions 0, 1, 2
            hdf5_path = f'{DATA_DIR}/exams_part{part}.hdf5'
            if os.path.exists(hdf5_path):
                results = process_batch_from_hdf5(hdf5_path, meta)
                all_results.extend(results)
                print(f"  Partition {part}: {len(results)} valid records")
            else:
                print(f"  Partition {part}: NOT FOUND")

        if len(all_results) == 0:
            print("No results! Check HDF5 files.")
            sys.exit(1)

        df = pd.DataFrame(all_results)
        print(f"\nTotal processed: {len(df)} records")

        # Save cache
        df.to_csv(CACHE_CSV, index=False)
        print(f"Saved to {CACHE_CSV}")

    # Step 3: Merge with metadata
    df = df.merge(meta, on='exam_id', how='left')
    print(f"After merge: {len(df)} records with metadata")

    # Step 4: Add derived features
    df = add_derived_features(df)

    # Step 5: Run analyses
    print(f"\n{'#'*60}")
    print(f"# CODE-15% VALIDATION RESULTS")
    print(f"# Dataset: CODE-15% (Brazil)")
    print(f"# N = {len(df):,}")
    print(f"{'#'*60}")

    # β distributions
    norm_df = df[df['normal_ecg'] == True]
    lbbb_df = df[df['LBBB'] == True]
    rbbb_df = df[df['RBBB'] == True]
    print(f"\nβ distributions:")
    print(f"  NORM: β = {norm_df['beta_mean'].mean():.4f} ± {norm_df['beta_mean'].std():.4f} (N={len(norm_df)})")
    print(f"  LBBB: β = {lbbb_df['beta_mean'].mean():.4f} ± {lbbb_df['beta_mean'].std():.4f} (N={len(lbbb_df)})")
    print(f"  RBBB: β = {rbbb_df['beta_mean'].mean():.4f} ± {rbbb_df['beta_mean'].std():.4f} (N={len(rbbb_df)})")

    # Analysis 1
    lbbb_res = analysis_lbbb_rbbb(df)

    # Analysis 2
    mort_res = analysis_mortality(df)

    # Analysis 3
    age_res = analysis_bio_age(df)

    # Step 6: Generate figure
    plot_code15_results(df, lbbb_res, mort_res, age_res)

    # Final summary
    print(f"\n{'='*60}")
    print(f"CODE-15% VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"N processed: {len(df):,}")
    if lbbb_res:
        print(f"LBBB vs RBBB AUC: {lbbb_res['auc_gbm']:.4f} (GBM)")
        print(f"                   CV: {lbbb_res['cv_mean']:.4f} ± {lbbb_res['cv_std']:.4f}")
    if mort_res:
        print(f"Mortality HR (β, adjusted): {mort_res['hr_adj']:.3f} "
              f"[{mort_res['ci_adj'][0]:.3f}-{mort_res['ci_adj'][1]:.3f}]")
    if age_res:
        print(f"β vs chrono age: ρ = {age_res['rho_beta_age']:.4f}")
        print(f"β vs DNN age gap: ρ = {age_res['rho_beta_gap']:.4f}")
        if 'c_beta' in age_res:
            print(f"C-index (β): {age_res['c_beta']:.4f}")
            print(f"C-index (DNN age): {age_res['c_dnn']:.4f}")
    print(f"\nDone!")
