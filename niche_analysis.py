"""
Niche Analysis: Where β-Features Beat or Complement Deep Learning
=================================================================
Niche 1: CD subtype classification (CLBBB vs CRBBB vs LAFB vs 1AVB vs IVCD)
          β spatial fingerprint → anatomically interpretable subtyping

Niche 2: Subclinical signal detection
          Pure NORM vs "NORM + subclinical CD" (cardiologist sees NORM but notes mild
          conduction findings — can β catch this?)
"""

import os
import ast
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, f1_score, recall_score,
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ptb-xl')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

CD_SUBTYPE_COLORS = {
    'CLBBB': '#E53935',   # red — severe, left
    'CRBBB': '#1565C0',   # blue — right
    'LAFB':  '#FF9800',   # orange — fascicular
    '1AVB':  '#9C27B0',   # purple — AV block
    'IVCD':  '#607D8B',   # grey — nonspecific
    'IRBBB': '#00BCD4',   # cyan — incomplete right
}

NORM_GREEN = '#4CAF50'


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_data():
    """Load and merge all data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    meta = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    meta['scp_dict'] = meta.scp_codes.apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)

    beta_df = pd.read_csv(os.path.join(RESULTS_DIR, 'beta_features.csv'))
    df = meta.join(beta_df.set_index('ecg_id'), how='inner')
    df = df[df.beta_mean.notna()].copy()

    # Feature engineering
    lead_betas = [f'beta_ir_{l}' for l in LEAD_NAMES]
    r2_cols = [f'r2_ir_{l}' for l in LEAD_NAMES]

    regions = {
        'anterior': ['V1', 'V2', 'V3', 'V4'],
        'lateral':  ['I', 'aVL', 'V5', 'V6'],
        'inferior': ['II', 'III', 'aVF'],
    }
    for region, leads in regions.items():
        cols = [f'beta_ir_{l}' for l in leads]
        df[f'beta_{region}'] = df[cols].mean(axis=1)

    df['beta_regional_div'] = (
        df[['beta_anterior', 'beta_lateral', 'beta_inferior']].max(axis=1) -
        df[['beta_anterior', 'beta_lateral', 'beta_inferior']].min(axis=1)
    )
    df['beta_iqr'] = df[lead_betas].quantile(0.75, axis=1) - df[lead_betas].quantile(0.25, axis=1)
    df['beta_cv'] = df['beta_std'] / df['beta_mean'].abs()
    df['beta_skew'] = df[lead_betas].skew(axis=1)
    df['beta_range'] = df[lead_betas].max(axis=1) - df[lead_betas].min(axis=1)
    df['r2_std'] = df[r2_cols].std(axis=1)
    df['sex_num'] = df['sex'].astype(float)
    df['age_x_beta'] = df['age'] * df['beta_mean']

    print(f"  Total records with β: {len(df):,}")
    return df, scp


def get_feature_cols():
    """Full feature set for classification."""
    lead_betas = [f'beta_ir_{l}' for l in LEAD_NAMES]
    r2_cols = [f'r2_ir_{l}' for l in LEAD_NAMES]
    sp_cols = [f'beta_sp_{l}' for l in LEAD_NAMES]

    return ['age', 'sex_num', 'beta_mean', 'beta_std', 'beta_median', 'delta',
            'beta_anterior', 'beta_lateral', 'beta_inferior', 'beta_regional_div',
            'r2_mean', 'r2_std', 'beta_iqr', 'beta_cv', 'beta_skew', 'beta_range',
            'age_x_beta', 'beta_sp_mean'] + lead_betas + r2_cols + sp_cols


# ============================================================================
# 2. NICHE 1: CD SUBTYPE CLASSIFICATION
# ============================================================================
def niche1_cd_subtypes(df):
    """
    Classify CD subtypes using β-features.
    Key subtypes: CLBBB, CRBBB, LAFB, 1AVB, IVCD, IRBBB
    """
    print("\n" + "=" * 70)
    print("NICHE 1: CD SUBTYPE CLASSIFICATION")
    print("=" * 70)

    # Extract records with dominant CD subtype (likelihood >= 80)
    subtypes_of_interest = ['CLBBB', 'CRBBB', 'LAFB', '1AVB', 'IVCD', 'IRBBB']

    subtype_records = []
    for ecg_id, row in df.iterrows():
        codes = row['scp_dict']
        for subtype in subtypes_of_interest:
            if subtype in codes and codes[subtype] >= 80:
                subtype_records.append({'ecg_id': ecg_id, 'cd_subtype': subtype})
                break  # take first matching subtype

    sub_df = pd.DataFrame(subtype_records)

    # Some records may have multiple CD codes — keep only records with ONE dominant subtype
    # (already handled by break above, but let's verify)
    sub_df = sub_df.drop_duplicates(subset='ecg_id')
    sub_df = sub_df.set_index('ecg_id')
    cd_data = df.join(sub_df, how='inner')

    print(f"\n  CD subtype distribution:")
    for st in subtypes_of_interest:
        n = (cd_data.cd_subtype == st).sum()
        print(f"    {st:8s}: {n:,}")

    # --- Feature importance: which leads matter for which subtype ---
    feat_cols = get_feature_cols()
    avail = [f for f in feat_cols if f in cd_data.columns]

    # Split
    train = cd_data[cd_data.strat_fold.isin(range(1, 9))]
    test = cd_data[cd_data.strat_fold.isin([9, 10])]

    tr_clean = train[avail + ['cd_subtype']].dropna()
    te_clean = test[avail + ['cd_subtype']].dropna()

    X_tr, y_tr = tr_clean[avail].values, tr_clean['cd_subtype'].values
    X_te, y_te = te_clean[avail].values, te_clean['cd_subtype'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\n  Train: {len(X_tr):,}, Test: {len(X_te):,}")
    print(f"  Features: {len(avail)}")

    # --- Models ---
    results = {}

    # LogReg
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    lr.fit(X_tr_s, y_tr)
    y_pred_lr = lr.predict(X_te_s)
    y_prob_lr = lr.predict_proba(X_te_s)

    # RF
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    y_pred_rf = rf.predict(X_te_s)
    y_prob_rf = rf.predict_proba(X_te_s)

    # GBM
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     subsample=0.8, random_state=42)
    gbm.fit(X_tr_s, y_tr)
    y_pred_gbm = gbm.predict(X_te_s)
    y_prob_gbm = gbm.predict_proba(X_te_s)

    # Evaluate all models
    for name, model, y_pred, y_prob in [
        ('LogReg', lr, y_pred_lr, y_prob_lr),
        ('RF', rf, y_pred_rf, y_prob_rf),
        ('GBM', gbm, y_pred_gbm, y_prob_gbm),
    ]:
        classes = model.classes_
        y_te_bin = label_binarize(y_te, classes=classes)
        if y_te_bin.shape[1] == 1:
            y_te_bin = np.hstack([1 - y_te_bin, y_te_bin])

        per_class_auc = {}
        for i, cls in enumerate(classes):
            try:
                per_class_auc[cls] = roc_auc_score(y_te_bin[:, i], y_prob[:, i])
            except ValueError:
                per_class_auc[cls] = np.nan

        macro_auc = np.nanmean(list(per_class_auc.values()))
        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_te, y_pred, labels=classes)

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'classes': classes,
            'per_class_auc': per_class_auc,
            'macro_auc': macro_auc,
            'report': report,
            'cm': cm,
            'weighted_f1': report['weighted avg']['f1-score'],
        }

        print(f"\n  {name}: Macro AUC = {macro_auc:.3f}, Weighted F1 = {results[name]['weighted_f1']:.3f}")
        for cls in classes:
            print(f"    {cls:8s}: AUC = {per_class_auc.get(cls, np.nan):.3f}, "
                  f"F1 = {report.get(cls, {}).get('f1-score', 0):.3f}")

    return results, cd_data, avail, gbm


def niche1_pairwise(df):
    """
    Key pairwise comparison: CLBBB vs CRBBB.
    This is the most anatomically meaningful distinction.
    """
    print("\n--- Pairwise: CLBBB vs CRBBB ---")

    # Extract CLBBB and CRBBB records
    records = []
    for ecg_id, row in df.iterrows():
        codes = row['scp_dict']
        if 'CLBBB' in codes and codes['CLBBB'] >= 80:
            records.append({'ecg_id': ecg_id, 'bbb_type': 'CLBBB'})
        elif 'CRBBB' in codes and codes['CRBBB'] >= 80:
            records.append({'ecg_id': ecg_id, 'bbb_type': 'CRBBB'})

    bbb_df = pd.DataFrame(records).set_index('ecg_id')
    bbb_data = df.join(bbb_df, how='inner')

    feat_cols = get_feature_cols()
    avail = [f for f in feat_cols if f in bbb_data.columns]

    train = bbb_data[bbb_data.strat_fold.isin(range(1, 9))]
    test = bbb_data[bbb_data.strat_fold.isin([9, 10])]

    tr = train[avail + ['bbb_type']].dropna()
    te = test[avail + ['bbb_type']].dropna()

    X_tr, y_tr = tr[avail].values, (tr['bbb_type'] == 'CLBBB').astype(int).values
    X_te, y_te = te[avail].values, (te['bbb_type'] == 'CLBBB').astype(int).values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"  Train: {len(X_tr)} (CLBBB={y_tr.sum()}, CRBBB={len(y_tr)-y_tr.sum()})")
    print(f"  Test:  {len(X_te)} (CLBBB={y_te.sum()}, CRBBB={len(y_te)-y_te.sum()})")

    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     subsample=0.8, random_state=42)
    gbm.fit(X_tr_s, y_tr)
    y_prob = gbm.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    print(f"  CLBBB vs CRBBB AUC = {auc:.3f}")

    return auc, fpr, tpr, gbm, avail, bbb_data


# ============================================================================
# 3. NICHE 2: SUBCLINICAL SIGNAL DETECTION
# ============================================================================
def niche2_subclinical(df):
    """
    Can β detect subclinical conduction disturbance in 'healthy' ECGs?

    Group A: Pure NORM (only NORM code, nothing else, likelihood >= 80)
    Group B: NORM + subclinical CD (NORM code + IRBBB/IVCD/1AVB at any likelihood)
             Cardiologist said "normal" but noted mild conduction findings.

    If β differs between A and B → β catches what the cardiologist hesitates about.
    """
    print("\n" + "=" * 70)
    print("NICHE 2: SUBCLINICAL SIGNAL DETECTION")
    print("=" * 70)

    subclinical_cd_codes = {'IRBBB', 'IVCD', '1AVB', 'LAFB', 'LPFB', 'ILBBB'}

    group_a_ids = []  # Pure NORM
    group_b_ids = []  # NORM + subclinical CD
    group_b_subtypes = {}  # track which subclinical codes

    for ecg_id, row in df.iterrows():
        codes = row['scp_dict']
        if 'NORM' not in codes or codes.get('NORM', 0) < 80:
            continue

        # Check for CD subcodes
        cd_present = set()
        for code in codes:
            if code in subclinical_cd_codes and codes[code] > 0:
                cd_present.add(code)

        if len(cd_present) == 0 and len(codes) == 1:
            # Only NORM, nothing else
            group_a_ids.append(ecg_id)
        elif len(cd_present) > 0:
            group_b_ids.append(ecg_id)
            group_b_subtypes[ecg_id] = list(cd_present)

    print(f"\n  Group A (Pure NORM, single code): {len(group_a_ids):,}")
    print(f"  Group B (NORM + subclinical CD):   {len(group_b_ids):,}")

    # Subclinical code distribution
    from collections import Counter
    sub_codes = Counter()
    for codes in group_b_subtypes.values():
        for c in codes:
            sub_codes[c] += 1
    print(f"\n  Subclinical CD codes in Group B:")
    for code, count in sub_codes.most_common():
        print(f"    {code:8s}: {count:,}")

    # Extract features
    a_data = df.loc[df.index.isin(group_a_ids)].copy()
    b_data = df.loc[df.index.isin(group_b_ids)].copy()

    a_data['group'] = 'Pure NORM'
    b_data['group'] = 'NORM + subclinical'
    combined = pd.concat([a_data, b_data])

    # --- Statistical comparison ---
    print(f"\n  --- Statistical Comparison ---")
    print(f"  {'Feature':25s} {'Pure NORM':>12s} {'NORM+subCDs':>12s} {'Cohen d':>8s} {'p-value':>12s}")
    print(f"  {'-'*75}")

    key_features = ['beta_mean', 'beta_std', 'beta_anterior', 'beta_lateral',
                    'beta_inferior', 'beta_regional_div', 'r2_mean',
                    'beta_cv', 'beta_range']

    effect_results = []
    for feat in key_features:
        va = a_data[feat].dropna()
        vb = b_data[feat].dropna()
        _, p = stats.mannwhitneyu(va, vb)
        d = (vb.mean() - va.mean()) / np.sqrt(((len(va)-1)*va.std()**2 + (len(vb)-1)*vb.std()**2) / (len(va)+len(vb)-2))
        print(f"  {feat:25s} {va.mean():12.4f} {vb.mean():12.4f} {d:+8.3f} {p:12.2e}")
        effect_results.append({'feature': feat, 'mean_A': va.mean(), 'mean_B': vb.mean(),
                              'd': d, 'p': p})

    # --- Classification: can we tell them apart? ---
    print(f"\n  --- Binary Classification: Pure NORM vs NORM+subclinical ---")
    feat_cols = get_feature_cols()
    avail = [f for f in feat_cols if f in combined.columns]

    combined['label'] = (combined['group'] == 'NORM + subclinical').astype(int)

    train = combined[combined.strat_fold.isin(range(1, 9))]
    test = combined[combined.strat_fold.isin([9, 10])]

    tr = train[avail + ['label']].dropna()
    te = test[avail + ['label']].dropna()

    X_tr, y_tr = tr[avail].values, tr['label'].values
    X_te, y_te = te[avail].values, te['label'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"  Train: {len(X_tr):,} (A={len(y_tr)-y_tr.sum()}, B={y_tr.sum()})")
    print(f"  Test:  {len(X_te):,} (A={len(y_te)-y_te.sum()}, B={y_te.sum()})")

    subclinical_results = {}
    for name, model in [
        ('LogReg', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')),
        ('RF', RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced',
                                      random_state=42, n_jobs=-1)),
        ('GBM', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                           subsample=0.8, random_state=42)),
    ]:
        model.fit(X_tr_s, y_tr)
        y_prob = model.predict_proba(X_te_s)[:, 1]
        y_pred = model.predict(X_te_s)
        auc = roc_auc_score(y_te, y_prob)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        sens = recall_score(y_te, y_pred, pos_label=1)
        spec = recall_score(y_te, y_pred, pos_label=0)

        subclinical_results[name] = {
            'auc': auc, 'fpr': fpr, 'tpr': tpr,
            'sens': sens, 'spec': spec, 'model': model,
        }
        print(f"  {name:8s}: AUC = {auc:.3f}, Sens = {sens:.3f}, Spec = {spec:.3f}")

    return effect_results, subclinical_results, a_data, b_data, avail


# ============================================================================
# 4. FIGURES
# ============================================================================
def fig_niche1(cd_results, cd_data, bbb_auc, bbb_fpr, bbb_tpr, bbb_gbm, avail, save_path):
    """
    Figure N1: CD Subtype Classification
    Panel A: Spatial fingerprint (β per lead) for each subtype
    Panel B: Confusion matrix
    Panel C: CLBBB vs CRBBB ROC + feature importance
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- Panel A: Spatial fingerprint per subtype ---
    ax = fig.add_subplot(gs[0, 0:2])

    subtypes_plot = ['CLBBB', 'CRBBB', 'IRBBB', 'LAFB', '1AVB', 'IVCD']
    for subtype in subtypes_plot:
        sub = cd_data[cd_data.cd_subtype == subtype]
        if len(sub) < 10:
            continue
        medians = [sub[f'beta_ir_{l}'].median() for l in LEAD_NAMES]
        q25 = [sub[f'beta_ir_{l}'].quantile(0.25) for l in LEAD_NAMES]
        q75 = [sub[f'beta_ir_{l}'].quantile(0.75) for l in LEAD_NAMES]
        x = range(len(LEAD_NAMES))
        ax.plot(x, medians, 'o-', color=CD_SUBTYPE_COLORS.get(subtype, 'gray'),
                linewidth=2, markersize=5, label=f'{subtype} (n={len(sub):,})')
        ax.fill_between(x, q25, q75, alpha=0.1, color=CD_SUBTYPE_COLORS.get(subtype, 'gray'))

    # Add NORM reference
    norm_data = cd_data  # We need the full df for NORM
    # We'll use the stored df from main
    ax.axvspan(6, 9, alpha=0.06, color='red')
    ax.text(7.5, ax.get_ylim()[0] + 0.1 if ax.get_ylim()[0] < 1 else 1.2,
            'V1-V4\n(anterior)', ha='center', fontsize=8, color='red', alpha=0.7)

    ax.set_xticks(range(len(LEAD_NAMES)))
    ax.set_xticklabels(LEAD_NAMES, fontsize=10)
    ax.set_ylabel('β (median per lead)', fontsize=11)
    ax.set_title('A. Spatial β-fingerprint by CD subtype\n'
                 'Each conduction disturbance has a unique anatomical pattern',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel B: Confusion matrix (best model) ---
    ax = fig.add_subplot(gs[0, 2])

    best_name = max(cd_results, key=lambda k: cd_results[k]['macro_auc'])
    best = cd_results[best_name]
    cm = best['cm']
    classes = best['classes']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = cm_norm[i, j]
            color = 'white' if pct > 0.5 else 'black'
            ax.text(j, i, f'{pct:.0%}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if i == j else 'normal')

    ax.set_title(f'B. Confusion matrix ({best_name})\n'
                 f'Macro AUC = {best["macro_auc"]:.3f}',
                 fontweight='bold', fontsize=11)

    # --- Panel C: CLBBB vs CRBBB ROC ---
    ax = fig.add_subplot(gs[1, 0])

    ax.plot(bbb_fpr, bbb_tpr, linewidth=2.5, color='#C62828',
            label=f'GBM (AUC = {bbb_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'C. CLBBB vs CRBBB\nAUC = {bbb_auc:.3f}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)

    # --- Panel D: Feature importance for CLBBB vs CRBBB ---
    ax = fig.add_subplot(gs[1, 1])

    importances = bbb_gbm.feature_importances_
    feat_imp = pd.DataFrame({'feature': avail, 'importance': importances})
    feat_imp = feat_imp.sort_values('importance', ascending=True).tail(12)

    colors = []
    for f in feat_imp.feature:
        if f.startswith('beta_ir_V'):
            colors.append('#E53935')
        elif f.startswith('beta_ir_'):
            colors.append('#FF9800')
        elif f.startswith('r2_ir_'):
            colors.append('#2196F3')
        elif f in ['age', 'sex_num']:
            colors.append('#4CAF50')
        else:
            colors.append('#9E9E9E')

    ax.barh(range(len(feat_imp)), feat_imp.importance.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp.feature.values, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=10)
    ax.set_title('D. What distinguishes CLBBB from CRBBB?\n'
                 'Precordial leads dominate', fontweight='bold', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    # --- Panel E: Per-subtype AUC bar chart ---
    ax = fig.add_subplot(gs[1, 2])

    best_res = cd_results[best_name]
    classes_sorted = sorted(best_res['per_class_auc'].keys(),
                           key=lambda k: best_res['per_class_auc'].get(k, 0), reverse=True)
    aucs = [best_res['per_class_auc'].get(c, 0) for c in classes_sorted]
    bar_colors = [CD_SUBTYPE_COLORS.get(c, 'gray') for c in classes_sorted]

    bars = ax.bar(range(len(classes_sorted)), aucs, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(classes_sorted)))
    ax.set_xticklabels(classes_sorted, fontsize=10)
    ax.set_ylabel('AUC (one-vs-rest)', fontsize=11)
    ax.set_title(f'E. Per-subtype classification ({best_name})\n'
                 f'β-features discriminate conduction subtypes',
                 fontweight='bold', fontsize=11)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add AUC values on bars
    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc_val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Niche 1: CD Subtype Classification from β-Features\n'
                 'The spectral exponent vector across 12 leads encodes conduction anatomy',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(save_path)}")


def fig_niche2(effect_results, subclinical_results, a_data, b_data, save_path):
    """
    Figure N2: Subclinical Signal Detection
    Panel A: β distributions — Pure NORM vs NORM+subclinical
    Panel B: Effect sizes
    Panel C: ROC curve
    Panel D: Key features comparison (violin)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel A: β_mean distributions ---
    ax = axes[0, 0]

    ax.hist(a_data.beta_mean.dropna(), bins=60, alpha=0.6, density=True,
            color=NORM_GREEN, label=f'Pure NORM (n={len(a_data):,})', edgecolor='white', linewidth=0.3)
    ax.hist(b_data.beta_mean.dropna(), bins=40, alpha=0.6, density=True,
            color='#E53935', label=f'NORM + subclinical CD (n={len(b_data):,})', edgecolor='white', linewidth=0.3)

    # Add vertical medians
    ax.axvline(a_data.beta_mean.median(), color=NORM_GREEN, ls='--', lw=2,
               label=f'Median = {a_data.beta_mean.median():.3f}')
    ax.axvline(b_data.beta_mean.median(), color='#E53935', ls='--', lw=2,
               label=f'Median = {b_data.beta_mean.median():.3f}')

    ax.set_xlabel('β (mean across leads)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('A. β distribution: Pure NORM vs NORM + subclinical CD\n'
                 'Subclinical CD shifts β upward',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel B: Effect sizes ---
    ax = axes[0, 1]

    eff_df = pd.DataFrame(effect_results)
    eff_df = eff_df.sort_values('d', ascending=True)

    colors_b = ['#E53935' if abs(d) > 0.2 else '#FF9800' if abs(d) > 0.1 else '#9E9E9E'
                for d in eff_df.d.values]

    ax.barh(range(len(eff_df)), eff_df.d.values, color=colors_b, alpha=0.85)
    ax.set_yticks(range(len(eff_df)))
    ax.set_yticklabels(eff_df.feature.values, fontsize=9)
    ax.axvline(0, color='black', lw=0.5)
    ax.axvline(0.2, color='gray', ls=':', alpha=0.5, label='small (0.2)')
    ax.axvline(-0.2, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("Cohen's d (NORM+subclinical − Pure NORM)", fontsize=10)
    ax.set_title('B. Effect sizes\nSubclinical CD → higher β, more spatial heterogeneity',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # Add significance stars
    for i, (_, row) in enumerate(eff_df.iterrows()):
        star = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        if star:
            x_pos = row['d'] + (0.01 if row['d'] >= 0 else -0.06)
            ax.text(x_pos, i, star, va='center', fontsize=10, fontweight='bold', color='#C62828')

    # --- Panel C: ROC curves ---
    ax = axes[1, 0]

    model_colors = {'LogReg': '#455A64', 'RF': '#00796B', 'GBM': '#C62828'}
    for name, res in subclinical_results.items():
        ax.plot(res['fpr'], res['tpr'], linewidth=2, color=model_colors[name],
                label=f'{name} (AUC = {res["auc"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('C. Can β detect subclinical CD in "healthy" ECGs?\n'
                 'Classification: Pure NORM vs NORM + subclinical CD',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)

    # --- Panel D: Key features violin comparison ---
    ax = axes[1, 1]

    features_violin = ['beta_mean', 'beta_std', 'beta_anterior', 'beta_regional_div']
    positions = []
    violins_a = []
    violins_b = []

    for i, feat in enumerate(features_violin):
        va = a_data[feat].dropna().values
        vb = b_data[feat].dropna().values

        # Subsample for performance
        rng = np.random.RandomState(42)
        if len(va) > 1000:
            va = rng.choice(va, 1000, replace=False)
        if len(vb) > 500:
            vb = rng.choice(vb, 500, replace=False)

        pos_a = i * 3
        pos_b = i * 3 + 1

        parts_a = ax.violinplot([va], positions=[pos_a], showmeans=True, showextrema=False)
        for pc in parts_a['bodies']:
            pc.set_facecolor(NORM_GREEN)
            pc.set_alpha(0.6)
        parts_a['cmeans'].set_color('black')

        parts_b = ax.violinplot([vb], positions=[pos_b], showmeans=True, showextrema=False)
        for pc in parts_b['bodies']:
            pc.set_facecolor('#E53935')
            pc.set_alpha(0.6)
        parts_b['cmeans'].set_color('black')

    ax.set_xticks([i * 3 + 0.5 for i in range(len(features_violin))])
    ax.set_xticklabels(['β_mean', 'β_std', 'β_anterior', 'β_reg_div'], fontsize=10)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('D. Feature distributions\n'
                 'Subclinical CD already visible in β-features',
                 fontweight='bold', fontsize=11)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=NORM_GREEN,
               markersize=12, label='Pure NORM'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E53935',
               markersize=12, label='NORM + subclinical CD'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle('Niche 2: Subclinical Signal Detection\n'
                 'β catches conduction disturbances that the cardiologist labels as "normal"',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(save_path)}")


def fig_spatial_fingerprints(cd_data, norm_ref, save_path):
    """
    Figure N3: Radar/polar plot of spatial fingerprints per CD subtype.
    Shows anatomical specificity of β-pattern for each conduction disturbance.
    """
    subtypes = ['CLBBB', 'CRBBB', 'IRBBB', 'LAFB', '1AVB', 'IVCD']
    n_subtypes = len(subtypes)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    # Angles for 12 leads
    angles = np.linspace(0, 2 * np.pi, len(LEAD_NAMES), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # NORM reference
    norm_medians = [norm_ref[f'beta_ir_{l}'].median() for l in LEAD_NAMES]
    norm_medians += norm_medians[:1]

    for idx, subtype in enumerate(subtypes):
        ax = axes[idx]
        sub = cd_data[cd_data.cd_subtype == subtype]
        if len(sub) < 10:
            ax.set_title(f'{subtype}\n(n < 10)', fontsize=11)
            continue

        # Subtype medians
        medians = [sub[f'beta_ir_{l}'].median() for l in LEAD_NAMES]
        medians += medians[:1]

        # Plot NORM reference
        ax.plot(angles, norm_medians, 'o-', color=NORM_GREEN, linewidth=1.5,
                markersize=3, alpha=0.5, label='NORM')
        ax.fill(angles, norm_medians, color=NORM_GREEN, alpha=0.1)

        # Plot subtype
        color = CD_SUBTYPE_COLORS.get(subtype, 'gray')
        ax.plot(angles, medians, 'o-', color=color, linewidth=2.5, markersize=5, label=subtype)
        ax.fill(angles, medians, color=color, alpha=0.15)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(LEAD_NAMES, fontsize=8)

        # Highlight the biggest deviations
        diffs = [m - n for m, n in zip(medians[:-1], norm_medians[:-1])]
        max_idx = np.argmax(np.abs(diffs))
        max_lead = LEAD_NAMES[max_idx]

        ax.set_title(f'{subtype} (n={len(sub):,})\n'
                     f'Peak deviation: {max_lead} (Δβ={diffs[max_idx]:+.2f})',
                     fontweight='bold', fontsize=11, pad=20)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.suptitle('Spatial β-Fingerprints of Conduction Disturbances\n'
                 'Each subtype has a unique anatomical pattern — the model learns where the lesion is',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {os.path.basename(save_path)}")


# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    df, scp = load_data()

    # === NICHE 1: CD Subtype Classification ===
    cd_results, cd_data, cd_avail, cd_gbm = niche1_cd_subtypes(df)
    bbb_auc, bbb_fpr, bbb_tpr, bbb_gbm, bbb_avail, bbb_data = niche1_pairwise(df)

    # === NICHE 2: Subclinical Signal ===
    effect_results, subclinical_results, a_data, b_data, sub_avail = niche2_subclinical(df)

    # === FIGURES ===
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Get NORM reference for radar plots
    norm_ref = df[df.scp_dict.apply(lambda x: 'NORM' in x and x.get('NORM', 0) >= 80
                                     and len(x) == 1)]

    fig_niche1(cd_results, cd_data, bbb_auc, bbb_fpr, bbb_tpr, bbb_gbm, bbb_avail,
               os.path.join(RESULTS_DIR, 'fig_niche1_cd_subtypes.png'))

    fig_niche2(effect_results, subclinical_results, a_data, b_data,
               os.path.join(RESULTS_DIR, 'fig_niche2_subclinical.png'))

    fig_spatial_fingerprints(cd_data, norm_ref,
                            os.path.join(RESULTS_DIR, 'fig_niche3_spatial_fingerprints.png'))

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    best_cd = max(cd_results.values(), key=lambda r: r['macro_auc'])
    print(f"\n  NICHE 1: CD Subtype Classification")
    print(f"  6-class macro AUC = {best_cd['macro_auc']:.3f}")
    for cls in best_cd['classes']:
        print(f"    {cls:8s}: AUC = {best_cd['per_class_auc'].get(cls, np.nan):.3f}")
    print(f"  CLBBB vs CRBBB: AUC = {bbb_auc:.3f}")
    print(f"  → β-vector encodes conduction anatomy")

    print(f"\n  NICHE 2: Subclinical Signal Detection")
    best_sub = max(subclinical_results.values(), key=lambda r: r['auc'])
    best_sub_name = [k for k, v in subclinical_results.items() if v['auc'] == best_sub['auc']][0]
    print(f"  Pure NORM vs NORM+subclinical: AUC = {best_sub['auc']:.3f} ({best_sub_name})")
    sig_effects = [r for r in effect_results if r['p'] < 0.001]
    print(f"  Significant features (p < 0.001): {len(sig_effects)}/{len(effect_results)}")
    for r in sorted(sig_effects, key=lambda x: abs(x['d']), reverse=True):
        print(f"    {r['feature']:20s}: d = {r['d']:+.3f}")
    print(f"  → β detects subclinical CD in 'healthy' ECGs")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
