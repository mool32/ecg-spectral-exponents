"""
Part I: Cardiac Aging in Healthy Hearts (PTB-XL)
=================================================
From Orchestra to Musicians: aging of the healthy ECG
through spectral exponent β and its spatial structure.

Uses pre-computed β features from criticality_analysis.py.
"""

import os, ast, warnings
import numpy as np, pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11, 'figure.facecolor': 'white'})
sns.set_palette("colorblind")

DATA_DIR = 'ptb-xl'
RESULTS_DIR = 'results'
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

REGIONS = {
    'Anterior': ['V1','V2','V3','V4'],
    'Lateral':  ['I','aVL','V5','V6'],
    'Inferior': ['II','III','aVF'],
}

DECADES = [(18,29), (30,39), (40,49), (50,59), (60,69), (70,79), (80,100)]
DECADE_LABELS = ['18-29','30-39','40-49','50-59','60-69','70-79','80+']


# ============================================================
# 1. LOAD DATA & PREPARE NORM-ONLY SAMPLE
# ============================================================
def load_norm_data():
    """Load metadata + β features, filter to clean NORM only."""
    df = pd.read_csv(f'{DATA_DIR}/ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(f'{DATA_DIR}/scp_statements.csv', index_col=0)
    scp_diag = scp[scp.diagnostic == 1]

    def get_sc(codes):
        r = set()
        for c, l in codes.items():
            if c in scp_diag.index and l > 0:
                s = scp_diag.loc[c, 'diagnostic_class']
                if pd.notna(s): r.add(s)
        return r

    df['superclasses'] = df.scp_codes.apply(get_sc)
    df['n_sc'] = df.superclasses.apply(len)
    df['is_clean'] = df.n_sc == 1
    df['clean_superclass'] = df.apply(
        lambda r: list(r.superclasses)[0] if r.is_clean else None, axis=1)

    # Max likelihood of any diagnosis
    df['max_likelihood'] = df.scp_codes.apply(lambda c: max(c.values()) if c else 0)

    # Load β features
    beta_path = f'{RESULTS_DIR}/beta_features.csv'
    if not os.path.exists(beta_path):
        beta_path = f'{RESULTS_DIR}/beta_features_partial.csv'
    beta_df = pd.read_csv(beta_path, index_col='ecg_id')
    dm = df.join(beta_df, how='inner')

    # Filter: NORM only, clean, with valid β
    norm = dm[(dm.is_clean) & (dm.clean_superclass == 'NORM') & (dm.beta_mean.notna())].copy()

    # Additional filter: exclude borderline (any diagnosis likelihood > 50%)
    # Already clean NORM, but extra safety
    norm = norm[(norm.age >= 18) & (norm.age <= 95)]

    print(f"NORM sample: {len(norm)} records, age {norm.age.min():.0f}–{norm.age.max():.0f}")
    print(f"Sex: M={( norm.sex==0).sum()}, F={(norm.sex==1).sum()}")
    print(f"Decade distribution:")
    for lo, hi in DECADES:
        n = ((norm.age >= lo) & (norm.age <= hi)).sum()
        print(f"  {lo}-{hi}: {n}")

    return norm, dm


# ============================================================
# 2. COMPUTE AGING METRICS
# ============================================================
def compute_aging_metrics(norm):
    """Compute 7 aging-relevant metrics from pre-computed β arrays."""
    beta_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]

    # Already have: beta_mean, beta_std (=σ_β)
    # Need to add: IQR_β, mean pairwise correlation, regional β, HF power ratio, spectral entropy

    # 3. IQR_β
    norm['beta_iqr'] = norm[beta_cols].apply(
        lambda row: row.dropna().quantile(0.75) - row.dropna().quantile(0.25), axis=1)

    # 4. Mean pairwise correlation — computed per-record as mean off-diagonal of lead β
    # This needs the full dataset correlation, so we compute it per decade in the analysis
    # For individual records, we use β_std and IQR as proxies
    # Actually, we can compute individual-level "deviation from personal mean" pattern
    def lead_range(row):
        vals = row[beta_cols].dropna()
        return vals.max() - vals.min() if len(vals) >= 6 else np.nan
    norm['beta_range'] = norm.apply(lead_range, axis=1)

    # 5. Regional β
    for rname, leads in REGIONS.items():
        cols = [f'beta_ir_{l}' for l in leads]
        norm[f'beta_{rname.lower()}'] = norm[cols].mean(axis=1)

    # Regional divergence: max β_region - min β_region
    norm['beta_regional_div'] = norm[['beta_anterior','beta_lateral','beta_inferior']].apply(
        lambda r: r.max() - r.min(), axis=1)

    # 6. β coefficient of variation across leads
    norm['beta_cv'] = norm.apply(
        lambda r: r[beta_cols].std() / r[beta_cols].mean() if r[beta_cols].mean() > 0 else np.nan, axis=1)

    # BMI (if height/weight available)
    norm['bmi'] = norm.weight / ((norm.height / 100) ** 2)
    norm.loc[norm.bmi > 60, 'bmi'] = np.nan  # clean outliers
    norm.loc[norm.bmi < 12, 'bmi'] = np.nan

    print(f"\nComputed metrics. Sample: {len(norm)}")
    return norm


# ============================================================
# 3. FIG I-1: AGING TRAJECTORIES
# ============================================================
def fig_aging_trajectories(norm):
    """7 panels: metric vs age with LOESS + linear fit, color=sex."""
    metrics = [
        ('beta_mean', 'β_mean', 'Global spectral exponent'),
        ('beta_std', 'σ_β', 'Cross-lead dispersion'),
        ('beta_iqr', 'IQR_β', 'Cross-lead IQR'),
        ('beta_cv', 'CV_β', 'Coefficient of variation'),
        ('beta_regional_div', 'Regional div.', 'Regional β divergence'),
        ('beta_anterior', 'β_anterior', 'Anterior wall β'),
        ('beta_inferior', 'β_inferior', 'Inferior wall β'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes_flat = axes.flat

    sex_colors = {0: '#3498db', 1: '#e74c3c'}
    sex_labels = {0: 'Male', 1: 'Female'}

    summary_text = "Metric          | Spearman ρ |  p-value  | ρ_M    | ρ_F\n"
    summary_text += "-" * 65 + "\n"

    for i, (col, label, desc) in enumerate(metrics):
        ax = axes_flat[i]
        valid = norm.dropna(subset=[col, 'age'])

        for sex in [0, 1]:
            sub = valid[valid.sex == sex]
            ax.scatter(sub.age, sub[col], alpha=0.08, s=4,
                       color=sex_colors[sex], label=sex_labels[sex])

            # LOWESS
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(sub[col], sub.age, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=sex_colors[sex],
                    lw=2.5, alpha=0.8)

        # Overall Spearman
        rho, p = stats.spearmanr(valid.age, valid[col])
        rho_m, p_m = stats.spearmanr(valid[valid.sex==0].age, valid[valid.sex==0][col])
        rho_f, p_f = stats.spearmanr(valid[valid.sex==1].age, valid[valid.sex==1][col])

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_title(f'{label}\nρ={rho:+.3f} {sig}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Age')
        ax.set_ylabel(label)

        if i == 0:
            ax.legend(fontsize=8, markerscale=3)

        summary_text += f"{label:16s}| {rho:+.3f}     | {p:.2e} | {rho_m:+.3f} | {rho_f:+.3f}\n"

    # Last panel: summary stats
    ax = axes_flat[7]
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8.5,
            va='top', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Summary', fontweight='bold')

    plt.suptitle('Fig. I-1: Aging Trajectories in Healthy Hearts (NORM only)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{RESULTS_DIR}/FigI1_aging_trajectories.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved FigI1")
    return summary_text


# ============================================================
# 4. FIG I-2: DECADE PORTRAITS (correlation matrices)
# ============================================================
def fig_decade_portraits(norm):
    """12×12 β correlation matrix for each age decade."""
    beta_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    mean_corrs = []
    for i, ((lo, hi), label) in enumerate(zip(DECADES, DECADE_LABELS)):
        ax = axes.flat[i]
        sub = norm[(norm.age >= lo) & (norm.age <= hi)][beta_cols].dropna()
        if len(sub) < 20:
            ax.set_title(f'{label} (n<20)')
            mean_corrs.append(np.nan)
            continue

        corr = sub.corr()
        corr.index = corr.columns = LEAD_NAMES
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        mean_r = corr.values[np.tril_indices_from(corr.values, -1)].mean()
        mean_corrs.append(mean_r)

        sns.heatmap(corr, mask=mask, vmin=0, vmax=1, cmap='YlOrRd',
                    annot=True, fmt='.2f', square=True, ax=ax,
                    cbar_kws={'shrink': 0.6}, linewidths=0.3,
                    annot_kws={'fontsize': 7})
        ax.set_title(f'{label}\nn={len(sub)}, mean r={mean_r:.2f}',
                     fontsize=10, fontweight='bold')

    # Trend panel
    ax = axes.flat[7]
    valid_idx = [i for i, v in enumerate(mean_corrs) if not np.isnan(v)]
    mid_ages = [(lo+hi)/2 for lo, hi in DECADES]
    ax.plot([mid_ages[i] for i in valid_idx],
            [mean_corrs[i] for i in valid_idx],
            'o-', color='#c0392b', lw=2.5, markersize=8)
    ax.set_xlabel('Age (decade midpoint)')
    ax.set_ylabel('Mean inter-lead β correlation')
    ax.set_title('Coherence vs Age', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Spearman on decade-level
    if len(valid_idx) >= 4:
        rho, p = stats.spearmanr([mid_ages[i] for i in valid_idx],
                                  [mean_corrs[i] for i in valid_idx])
        ax.text(0.05, 0.05, f'ρ={rho:+.3f}, p={p:.3f}',
                transform=ax.transAxes, fontsize=10)

    plt.suptitle('Fig. I-2: Cross-Lead β Coherence by Age Decade',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{RESULTS_DIR}/FigI2_decade_portraits.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved FigI2")


# ============================================================
# 5. FIG I-3: REGIONAL DIVERGENCE
# ============================================================
def fig_regional_divergence(norm):
    """3 lines (anterior, lateral, inferior) β_mean vs age."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Regional β vs age
    ax = axes[0]
    region_colors = {'Anterior': '#e74c3c', 'Lateral': '#3498db', 'Inferior': '#2ecc71'}

    for rname in ['Anterior', 'Lateral', 'Inferior']:
        col = f'beta_{rname.lower()}'
        valid = norm.dropna(subset=[col, 'age'])
        ax.scatter(valid.age, valid[col], alpha=0.04, s=3, color=region_colors[rname])
        sm = lowess(valid[col], valid.age, frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color=region_colors[rname], lw=2.5, label=rname)

    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Median β (region)', fontsize=12)
    ax.set_title('Regional β Trajectories', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Regional divergence vs age
    ax = axes[1]
    valid = norm.dropna(subset=['beta_regional_div', 'age'])
    for sex, color, label in [(0, '#3498db', 'Male'), (1, '#e74c3c', 'Female')]:
        sub = valid[valid.sex == sex]
        ax.scatter(sub.age, sub.beta_regional_div, alpha=0.06, s=3, color=color)
        sm = lowess(sub.beta_regional_div, sub.age, frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.5, label=label)

    rho, p = stats.spearmanr(valid.age, valid.beta_regional_div)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Regional β divergence (max−min)', fontsize=12)
    ax.set_title(f'Regional Divergence (ρ={rho:+.3f}, p={p:.2e})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. I-3: Regional β Divergence with Age',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{RESULTS_DIR}/FigI3_regional_divergence.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved FigI3")


# ============================================================
# 6. FIG I-4: BREAKPOINT ANALYSIS
# ============================================================
def fig_breakpoint_analysis(norm):
    """Segmented regression for σ_β vs age. Find breakpoint."""
    from scipy.optimize import minimize_scalar

    valid = norm.dropna(subset=['beta_std', 'age']).copy()
    x = valid.age.values
    y = valid.beta_std.values

    # Try segmented regression: fit two lines, find optimal breakpoint
    def segmented_rss(bp):
        left = x <= bp
        right = x > bp
        if left.sum() < 30 or right.sum() < 30:
            return 1e10
        rss = 0
        for mask in [left, right]:
            xm, ym = x[mask], y[mask]
            sl, ic, _, _, _ = stats.linregress(xm, ym)
            rss += np.sum((ym - (ic + sl * xm)) ** 2)
        return rss

    result = minimize_scalar(segmented_rss, bounds=(30, 80), method='bounded')
    bp_age = result.x

    # Fit the two segments
    left = x <= bp_age
    right = x > bp_age
    sl_l, ic_l, _, _, _ = stats.linregress(x[left], y[left])
    sl_r, ic_r, _, _, _ = stats.linregress(x[right], y[right])

    # Overall linear
    sl_all, ic_all, rv_all, _, _ = stats.linregress(x, y)

    # Compare: segmented vs linear (F-test via RSS)
    rss_linear = np.sum((y - (ic_all + sl_all * x)) ** 2)
    rss_segmented = result.fun
    n = len(x)
    f_stat = ((rss_linear - rss_segmented) / 2) / (rss_segmented / (n - 4))
    p_ftest = 1 - stats.f.cdf(f_stat, 2, n - 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: σ_β vs age with breakpoint
    ax = axes[0]
    ax.scatter(x, y, alpha=0.06, s=3, color='gray')

    # LOWESS
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sm = lowess(y, x, frac=0.2)
    ax.plot(sm[:, 0], sm[:, 1], 'k', lw=2.5, label='LOWESS', zorder=5)

    # Segmented fit
    x_left = np.linspace(x[left].min(), bp_age, 100)
    x_right = np.linspace(bp_age, x[right].max(), 100)
    ax.plot(x_left, ic_l + sl_l * x_left, 'b--', lw=2, label=f'Left: slope={sl_l:.4f}/yr')
    ax.plot(x_right, ic_r + sl_r * x_right, 'r--', lw=2, label=f'Right: slope={sl_r:.4f}/yr')
    ax.axvline(bp_age, color='orange', ls=':', lw=2, alpha=0.8,
               label=f'Breakpoint: {bp_age:.0f} yr')

    rho, p = stats.spearmanr(x, y)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('σ_β (cross-lead dispersion)', fontsize=12)
    ax.set_title(f'σ_β vs Age (ρ={rho:+.3f}, p={p:.2e})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: summary + quadratic fit
    ax = axes[1]

    # Quadratic fit
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    from sklearn.linear_model import LinearRegression as LR
    model = LR().fit(X_poly, y)
    x_plot = np.linspace(18, 95, 200)
    y_pred = model.predict(poly.transform(x_plot.reshape(-1, 1)))

    ax.scatter(x, y, alpha=0.06, s=3, color='gray')
    ax.plot(x_plot, y_pred, 'purple', lw=2.5, label='Quadratic fit')
    ax.plot(sm[:, 0], sm[:, 1], 'k', lw=2.5, label='LOWESS', alpha=0.5)

    txt = f"Breakpoint analysis:\n"
    txt += f"  Optimal breakpoint: {bp_age:.1f} years\n"
    txt += f"  Left slope:  {sl_l:+.5f}/yr\n"
    txt += f"  Right slope: {sl_r:+.5f}/yr\n"
    txt += f"  Slope ratio: {sl_r/sl_l:.1f}x\n" if sl_l != 0 else ""
    txt += f"  F-test (segmented vs linear): F={f_stat:.1f}, p={p_ftest:.2e}\n"
    txt += f"\nOverall Spearman: ρ={rho:+.3f}, p={p:.2e}\n"
    txt += f"N = {len(x)}"

    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_xlabel('Age')
    ax.set_ylabel('σ_β')
    ax.set_title('Quadratic Fit & Breakpoint', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Fig. I-4: Breakpoint Analysis of Cardiac Aging',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{RESULTS_DIR}/FigI4_breakpoint.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved FigI4 (breakpoint at {bp_age:.0f} years)")
    return bp_age


# ============================================================
# 7. CONFOUND ANALYSIS + ANCOVA
# ============================================================
def aging_confound_analysis(norm):
    """ANCOVA: metric ~ age + sex + device (+ BMI if available)."""
    print("\n=== AGING CONFOUND ANALYSIS ===")
    metrics = ['beta_mean', 'beta_std', 'beta_iqr', 'beta_cv', 'beta_regional_div']

    for col in metrics:
        valid = norm.dropna(subset=[col, 'age', 'sex'])
        if 'device' in valid.columns:
            valid = valid.dropna(subset=['device'])

        # Partial correlation: metric ~ age controlling for sex
        pc = pg.partial_corr(data=valid, x='age', y=col, covar='sex', method='spearman')
        rho = pc['r'].values[0]
        p = pc['p-val'].values[0]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {col:20s}: partial ρ(age|sex) = {rho:+.3f}, p = {p:.2e} {sig}")

    # Full regression: β_std ~ age + sex + age×sex
    print("\n  Multiple regression: σ_β ~ age + sex + age×sex")
    valid = norm.dropna(subset=['beta_std', 'age', 'sex']).copy()
    valid['age_x_sex'] = valid.age * valid.sex
    import statsmodels.api as sm
    X = sm.add_constant(valid[['age', 'sex', 'age_x_sex']])
    model = sm.OLS(valid.beta_std, X).fit()
    print(model.summary2().tables[1].to_string())


# ============================================================
# 8. BIOLOGICAL AGE MODEL (Part IV-C)
# ============================================================
def biological_age_model(norm, dm):
    """Train model: β features → chronological age (on NORM), apply to pathological."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    beta_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]
    features = ['beta_mean', 'beta_std', 'beta_iqr', 'beta_cv',
                'beta_anterior', 'beta_lateral', 'beta_inferior', 'beta_regional_div',
                'sex'] + beta_cols

    # Filter features that exist
    avail = [f for f in features if f in norm.columns]
    train_data = norm[avail + ['age']].dropna()

    X = train_data[avail]
    y = train_data['age']

    print(f"\nBiological Age Model: {len(X)} training samples, {len(avail)} features")

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42,
                                       learning_rate=0.05)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"  CV R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X, y)

    # Predicted age for NORM
    norm_pred = norm.copy()
    valid_mask = norm_pred[avail].notna().all(axis=1)
    norm_pred.loc[valid_mask, 'predicted_age'] = model.predict(norm_pred.loc[valid_mask, avail])
    norm_pred['age_accel'] = norm_pred['predicted_age'] - norm_pred['age']

    # Apply to pathological records
    all_data = dm[dm.beta_mean.notna()].copy()
    for rname, leads in REGIONS.items():
        cols = [f'beta_ir_{l}' for l in leads]
        all_data[f'beta_{rname.lower()}'] = all_data[cols].mean(axis=1)
    all_data['beta_iqr'] = all_data[beta_cols].apply(
        lambda row: row.dropna().quantile(0.75) - row.dropna().quantile(0.25), axis=1)
    all_data['beta_cv'] = all_data.apply(
        lambda r: r[beta_cols].std() / r[beta_cols].mean() if r[beta_cols].mean() > 0 else np.nan, axis=1)
    all_data['beta_regional_div'] = all_data[['beta_anterior','beta_lateral','beta_inferior']].apply(
        lambda r: r.max() - r.min(), axis=1)

    path_mask = all_data[avail].notna().all(axis=1)
    all_data.loc[path_mask, 'predicted_age'] = model.predict(all_data.loc[path_mask, avail])
    all_data['age_accel'] = all_data['predicted_age'] - all_data['age']

    # Assign superclass to all_data
    scp = pd.read_csv(f'{DATA_DIR}/scp_statements.csv', index_col=0)
    scp_diag = scp[scp.diagnostic == 1]
    df_meta = pd.read_csv(f'{DATA_DIR}/ptbxl_database.csv', index_col='ecg_id')
    df_meta.scp_codes = df_meta.scp_codes.apply(ast.literal_eval)
    def get_primary(codes):
        bs, bl = None, 0
        for c, l in codes.items():
            if c in scp_diag.index and l > bl:
                s = scp_diag.loc[c, 'diagnostic_class']
                if pd.notna(s): bs, bl = s, l
        return bs
    df_meta['primary_sc'] = df_meta.scp_codes.apply(get_primary)
    all_data = all_data.join(df_meta[['primary_sc']], how='left')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Predicted vs actual for NORM
    ax = axes[0]
    v = norm_pred.dropna(subset=['predicted_age'])
    ax.scatter(v.age, v.predicted_age, alpha=0.05, s=3, color='#27ae60')
    ax.plot([18, 95], [18, 95], 'k--', alpha=0.5)
    rho, _ = stats.spearmanr(v.age, v.predicted_age)
    mae = np.abs(v.age - v.predicted_age).mean()
    ax.set_xlabel('Chronological Age')
    ax.set_ylabel('Predicted Age (from β)')
    ax.set_title(f'NORM: β-Age (ρ={rho:.3f}, MAE={mae:.1f} yr)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Age acceleration by diagnosis
    ax = axes[1]
    sc_order = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    sc_colors = {'NORM':'#27ae60','MI':'#c0392b','STTC':'#2980b9','CD':'#8e44ad','HYP':'#d68910'}
    data_aa = []
    labels_aa = []
    for sc in sc_order:
        vals = all_data[all_data.primary_sc == sc].age_accel.dropna()
        if len(vals) > 10:
            data_aa.append(vals.values)
            labels_aa.append(f'{sc}\n(n={len(vals)})')

    if data_aa:
        parts = ax.violinplot(data_aa, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            c = list(sc_colors.values())[i] if i < len(sc_colors) else 'gray'
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(labels_aa)+1))
        ax.set_xticklabels(labels_aa)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('Age Acceleration (years)')
        ax.set_title('Cardiac Age Acceleration by Diagnosis', fontweight='bold')

        # Stats
        for i, sc in enumerate(sc_order):
            if sc == 'NORM': continue
            vals = all_data[all_data.primary_sc == sc].age_accel.dropna()
            vals_n = all_data[all_data.primary_sc == 'NORM'].age_accel.dropna()
            if len(vals) > 10 and len(vals_n) > 10:
                _, p = stats.mannwhitneyu(vals, vals_n)
                med = vals.median()
                print(f"  {sc}: median age accel = {med:+.1f} yr, p vs NORM = {p:.2e}")

    plt.suptitle('Biological Age Model', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{RESULTS_DIR}/FigIV_biological_age.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved FigIV_biological_age")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Part I: Cardiac Aging in Healthy Hearts")
    print("=" * 60)

    norm, dm = load_norm_data()
    norm = compute_aging_metrics(norm)

    print("\n--- Fig I-1: Aging Trajectories ---")
    summary = fig_aging_trajectories(norm)
    print(summary)

    print("\n--- Fig I-2: Decade Portraits ---")
    fig_decade_portraits(norm)

    print("\n--- Fig I-3: Regional Divergence ---")
    fig_regional_divergence(norm)

    print("\n--- Fig I-4: Breakpoint Analysis ---")
    bp = fig_breakpoint_analysis(norm)

    print("\n--- Confound Analysis ---")
    aging_confound_analysis(norm)

    print("\n--- Part IV-C: Biological Age Model ---")
    biological_age_model(norm, dm)

    print("\n" + "=" * 60)
    print("PART I COMPLETE. Figures in:", RESULTS_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
