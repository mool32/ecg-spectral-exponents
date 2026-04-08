"""
Generate the final analysis notebook as a Python script (convertible to .ipynb).
Run after all data has been processed.
This script generates all figures and statistics for the paper/notebook.
"""

import os, ast, warnings, sys
import numpy as np, pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pingouin as pg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Style ──
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'figure.facecolor': 'white',
})
sns.set_palette("colorblind")

DATA_DIR = 'ptb-xl'
RESULTS_DIR = 'results'
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
SUPERCLASSES = ['NORM','MI','STTC','CD','HYP']
SC_LABELS = {
    'NORM': 'Normal', 'MI': 'Myocardial\nInfarction',
    'STTC': 'ST/T\nChanges', 'CD': 'Conduction\nDisturbance',
    'HYP': 'Hypertrophy'
}
SC_COLORS = {'NORM':'#27ae60','MI':'#c0392b','STTC':'#2980b9','CD':'#8e44ad','HYP':'#d68910'}


def load_data():
    """Load metadata + β features."""
    df = pd.read_csv(f'{DATA_DIR}/ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(f'{DATA_DIR}/scp_statements.csv', index_col=0)
    scp = scp[scp.diagnostic == 1]

    def get_sc(c):
        r = set()
        for k, v in c.items():
            if k in scp.index and v > 0:
                s = scp.loc[k, 'diagnostic_class']
                if pd.notna(s): r.add(s)
        return r

    def get_primary(c):
        bs, bl = None, 0
        for k, v in c.items():
            if k in scp.index and v > bl:
                s = scp.loc[k, 'diagnostic_class']
                if pd.notna(s): bs, bl = s, v
        return bs

    def get_subs(c):
        return {k: v for k, v in c.items() if k in scp.index and v > 0}

    df['superclasses'] = df.scp_codes.apply(get_sc)
    df['n_sc'] = df.superclasses.apply(len)
    df['is_clean'] = df.n_sc == 1
    df['clean_superclass'] = df.apply(lambda r: list(r.superclasses)[0] if r.is_clean else None, axis=1)
    df['primary_superclass'] = df.scp_codes.apply(get_primary)
    df['max_likelihood'] = df.scp_codes.apply(lambda c: max(c.values()) if c else 0)
    df['subclasses'] = df.scp_codes.apply(get_subs)

    # Load beta features
    beta_path = f'{RESULTS_DIR}/beta_features_partial.csv'
    if not os.path.exists(beta_path):
        beta_path = f'{RESULTS_DIR}/beta_features.csv'
    beta_df = pd.read_csv(beta_path, index_col='ecg_id')

    dm = df.join(beta_df, how='inner')
    return dm, scp


def generate_all_figures(dm, scp):
    """Generate publication-quality figures."""
    clean = dm[dm.is_clean & dm.beta_mean.notna()].copy()
    beta_norm = clean[clean.clean_superclass == 'NORM'].beta_mean.median()
    beta_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]

    print(f"Dataset: {len(dm)} records, {len(clean)} clean")
    print(f"β_NORM reference: {beta_norm:.3f}")
    for sc in SUPERCLASSES:
        n = (clean.clean_superclass == sc).sum()
        print(f"  {sc}: {n}")

    # ================================================================
    # FIGURE 1: CRITICALITY LANDSCAPE (enhanced raincloud)
    # ================================================================
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1]*5, wspace=0.05)

    all_betas = []
    for sc in SUPERCLASSES:
        all_betas.append(clean[clean.clean_superclass == sc].beta_mean.dropna().values)

    # Combine into one axis for nicer layout
    fig, ax = plt.subplots(figsize=(12, 7))

    # Box + strip
    data_list = []
    for i, sc in enumerate(SUPERCLASSES):
        vals = clean[clean.clean_superclass == sc].beta_mean.dropna()
        for v in vals:
            data_list.append({'Diagnosis': SC_LABELS[sc], 'β': v, 'SC': sc})
    plot_df = pd.DataFrame(data_list)

    # Violin
    parts = ax.violinplot(all_betas, positions=range(5), showmeans=False, showmedians=False, widths=0.75)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(SC_COLORS.values())[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor('none')

    # Box overlay
    bp = ax.boxplot(all_betas, positions=range(5), widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3,
                    medianprops=dict(color='white', linewidth=2),
                    boxprops=dict(linewidth=0))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(list(SC_COLORS.values())[i])
        patch.set_alpha(0.9)

    # Reference lines
    ax.axhline(beta_norm, color='#27ae60', ls='--', lw=2, alpha=0.6,
               label=f'β_NORM = {beta_norm:.2f}', zorder=1)
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.4,
               label='β = 1 (1/f pink noise)', zorder=1)

    # Annotations
    for i, sc in enumerate(SUPERCLASSES):
        n = len(all_betas[i])
        med = np.median(all_betas[i])
        ax.text(i, ax.get_ylim()[1] - 0.05, f'n={n}', ha='center', fontsize=9, color='gray')

    ax.set_xticks(range(5))
    ax.set_xticklabels([SC_LABELS[s] for s in SUPERCLASSES], fontsize=11)
    ax.set_ylabel('β  (aperiodic spectral exponent)', fontsize=13)
    ax.set_title('The Criticality Landscape of the Heart', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0.2, 3.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/Fig1_criticality_landscape_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig1 ✓")

    # ================================================================
    # FIGURE 2: SPATIAL β FINGERPRINT
    # ================================================================
    scp_diag = scp

    rows = []
    for idx, rec in dm.iterrows():
        if pd.isna(rec.get('beta_mean', np.nan)):
            continue
        for code, lk in rec.subclasses.items():
            if code in scp_diag.index and lk >= 80:
                row = {'subclass': code, 'superclass': scp_diag.loc[code, 'diagnostic_class']}
                for c in beta_cols:
                    row[c] = rec.get(c, np.nan)
                rows.append(row)

    sub_df = pd.DataFrame(rows)
    counts = sub_df.subclass.value_counts()
    valid_subs = counts[counts >= 20].index
    sub_df = sub_df[sub_df.subclass.isin(valid_subs)]

    if not sub_df.empty:
        hm = sub_df.groupby('subclass')[beta_cols].median()
        hm.columns = LEAD_NAMES

        # Normalize: subtract NORM median per lead
        norm_row = hm.loc['NORM'] if 'NORM' in hm.index else hm.median()
        hm_diff = hm.subtract(norm_row)
        hm_diff = hm_diff.loc[hm_diff.abs().mean(axis=1).sort_values(ascending=False).index]

        fig, axes = plt.subplots(1, 2, figsize=(20, max(5, len(hm)*0.45)),
                                 gridspec_kw={'width_ratios': [1, 1]})

        # Absolute values
        ax = axes[0]
        sns.heatmap(hm.loc[hm_diff.index], center=beta_norm, cmap='RdBu_r',
                    annot=True, fmt='.2f', linewidths=0.3, ax=ax,
                    cbar_kws={'label': 'Median β', 'shrink': 0.8})
        ax.set_title('Absolute β', fontsize=13, fontweight='bold')
        ax.set_ylabel('Subclass')

        # Difference from NORM
        ax = axes[1]
        sns.heatmap(hm_diff, center=0, cmap='RdBu_r',
                    annot=True, fmt='.2f', linewidths=0.3, ax=ax,
                    cbar_kws={'label': 'Δβ from NORM', 'shrink': 0.8})
        ax.set_title('Deviation from Normal', fontsize=13, fontweight='bold')
        ax.set_ylabel('')

        plt.suptitle('Figure 2: Spatial β Fingerprint', fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{RESULTS_DIR}/Fig2_spatial_fingerprint_final.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("  Fig2 ✓")

    # ================================================================
    # FIGURE 3: CROSS-LEAD COHERENCE
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, sc in enumerate(SUPERCLASSES):
        ax = axes.flat[i]
        grp = clean[clean.clean_superclass == sc][beta_cols].dropna()
        if len(grp) < 10:
            ax.set_title(f'{sc} (n<10)'); continue
        corr = grp.corr()
        corr.index = corr.columns = LEAD_NAMES
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, vmin=0, vmax=1, cmap='YlOrRd',
                    annot=True, fmt='.2f', square=True, ax=ax,
                    cbar_kws={'shrink': 0.6}, linewidths=0.5)
        mean_corr = corr.values[np.tril_indices_from(corr.values, -1)].mean()
        ax.set_title(f'{SC_LABELS[sc]}\nn={len(grp)}, mean r={mean_corr:.2f}',
                     fontsize=11, fontweight='bold')

    # Difference
    ax = axes.flat[5]
    nc = clean[clean.clean_superclass == 'NORM'][beta_cols].dropna().corr()
    pc = clean[clean.clean_superclass != 'NORM'][beta_cols].dropna().corr()
    diff = nc - pc
    diff.index = diff.columns = LEAD_NAMES
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
    sns.heatmap(diff, mask=mask, center=0, cmap='RdBu_r',
                annot=True, fmt='.2f', square=True, ax=ax,
                cbar_kws={'shrink': 0.6}, linewidths=0.5)
    ax.set_title('NORM − Pathology\n(coherence loss)', fontsize=11, fontweight='bold')

    plt.suptitle('Figure 3: Cross-Lead β Coherence', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{RESULTS_DIR}/Fig3_coherence_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig3 ✓")

    # ================================================================
    # FIGURE 4: DISTANCE FROM CRITICALITY vs CONFIDENCE
    # ================================================================
    v = dm[dm.beta_mean.notna() & dm.primary_superclass.notna()].copy()
    v['delta_norm'] = np.abs(v.beta_mean - beta_norm)

    # Only pathological records for this analysis
    path_only = v[v.primary_superclass != 'NORM']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]

    for sc in ['MI', 'STTC', 'CD', 'HYP']:
        s = path_only[path_only.primary_superclass == sc]
        if len(s) > 0:
            ax.scatter(s.max_likelihood, s.delta_norm, alpha=0.2, s=12,
                       color=SC_COLORS[sc], label=f'{sc} (n={len(s)})', edgecolors='none')
            # Trend line
            if len(s) > 20:
                z = np.polyfit(s.max_likelihood, s.delta_norm, 1)
                x_fit = np.linspace(s.max_likelihood.min(), s.max_likelihood.max(), 100)
                ax.plot(x_fit, np.polyval(z, x_fit), color=SC_COLORS[sc], lw=2, alpha=0.7)

    ax.set_xlabel('Diagnostic Likelihood (%)', fontsize=12)
    ax.set_ylabel('|β − β_NORM|', fontsize=12)
    ax.set_title('Distance from Normal vs Diagnostic Confidence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Stats
    ax2 = axes[1]
    txt = f"Reference: β_NORM = {beta_norm:.3f}\n\n"
    txt += "Spearman ρ (pathological records only):\n"
    txt += "  |β − β_NORM| vs diagnostic likelihood\n\n"
    for sc in ['MI', 'STTC', 'CD', 'HYP']:
        s = path_only[path_only.primary_superclass == sc].dropna(subset=['delta_norm', 'max_likelihood'])
        if len(s) > 20:
            rho, p = stats.spearmanr(s.max_likelihood, s.delta_norm)
            txt += f"  {sc:5s}: ρ = {rho:+.3f}, p = {p:.2e}, n = {len(s)}\n"

    rho_all, p_all = stats.spearmanr(path_only.max_likelihood, path_only.delta_norm)
    txt += f"\n  ALL  : ρ = {rho_all:+.3f}, p = {p_all:.2e}, n = {len(path_only)}\n"

    ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, fontsize=11,
             va='top', fontfamily='monospace')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/Fig4_distance_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig4 ✓")

    # ================================================================
    # FIGURE 5: REGIONAL β MAP
    # ================================================================
    regions = {
        'Anterior\n(V1-V4)': ['V1','V2','V3','V4'],
        'Lateral\n(I,aVL,V5,V6)': ['I','aVL','V5','V6'],
        'Inferior\n(II,III,aVF)': ['II','III','aVF'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (rname, leads) in zip(axes, regions.items()):
        cols = [f'beta_ir_{l}' for l in leads]
        vals = {}
        for sc in SUPERCLASSES:
            sub = clean[clean.clean_superclass == sc]
            vals[sc] = sub[cols].median().mean()

        bars = ax.bar(range(5), [vals[s] for s in SUPERCLASSES],
                      color=[SC_COLORS[s] for s in SUPERCLASSES], alpha=0.8,
                      edgecolor='white', linewidth=1)
        ax.axhline(beta_norm, color='green', ls='--', lw=1.5, alpha=0.6)
        ax.set_xticks(range(5))
        ax.set_xticklabels(SUPERCLASSES, fontsize=10)
        ax.set_ylabel('Median β')
        ax.set_title(rname, fontsize=12, fontweight='bold')
        ax.set_ylim(beta_norm - 0.4, beta_norm + 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Figure 5: Regional β by Diagnosis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{RESULTS_DIR}/Fig5_regional_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig5 ✓")

    # ================================================================
    # FIGURE 6: PREDICTIVE VALIDATION
    # ================================================================
    pred = dm[dm.is_clean & dm.beta_mean.notna()].copy()
    pred['is_norm'] = (pred.clean_superclass == 'NORM').astype(int)
    pred['sex_num'] = pred.sex.astype(float)
    pred['delta_norm'] = np.abs(pred.beta_mean - beta_norm)

    train = pred[pred.strat_fold.isin(range(1, 9))]
    test = pred[pred.strat_fold.isin([9, 10])]

    beta_lead_cols = [f'beta_ir_{l}' for l in LEAD_NAMES]
    models = {
        'Age + Sex':              ['age', 'sex_num'],
        '+ β_mean':               ['age', 'sex_num', 'beta_mean'],
        '+ β + δ + σ_β':          ['age', 'sex_num', 'beta_mean', 'delta_norm', 'beta_std'],
        '+ all 12 lead β':        ['age', 'sex_num'] + beta_lead_cols,
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    colors_line = ['#95a5a6', '#e67e22', '#c0392b', '#8e44ad']

    for idx_m, (name, feats) in enumerate(models.items()):
        avail = [f for f in feats if f in train.columns]
        tr = train[avail + ['is_norm']].dropna()
        te = test[avail + ['is_norm']].dropna()
        if len(tr) < 30 or len(te) < 10:
            continue
        sc_obj = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc_obj.fit_transform(tr[avail]), tr.is_norm)
        y_p = clf.predict_proba(sc_obj.transform(te[avail]))[:, 1]
        auc_val = roc_auc_score(te.is_norm, y_p)
        fpr, tpr, _ = roc_curve(te.is_norm, y_p)
        ax.plot(fpr, tpr, lw=2.5, color=colors_line[idx_m],
                label=f'{name}  AUC={auc_val:.3f}')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Figure 6: Predictive Validation\n(NORM vs Pathology)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/Fig6_prediction_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Fig6 ✓")

    # ================================================================
    # SUPPLEMENTARY: STATISTICAL SUMMARY
    # ================================================================
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)

    # Kruskal-Wallis
    groups = [clean[clean.clean_superclass == sc].beta_mean.dropna().values for sc in SUPERCLASSES]
    H, p_kw = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis: H={H:.2f}, p={p_kw:.2e}")

    # Pairwise
    print("\nPairwise comparisons (Bonferroni-corrected):")
    n_comp = 10
    for i in range(5):
        for j in range(i+1, 5):
            d = pg.compute_effsize(groups[i], groups[j], eftype='cohen')
            _, p = stats.mannwhitneyu(groups[i], groups[j])
            p_adj = min(p * n_comp, 1.0)
            sig = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
            print(f"  {SUPERCLASSES[i]:4s} vs {SUPERCLASSES[j]:4s}: "
                  f"Δmedian={np.median(groups[i])-np.median(groups[j]):+.3f}, "
                  f"d={d:+.3f}, p_adj={p_adj:.3e} {sig}")

    # ANCOVA
    print("\nANCOVA (β ~ diagnosis + age + sex):")
    cv = clean.dropna(subset=['beta_mean', 'age', 'sex', 'clean_superclass'])
    if len(cv) > 50:
        res = pg.ancova(data=cv, dv='beta_mean', between='clean_superclass', covar=['age', 'sex'])
        print(res.to_string())

    # Cross-lead dispersion
    print("\nCross-lead σ_β by diagnosis:")
    for sc in SUPERCLASSES:
        v = clean[clean.clean_superclass == sc].beta_std
        print(f"  {sc}: {v.median():.3f} [{v.quantile(0.25):.3f}–{v.quantile(0.75):.3f}]")

    sig_disp, p_disp = stats.kruskal(*[clean[clean.clean_superclass == sc].beta_std.dropna().values
                                        for sc in SUPERCLASSES])
    print(f"  Kruskal-Wallis: H={sig_disp:.2f}, p={p_disp:.2e}")


if __name__ == '__main__':
    dm, scp = load_data()
    generate_all_figures(dm, scp)
    print("\n✓ All final figures generated.")
