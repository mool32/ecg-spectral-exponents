#!/usr/bin/env python3
"""
Gap Analysis: Blocks A, B, C — Ion channels, Composition, Coupled subsystems
=============================================================================
Builds on existing TMS data (results/tms_cardiomyocytes.h5ad, tms_fibroblasts.h5ad).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.stats import kurtosis, skew, mannwhitneyu, gaussian_kde
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

AGE_ORDER = ['3-month-old stage', '18-month-old stage', '20-month-old stage and over']
AGE_SHORT = {'3-month-old stage': '3m', '18-month-old stage': '18m',
             '20-month-old stage and over': '20+m'}
AGE_LABELS = {'3-month-old stage': '3 mo (Young)',
              '18-month-old stage': '18 mo (Middle)',
              '20-month-old stage and over': '20+ mo (Old)'}
AGE_COLORS = {'3-month-old stage': '#2196F3',
              '18-month-old stage': '#FF9800',
              '20-month-old stage and over': '#E53935'}

# Gene categories
COUPLING = ['Gja1', 'Gja5', 'Gjc1']
CHANNELS = ['Scn5a', 'Kcnj2', 'Cacna1c', 'Kcnq1', 'Hcn4']
STRUCTURAL = ['Dsp', 'Jup', 'Pkp2']
CONTRACTILE = ['Ryr2', 'Atp2a2']
FIBROSIS = ['Col1a1', 'Col3a1', 'Acta2', 'Postn']
HOUSEKEEPING = ['Gapdh', 'Actb']

ALL_GENES = COUPLING + CHANNELS + STRUCTURAL + CONTRACTILE + FIBROSIS + HOUSEKEEPING
CATEGORY_MAP = {}
for g in COUPLING: CATEGORY_MAP[g] = 'Gap Junction'
for g in CHANNELS: CATEGORY_MAP[g] = 'Ion Channel'
for g in STRUCTURAL: CATEGORY_MAP[g] = 'Structural'
for g in CONTRACTILE: CATEGORY_MAP[g] = 'Contractile'
for g in FIBROSIS: CATEGORY_MAP[g] = 'Fibrosis'
for g in HOUSEKEEPING: CATEGORY_MAP[g] = 'Housekeeping'

# ECG mapping for each gene
ECG_LINK = {
    'Gja1': 'Inter-cell conduction speed → QRS width, β',
    'Gja5': 'Conduction system coupling → PR interval',
    'Scn5a': 'Depolarization speed → QRS shape, β',
    'Kcnj2': 'Resting potential → Arrhythmogenicity',
    'Cacna1c': 'AP plateau → ST-T morphology (STTC)',
    'Kcnq1': 'Repolarization → QT duration',
    'Hcn4': 'Pacemaker current → Heart rate',
}


def get_expr(adata, gene, gmap):
    if gene not in gmap:
        return None
    idx = gmap[gene]
    col = adata[:, idx].X
    if hasattr(col, 'toarray'):
        col = col.toarray()
    return col.flatten()


def bimod_coeff(x):
    n = len(x)
    if n < 10:
        return np.nan
    s = skew(x)
    k = kurtosis(x, fisher=True)
    return (s**2 + 1) / (k + 3)


def shannon_entropy(x, bins=30):
    c, _ = np.histogram(x, bins=bins)
    p = c / c.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


# ═══════════════════════════════════════════════════════════════════
# BLOCK A: Ion Channel Aging
# ═══════════════════════════════════════════════════════════════════

def block_A(cm, cm_atrial, gene_map):
    """Comprehensive ion channel + gap junction aging analysis."""
    print("\n" + "="*70)
    print("BLOCK A: Ion Channel & Gap Junction Aging Profile")
    print("="*70)

    # Compute metrics for ALL genes, in atrial-only subset
    results = []
    for gene in ALL_GENES:
        for age in AGE_ORDER:
            mask = cm_atrial.obs['development_stage'] == age
            if mask.sum() == 0:
                continue
            expr = get_expr(cm_atrial[mask], gene, gene_map)
            if expr is None:
                continue
            n = len(expr)
            n_nz = np.sum(expr > 0)
            frac_det = n_nz / n
            ep = expr[expr > 0]

            if len(ep) < 5:
                results.append({
                    'gene': gene, 'category': CATEGORY_MAP.get(gene, ''),
                    'age': age, 'n_cells': n, 'n_nonzero': n_nz,
                    'frac_detected': frac_det,
                    'mean': np.nan, 'median': np.nan, 'cv': np.nan,
                    'fano': np.nan, 'bc': np.nan, 'entropy': np.nan
                })
                continue

            mu = np.mean(ep)
            results.append({
                'gene': gene, 'category': CATEGORY_MAP.get(gene, ''),
                'age': age, 'n_cells': n, 'n_nonzero': n_nz,
                'frac_detected': frac_det,
                'mean': mu, 'median': np.median(ep),
                'cv': np.std(ep)/mu if mu > 0 else np.nan,
                'fano': np.var(ep)/mu if mu > 0 else np.nan,
                'bc': bimod_coeff(ep),
                'entropy': shannon_entropy(ep),
            })

    df = pd.DataFrame(results)
    df.to_csv('results/block_A_ion_channel_metrics.csv', index=False)

    # ── Compute fold changes (young vs old) ──
    fc_data = []
    for gene in ALL_GENES:
        gdf = df[df.gene == gene]
        y = gdf[gdf.age == AGE_ORDER[0]]
        o = gdf[gdf.age == AGE_ORDER[-1]]
        if len(y) == 0 or len(o) == 0:
            continue

        yv = y.iloc[0]
        ov = o.iloc[0]

        # Fraction-low: fraction with expr < young P25
        young_mask = cm_atrial.obs['development_stage'] == AGE_ORDER[0]
        young_expr = get_expr(cm_atrial[young_mask], gene, gene_map)
        if young_expr is not None:
            yp = young_expr[young_expr > 0]
            thresh = np.percentile(yp, 25) if len(yp) > 5 else 0
        else:
            thresh = 0

        frac_low_young = np.nan
        frac_low_old = np.nan
        for age, label in [(AGE_ORDER[0], 'young'), (AGE_ORDER[-1], 'old')]:
            mask = cm_atrial.obs['development_stage'] == age
            expr = get_expr(cm_atrial[mask], gene, gene_map)
            if expr is not None:
                fl = np.sum(expr < thresh) / len(expr)
                if label == 'young':
                    frac_low_young = fl
                else:
                    frac_low_old = fl

        # MW test
        young_all = get_expr(cm_atrial[cm_atrial.obs['development_stage'] == AGE_ORDER[0]],
                             gene, gene_map)
        old_all = get_expr(cm_atrial[cm_atrial.obs['development_stage'] == AGE_ORDER[-1]],
                           gene, gene_map)
        if young_all is not None and old_all is not None:
            U, p_mw = mannwhitneyu(young_all, old_all, alternative='two-sided')
            n1, n2 = len(young_all), len(old_all)
            r_rb = 1 - 2*U/(n1*n2)
        else:
            p_mw, r_rb = np.nan, np.nan

        fc_data.append({
            'gene': gene,
            'category': CATEGORY_MAP.get(gene, ''),
            'ecg_link': ECG_LINK.get(gene, ''),
            'mean_young': yv['mean'], 'mean_old': ov['mean'],
            'mean_fc': ov['mean']/yv['mean'] if yv['mean'] > 0 else np.nan,
            'cv_young': yv['cv'], 'cv_old': ov['cv'],
            'cv_fc': ov['cv']/yv['cv'] if yv['cv'] > 0 else np.nan,
            'fano_young': yv['fano'], 'fano_old': ov['fano'],
            'fano_fc': ov['fano']/yv['fano'] if yv['fano'] > 0 else np.nan,
            'det_young': yv['frac_detected'], 'det_old': ov['frac_detected'],
            'det_change': ov['frac_detected'] - yv['frac_detected'],
            'frac_low_young': frac_low_young, 'frac_low_old': frac_low_old,
            'MW_p': p_mw, 'r_rb': r_rb,
        })

    fc_df = pd.DataFrame(fc_data)
    fc_df.to_csv('results/block_A_fold_changes.csv', index=False)

    # ── Print summary ──
    print("\nAtrial-only cardiomyocytes: young vs old (20+m)")
    print(f"{'Gene':12s} {'Category':12s} {'CV_y':>6s} {'CV_o':>6s} {'CV_FC':>6s} "
          f"{'Det_y':>6s} {'Det_o':>6s} {'MW_p':>10s} {'r_rb':>6s}")
    print("-"*80)
    for _, r in fc_df.iterrows():
        sig = '***' if r['MW_p'] < 0.001 else '**' if r['MW_p'] < 0.01 else '*' if r['MW_p'] < 0.05 else 'ns'
        print(f"{r['gene']:12s} {r['category']:12s} {r['cv_young']:6.2f} {r['cv_old']:6.2f} "
              f"{r['cv_fc']:6.2f} {r['det_young']:6.1%} {r['det_old']:6.1%} "
              f"{r['MW_p']:10.1e} {r['r_rb']:+6.3f} {sig}")

    # ── Fig A-1: Heatmap of aging profile ──
    fig_A1(fc_df)

    # ── Fig A-2: CV trajectories by category ──
    fig_A2(df)

    return df, fc_df


def fig_A1(fc_df):
    """Fig A-1: Heatmap — gene × metric fold changes."""
    # Exclude housekeeping for main heatmap
    cardiac_df = fc_df[~fc_df.category.isin(['Housekeeping', 'Fibrosis'])].copy()
    cardiac_df = cardiac_df.set_index('gene')

    metrics = ['mean_fc', 'cv_fc', 'fano_fc', 'det_change']
    labels = ['Mean Expression\n(fold change)', 'CV\n(fold change)',
              'Fano Factor\n(fold change)', 'Detection Rate\n(Δ)']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [3, 1.2]})

    # Panel A: Heatmap
    ax = axes[0]
    data = cardiac_df[metrics].values.copy()
    # Center fold changes at 1.0, det_change at 0
    gene_names = cardiac_df.index.tolist()
    categories = cardiac_df['category'].tolist()

    # Color by divergence from 1 (for FC) and 0 (for det)
    display = np.zeros_like(data)
    for j in range(3):  # FC columns: log2 fold change
        display[:, j] = np.log2(np.clip(data[:, j], 0.1, 10))
    display[:, 3] = data[:, 3]  # det_change as-is

    im = ax.imshow(display, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')

    # Annotate
    for i in range(len(gene_names)):
        for j in range(len(metrics)):
            val = data[i, j]
            if j < 3:
                txt = f'{val:.2f}' if not np.isnan(val) else 'n/a'
            else:
                txt = f'{val:+.1%}' if not np.isnan(val) else 'n/a'
            color = 'white' if abs(display[i, j]) > 0.8 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(gene_names)))
    # Add category color coding
    cat_colors = {'Gap Junction': '#E91E63', 'Ion Channel': '#2196F3',
                  'Structural': '#4CAF50', 'Contractile': '#FF9800'}
    for i, (g, cat) in enumerate(zip(gene_names, categories)):
        ax.get_yticklabels()
    ax.set_yticklabels([f'{g} ({cat[:3]})' for g, cat in zip(gene_names, categories)],
                       fontsize=10)

    # Category color bars
    for i, cat in enumerate(categories):
        ax.add_patch(plt.Rectangle((-0.6, i-0.5), 0.15, 1,
                                   facecolor=cat_colors.get(cat, 'gray'), clip_on=False))

    plt.colorbar(im, ax=ax, shrink=0.7, label='log₂(fold change old/young)')
    ax.set_title('A. Molecular Aging Profile (Atrial Cardiomyocytes)',
                 fontweight='bold', fontsize=13)

    # Panel B: Statistical significance
    ax = axes[1]
    cardiac_df_plot = fc_df[~fc_df.category.isin(['Housekeeping', 'Fibrosis'])].copy()
    genes_order = cardiac_df_plot['gene'].tolist()
    pvals = cardiac_df_plot['MW_p'].values
    log_p = -np.log10(np.clip(pvals, 1e-30, 1))
    colors = ['#E53935' if p < 0.001 else '#FF9800' if p < 0.05 else '#9E9E9E'
              for p in pvals]
    y_pos = np.arange(len(genes_order))
    ax.barh(y_pos, log_p, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5,
               label='p=0.05')
    ax.axvline(x=-np.log10(0.001), color='gray', linestyle=':', alpha=0.5,
               label='p=0.001')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes_order, fontsize=10)
    ax.set_xlabel('-log₁₀(p)')
    ax.set_title('B. Significance', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig_A1_ion_channel_heatmap.png')
    plt.close()
    print("  Saved fig_A1_ion_channel_heatmap.png")


def fig_A2(metrics_df):
    """Fig A-2: CV trajectories by gene category."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    categories = [
        ('Gap Junction', COUPLING, '#E91E63'),
        ('Ion Channel', CHANNELS, '#2196F3'),
        ('Structural', STRUCTURAL, '#4CAF50'),
        ('Contractile', CONTRACTILE, '#FF9800'),
        ('Fibrosis', FIBROSIS, '#795548'),
        ('Housekeeping', HOUSEKEEPING, '#9E9E9E'),
    ]

    x = np.arange(len(AGE_ORDER))
    for ax, (cat_name, genes, color) in zip(axes.flat, categories):
        for g in genes:
            gdf = metrics_df[(metrics_df.gene == g)]
            cvs = []
            for age in AGE_ORDER:
                row = gdf[gdf.age == age]
                cvs.append(row['cv'].values[0] if len(row) > 0 else np.nan)
            ax.plot(x, cvs, 'o-', label=g, linewidth=1.5, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER])
        ax.set_ylabel('CV (σ/μ among detected cells)')
        ax.set_title(f'{cat_name}', fontweight='bold', color=color)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Fig A-2: Cell-to-Cell Variability Trajectories by Gene Category',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/fig_A2_cv_by_category.png')
    plt.close()
    print("  Saved fig_A2_cv_by_category.png")


# ═══════════════════════════════════════════════════════════════════
# BLOCK B: Composition Shift
# ═══════════════════════════════════════════════════════════════════

def block_B(cm_full, fb_full):
    """Cell composition shift across aging."""
    print("\n" + "="*70)
    print("BLOCK B: Cell Composition Shift")
    print("="*70)

    # Load full heart data from composition CSV if available, or compute from cm+fb
    comp_data = []

    # We have cardiomyocytes and fibroblasts separately
    # Count by age and cell_type
    for label, adata in [('Cardiomyocytes', cm_full), ('Fibroblasts', fb_full)]:
        for age in AGE_ORDER:
            mask = adata.obs['development_stage'] == age
            n = mask.sum()
            if n == 0:
                continue
            if label == 'Cardiomyocytes':
                # Subtypes
                sub = adata[mask]
                for ct in sub.obs['cell_type'].unique():
                    n_ct = (sub.obs['cell_type'] == ct).sum()
                    comp_data.append({'age': age, 'cell_type': ct, 'n': n_ct})
            else:
                comp_data.append({'age': age, 'cell_type': 'cardiac fibroblast', 'n': n})

    comp_df = pd.DataFrame(comp_data)

    # Compute proportions within each age
    for age in AGE_ORDER:
        sub = comp_df[comp_df.age == age]
        total = sub.n.sum()
        comp_df.loc[comp_df.age == age, 'fraction'] = sub.n / total
        print(f"\n  {AGE_LABELS[age]}:")
        for _, r in sub.iterrows():
            frac = r['n'] / total
            print(f"    {r['cell_type']:40s}: {r['n']:5d} ({frac:.1%})")

    comp_df.to_csv('results/block_B_composition.csv', index=False)

    # ── Fig B-1: Stacked bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Stacked bar (proportions)
    ax = axes[0]
    cell_types = comp_df.cell_type.unique()
    ct_colors = {
        'regular atrial cardiac myocyte': '#E53935',
        'regular ventricular cardiac myocyte': '#FF7043',
        'cardiac fibroblast': '#42A5F5',
    }
    bottom = np.zeros(len(AGE_ORDER))
    x = np.arange(len(AGE_ORDER))
    for ct in cell_types:
        fracs = []
        for age in AGE_ORDER:
            row = comp_df[(comp_df.age == age) & (comp_df.cell_type == ct)]
            fracs.append(row['fraction'].values[0] if len(row) > 0 else 0)
        ax.bar(x, fracs, bottom=bottom, label=ct.replace('regular ', ''),
               color=ct_colors.get(ct, '#9E9E9E'), alpha=0.85, edgecolor='white')
        bottom += fracs

    ax.set_xticks(x)
    ax.set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER], fontsize=12)
    ax.set_ylabel('Fraction of Cells in Dataset')
    ax.set_title('A. Cell Type Composition (TMS Heart FACS)', fontweight='bold')
    ax.legend(fontsize=9, loc='center right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Absolute counts
    ax = axes[1]
    bottom = np.zeros(len(AGE_ORDER))
    for ct in cell_types:
        counts = []
        for age in AGE_ORDER:
            row = comp_df[(comp_df.age == age) & (comp_df.cell_type == ct)]
            counts.append(row['n'].values[0] if len(row) > 0 else 0)
        ax.bar(x, counts, bottom=bottom, label=ct.replace('regular ', ''),
               color=ct_colors.get(ct, '#9E9E9E'), alpha=0.85, edgecolor='white')
        # Add count labels
        for xi, c in enumerate(counts):
            if c > 0:
                ax.text(xi, bottom[xi] + c/2, str(c), ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER], fontsize=12)
    ax.set_ylabel('Number of Cells')
    ax.set_title('B. Absolute Cell Counts', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Fig B-1: Heart Cell Composition Shift with Age (TMS)',
                 fontsize=14, fontweight='bold', y=1.01)

    # Add caveat box
    ax.text(0.02, 0.98, '⚠ Proportions reflect dataset sampling,\n'
            'not true tissue composition.\nFACS enrichment varies by age.',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/fig_B1_composition.png')
    plt.close()
    print("  Saved fig_B1_composition.png")

    return comp_df


# ═══════════════════════════════════════════════════════════════════
# BLOCK C: Coupled Subsystems
# ═══════════════════════════════════════════════════════════════════

def block_C(cm_atrial, gene_map):
    """Analyze working myocardium vs conduction system markers."""
    print("\n" + "="*70)
    print("BLOCK C: Coupled Subsystems")
    print("="*70)

    # Check HCN4 and GJA5 expression
    for gene in ['Hcn4', 'Gja5', 'Gja1']:
        print(f"\n  {gene} expression across ages (atrial only):")
        for age in AGE_ORDER:
            mask = cm_atrial.obs['development_stage'] == age
            expr = get_expr(cm_atrial[mask], gene, gene_map)
            if expr is not None:
                n_det = np.sum(expr > 0)
                n_tot = len(expr)
                mean_det = np.mean(expr[expr > 0]) if n_det > 0 else 0
                print(f"    {AGE_SHORT[age]}: detected {n_det}/{n_tot} "
                      f"({n_det/n_tot:.1%}), mean(det)={mean_det:.1f}")

    # Try to separate working vs conduction-like cells
    # Working: Gja1-high, Hcn4-low
    # Conduction-like: Hcn4 > 0 or Gja5 > 0

    gja1_expr = get_expr(cm_atrial, 'Gja1', gene_map)
    hcn4_expr = get_expr(cm_atrial, 'Hcn4', gene_map)
    gja5_expr = get_expr(cm_atrial, 'Gja5', gene_map)

    if gja1_expr is None or hcn4_expr is None:
        print("\n  Cannot separate subsystems — missing gene data")
        return None

    n_hcn4 = np.sum(hcn4_expr > 0)
    n_gja5 = np.sum(gja5_expr > 0)
    n_both = np.sum((hcn4_expr > 0) | (gja5_expr > 0))
    print(f"\n  Conduction markers in atrial cardiomyocytes:")
    print(f"    HCN4+: {n_hcn4}/{len(hcn4_expr)} ({n_hcn4/len(hcn4_expr):.1%})")
    print(f"    GJA5+: {n_gja5}/{len(gja5_expr)} ({n_gja5/len(gja5_expr):.1%})")
    print(f"    HCN4+ or GJA5+: {n_both}/{len(hcn4_expr)}")

    if n_both < 20:
        print("\n  ⚠ Too few conduction-like cells for reliable analysis.")
        print("  Recording as limitation — need targeted scRNA-seq of conduction system.")

    # Even with small N, compute aging trajectories for both subsystems
    # Working myocardium: GJA1 > median
    gja1_med = np.median(gja1_expr[gja1_expr > 0])
    working_mask = gja1_expr > gja1_med

    # Conduction-like: HCN4 > 0 or GJA5 > 0
    conduction_mask = (hcn4_expr > 0) | (gja5_expr > 0)

    print(f"\n  Working myocardium (GJA1 > median): {working_mask.sum()} cells")
    print(f"  Conduction-like (HCN4+ or GJA5+): {conduction_mask.sum()} cells")

    # Compare aging for GJA1 in both populations
    results = []
    for pop_name, pop_mask in [('Working Myocardium', working_mask),
                                ('Conduction-like', conduction_mask)]:
        for age in AGE_ORDER:
            age_mask = cm_atrial.obs['development_stage'].values == age
            combined = age_mask & pop_mask
            n_total = combined.sum()
            if n_total < 5:
                results.append({'population': pop_name, 'age': age,
                                'n': n_total, 'gja1_mean': np.nan, 'gja1_cv': np.nan})
                continue
            expr = gja1_expr[combined]
            ep = expr[expr > 0]
            mu = np.mean(ep) if len(ep) > 0 else np.nan
            cv = np.std(ep)/mu if len(ep) > 5 and mu > 0 else np.nan
            results.append({'population': pop_name, 'age': age,
                            'n': n_total, 'gja1_mean': mu, 'gja1_cv': cv})
            print(f"  {pop_name:25s} {AGE_SHORT[age]}: n={n_total:4d}, "
                  f"GJA1 mean={mu:.0f}, CV={cv:.3f}" if not np.isnan(cv)
                  else f"  {pop_name:25s} {AGE_SHORT[age]}: n={n_total:4d}, insufficient")

    res_df = pd.DataFrame(results)
    res_df.to_csv('results/block_C_subsystems.csv', index=False)

    # ── Fig C-1: Dual aging trajectories ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(AGE_ORDER))
    for pop in ['Working Myocardium', 'Conduction-like']:
        pdf = res_df[res_df.population == pop]
        means = [pdf[pdf.age == a]['gja1_mean'].values[0]
                 if len(pdf[pdf.age == a]) > 0 else np.nan for a in AGE_ORDER]
        cvs = [pdf[pdf.age == a]['gja1_cv'].values[0]
               if len(pdf[pdf.age == a]) > 0 else np.nan for a in AGE_ORDER]
        ns = [pdf[pdf.age == a]['n'].values[0]
              if len(pdf[pdf.age == a]) > 0 else 0 for a in AGE_ORDER]

        style = 'o-' if pop == 'Working Myocardium' else 's--'
        color = '#E53935' if pop == 'Working Myocardium' else '#1565C0'
        lbl = f'{pop} (n={ns})'

        axes[0].plot(x, means, style, color=color, label=pop, linewidth=2, markersize=8)
        axes[1].plot(x, cvs, style, color=color, label=pop, linewidth=2, markersize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER])
    axes[0].set_ylabel('Mean GJA1 Expression (detected cells)')
    axes[0].set_title('A. GJA1 Mean by Subsystem', fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER])
    axes[1].set_ylabel('GJA1 CV')
    axes[1].set_title('B. GJA1 Variability by Subsystem', fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.suptitle('Fig C-1: Dual Subsystem Aging — Working Myocardium vs Conduction',
                 fontsize=14, fontweight='bold', y=1.01)

    # Caveat
    axes[1].text(0.02, 0.98,
                 f'⚠ Conduction-like defined as\n'
                 f'HCN4+ or GJA5+ ({conduction_mask.sum()} cells)\n'
                 f'Small N — interpret cautiously',
                 transform=axes[1].transAxes, fontsize=8, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/fig_C1_subsystems.png')
    plt.close()
    print("  Saved fig_C1_subsystems.png")

    return res_df


# ═══════════════════════════════════════════════════════════════════
# BLOCK E: Bifurcation Diagram
# ═══════════════════════════════════════════════════════════════════

def block_E():
    """Create 'Two Roads Diverged' bifurcation diagram."""
    print("\n" + "="*70)
    print("BLOCK E: Bifurcation Diagram")
    print("="*70)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')

    # Title
    ax.text(5, 8.2, 'Two Roads Diverged: Aging vs Pathology',
            ha='center', fontsize=16, fontweight='bold')

    # ── Central node: healthy heart ──
    from matplotlib.patches import FancyBboxPatch
    healthy_box = FancyBboxPatch((3.5, 5.5), 3, 0.9, boxstyle="round,pad=0.15",
                                  facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax.add_patch(healthy_box)
    ax.text(5, 6.0, 'Healthy Heart\nβ ≈ 1.76', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')

    # ── Left branch: Aging (β decreases) ──
    # Arrow down-left
    ax.annotate('', xy=(2.5, 4.3), xytext=(4.2, 5.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.5))
    ax.text(2.8, 4.9, 'AGING', fontsize=12, fontweight='bold', color='#1565C0',
            rotation=30)

    # Aging box
    aging_box = FancyBboxPatch((0.5, 3.0), 4, 1.2, boxstyle="round,pad=0.15",
                                facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(aging_box)
    ax.text(2.5, 3.85, 'β ↓  (< 1.76)', ha='center', fontsize=12,
            fontweight='bold', color='#1565C0')
    ax.text(2.5, 3.45, 'Diffuse, fine-grained degradation', ha='center',
            fontsize=9, color='#333')
    ax.text(2.5, 3.15, 'High-frequency fragmentation', ha='center',
            fontsize=9, color='#333')

    # Molecular mechanism boxes (aging)
    mechanisms_aging = [
        (0.2, 1.8, 'GJA1 ↓\nGap junction loss\nCV +89%', '#E3F2FD'),
        (2.0, 1.8, 'SCN5A ↓\nIon remodeling\nDepolz. slows', '#E8EAF6'),
        (3.8, 1.8, 'Fibrosis ↑\nCollagen\ndeposition', '#FFF3E0'),
    ]
    for x, y, txt, fc in mechanisms_aging:
        box = FancyBboxPatch((x, y), 1.6, 1.0, boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor='#666', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.8, y + 0.5, txt, ha='center', va='center', fontsize=8)
        ax.annotate('', xy=(x + 0.8, y + 1.0), xytext=(x + 0.8, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1))

    # ECG manifestation
    ax.text(2.5, 0.8, 'ECG: σ_β ↑, HF power ↑,\nInter-lead correlation ↓',
            ha='center', fontsize=10, style='italic', color='#1565C0',
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.5))

    # ── Right branch: Pathology (β increases) ──
    ax.annotate('', xy=(7.5, 4.3), xytext=(5.8, 5.5),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5))
    ax.text(6.8, 4.9, 'PATHOLOGY', fontsize=12, fontweight='bold', color='#C62828',
            rotation=-30)

    # Pathology box
    path_box = FancyBboxPatch((5.5, 3.0), 4.5, 1.2, boxstyle="round,pad=0.15",
                               facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(path_box)
    ax.text(7.75, 3.85, 'β ↑  (> 1.76)', ha='center', fontsize=12,
            fontweight='bold', color='#C62828')
    ax.text(7.75, 3.45, 'Focal, large-scale disruption', ha='center',
            fontsize=9, color='#333')
    ax.text(7.75, 3.15, 'Smooth, slow signal (loss of complexity)', ha='center',
            fontsize=9, color='#333')

    # Molecular mechanisms (pathology)
    mechanisms_path = [
        (5.6, 1.8, 'GJA1 lateral.\nCx43 redistrib.\n→ slow conduct.', '#FFEBEE'),
        (7.4, 1.8, 'Bundle branch\nblock\n→ CD (β ≈ 2.03)', '#FCE4EC'),
        (9.2, 1.8, 'Hypertrophy\nThick walls\n→ HYP', '#FFF3E0'),
    ]
    for x, y, txt, fc in mechanisms_path:
        box = FancyBboxPatch((x, y), 1.6, 1.0, boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor='#666', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.8, y + 0.5, txt, ha='center', va='center', fontsize=8)
        ax.annotate('', xy=(x + 0.8, y + 1.0), xytext=(x + 0.8, 3.0),
                    arrowprops=dict(arrowstyle='->', color='#999', lw=1))

    # ECG manifestation
    ax.text(7.75, 0.8, 'ECG: β → 2.0+, QRS wide,\nCoherence paradoxically ↑',
            ha='center', fontsize=10, style='italic', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.5))

    # ── Bottom: key insight ──
    ax.text(5, 0.0, 'Same parameter (β), opposite directions, different molecular mechanisms',
            ha='center', fontsize=11, fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='#FFC107', linewidth=2))

    plt.tight_layout()
    plt.savefig('results/fig_E1_bifurcation.png')
    plt.close()
    print("  Saved fig_E1_bifurcation.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("GAP ANALYSIS — Blocks A, B, C, E")
    print("="*70)

    # Load
    cm = sc.read_h5ad('results/tms_cardiomyocytes.h5ad')
    fb = sc.read_h5ad('results/tms_fibroblasts.h5ad')
    gene_map = dict(zip(cm.var['feature_name'], cm.var.index))

    # Filter ages
    cm = cm[cm.obs['development_stage'].isin(AGE_ORDER)].copy()
    fb = fb[fb.obs['development_stage'].isin(AGE_ORDER)].copy()

    # Atrial-only subset
    cm_atrial = cm[cm.obs['cell_type'] == 'regular atrial cardiac myocyte'].copy()
    print(f"Loaded: {cm.shape[0]} cardiomyocytes ({cm_atrial.shape[0]} atrial), "
          f"{fb.shape[0]} fibroblasts")

    # Block A
    metrics_df, fc_df = block_A(cm, cm_atrial, gene_map)

    # Block B
    comp_df = block_B(cm, fb)

    # Block C
    subsys_df = block_C(cm_atrial, gene_map)

    # Block E
    block_E()

    print("\n" + "="*70)
    print("ALL BLOCKS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  results/block_A_ion_channel_metrics.csv")
    print("  results/block_A_fold_changes.csv")
    print("  results/fig_A1_ion_channel_heatmap.png")
    print("  results/fig_A2_cv_by_category.png")
    print("  results/block_B_composition.csv")
    print("  results/fig_B1_composition.png")
    print("  results/block_C_subsystems.csv")
    print("  results/fig_C1_subsystems.png")
    print("  results/fig_E1_bifurcation.png")


if __name__ == '__main__':
    main()
