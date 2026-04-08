#!/usr/bin/env python3
"""
Part II: Molecular Substrate — Tabula Muris Senis scRNA-seq Analysis
=====================================================================
Analyzes cell-to-cell variability of key cardiac genes across aging
in mouse cardiomyocytes from the Tabula Muris Senis dataset.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ── Color palette ────────────────────────────────────────────────────
AGE_COLORS = {
    '3-month-old stage': '#2196F3',        # young — blue
    '18-month-old stage': '#FF9800',       # middle — orange
    '20-month-old stage and over': '#E53935'  # old — red
}
AGE_LABELS = {
    '3-month-old stage': '3 mo (Young)',
    '18-month-old stage': '18 mo (Middle)',
    '20-month-old stage and over': '20+ mo (Old)'
}
AGE_ORDER = ['3-month-old stage', '18-month-old stage', '20-month-old stage and over']

# Human-equivalent ages for cross-scale figure
AGE_HUMAN_EQUIV = {
    '3-month-old stage': 25,
    '18-month-old stage': 55,
    '20-month-old stage and over': 70
}

# ── Target genes ─────────────────────────────────────────────────────
COUPLING_GENES = ['Gja1', 'Gja5', 'Gjc1']
CHANNEL_GENES = ['Scn5a', 'Kcnj2', 'Cacna1c']
STRUCTURAL_GENES = ['Dsp', 'Jup', 'Pkp2']
FIBROSIS_GENES = ['Col1a1', 'Col3a1', 'Acta2', 'Postn']
HOUSEKEEPING_GENES = ['Gapdh', 'Actb']
CONTRACTILE_GENES = ['Ryr2', 'Atp2a2']

CORE_GENES = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp']
ALL_TARGET_GENES = COUPLING_GENES + CHANNEL_GENES + STRUCTURAL_GENES + CONTRACTILE_GENES


def load_data():
    """Load TMS cardiomyocytes and fibroblasts."""
    print("Loading cardiomyocytes...")
    cm = sc.read_h5ad('results/tms_cardiomyocytes.h5ad')
    print(f"  Shape: {cm.shape}")

    print("Loading fibroblasts...")
    fb = sc.read_h5ad('results/tms_fibroblasts.h5ad')
    print(f"  Shape: {fb.shape}")

    # Build gene name -> index mapping
    gene_map = dict(zip(cm.var['feature_name'], cm.var.index))

    # Filter to only cells in our 3 age groups
    cm = cm[cm.obs['development_stage'].isin(AGE_ORDER)].copy()
    fb = fb[fb.obs['development_stage'].isin(AGE_ORDER)].copy()
    print(f"  Cardiomyocytes after age filter: {cm.shape[0]}")
    print(f"  Fibroblasts after age filter: {fb.shape[0]}")

    return cm, fb, gene_map


def get_gene_expr(adata, gene_name, gene_map):
    """Extract expression vector for a gene by name."""
    if gene_name not in gene_map:
        return None
    idx = gene_map[gene_name]
    # Handle sparse matrices
    col = adata[:, idx].X
    if hasattr(col, 'toarray'):
        col = col.toarray()
    return col.flatten()


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient: (skew^2 + 1) / (kurtosis + 3 * (n-1)^2/((n-2)(n-3)))."""
    x = x[x > 0]  # only non-zero for expression
    n = len(x)
    if n < 10:
        return np.nan
    s = skew(x)
    k = kurtosis(x, fisher=True)  # excess kurtosis
    bc = (s**2 + 1) / (k + 3)
    return bc


def compute_gene_metrics(adata, gene_name, gene_map):
    """Compute per-age-group metrics for a single gene."""
    results = []
    for age in AGE_ORDER:
        mask = adata.obs['development_stage'] == age
        expr = get_gene_expr(adata[mask], gene_name, gene_map)
        if expr is None:
            continue

        n_cells = len(expr)
        n_nonzero = np.sum(expr > 0)
        frac_detected = n_nonzero / n_cells

        # Use only detected cells for variability metrics
        expr_pos = expr[expr > 0]

        if len(expr_pos) < 5:
            results.append({
                'gene': gene_name, 'age': age,
                'mean': np.nan, 'std': np.nan, 'cv': np.nan,
                'fano': np.nan, 'bc': np.nan, 'entropy': np.nan,
                'frac_detected': frac_detected, 'n_cells': n_cells,
                'n_nonzero': n_nonzero
            })
            continue

        mean_val = np.mean(expr_pos)
        std_val = np.std(expr_pos)
        cv = std_val / mean_val if mean_val > 0 else np.nan
        fano = np.var(expr_pos) / mean_val if mean_val > 0 else np.nan
        bc = bimodality_coefficient(expr_pos)

        # Shannon entropy of binned distribution
        counts, _ = np.histogram(expr_pos, bins=30)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        results.append({
            'gene': gene_name, 'age': age,
            'mean': mean_val, 'std': std_val, 'cv': cv,
            'fano': fano, 'bc': bc, 'entropy': entropy,
            'frac_detected': frac_detected, 'n_cells': n_cells,
            'n_nonzero': n_nonzero
        })

    return pd.DataFrame(results)


def compute_all_metrics(cm, gene_map):
    """Compute metrics for all target genes + housekeeping controls."""
    all_genes = ALL_TARGET_GENES + HOUSEKEEPING_GENES
    dfs = []
    for g in all_genes:
        df = compute_gene_metrics(cm, g, gene_map)
        dfs.append(df)
        det = df['frac_detected'].values
        cv = df['cv'].values
        print(f"  {g:12s}: detected={det}, CV={np.round(cv, 3)}")
    return pd.concat(dfs, ignore_index=True)


def compute_gja1_low_fraction(cm, gene_map):
    """Fraction of cardiomyocytes with GJA1 below young 25th percentile."""
    # Threshold from young (3-month)
    young_mask = cm.obs['development_stage'] == '3-month-old stage'
    young_expr = get_gene_expr(cm[young_mask], 'Gja1', gene_map)
    young_pos = young_expr[young_expr > 0]
    threshold = np.percentile(young_pos, 25)
    print(f"\n  GJA1-low threshold (young P25): {threshold:.2f}")

    results = []
    for age in AGE_ORDER:
        mask = cm.obs['development_stage'] == age
        expr = get_gene_expr(cm[mask], 'Gja1', gene_map)
        n_total = len(expr)
        n_low = np.sum(expr < threshold)  # includes zeros
        frac_low = n_low / n_total
        results.append({'age': age, 'n_total': n_total, 'n_low': n_low,
                        'frac_gja1_low': frac_low})
        print(f"  {AGE_LABELS[age]:20s}: {frac_low:.3f} ({n_low}/{n_total})")
    return pd.DataFrame(results)


def compute_gene_correlations(cm, gene_map, genes=None):
    """Gene-gene correlation matrix among cardiomyocytes, per age group."""
    if genes is None:
        genes = CORE_GENES
    corr_by_age = {}
    for age in AGE_ORDER:
        mask = cm.obs['development_stage'] == age
        sub = cm[mask]
        expr_matrix = []
        valid_genes = []
        for g in genes:
            e = get_gene_expr(sub, g, gene_map)
            if e is not None:
                expr_matrix.append(e)
                valid_genes.append(g)
        if len(expr_matrix) < 2:
            continue
        expr_df = pd.DataFrame(np.array(expr_matrix).T, columns=valid_genes)
        corr = expr_df.corr(method='spearman')
        corr_by_age[age] = corr
        # Mean off-diagonal
        mask_tri = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        mean_offdiag = corr.values[mask_tri].mean()
        print(f"  {AGE_LABELS[age]:20s}: mean off-diagonal r = {mean_offdiag:.3f}")
    return corr_by_age


def compute_fibroblast_ratio(cm, fb):
    """Compute fibroblast / cardiomyocyte ratio by age."""
    results = []
    for age in AGE_ORDER:
        n_cm = (cm.obs['development_stage'] == age).sum()
        n_fb = (fb.obs['development_stage'] == age).sum()
        ratio = n_fb / n_cm if n_cm > 0 else np.nan
        results.append({
            'age': age, 'n_cardiomyocytes': n_cm,
            'n_fibroblasts': n_fb, 'fb_cm_ratio': ratio
        })
        print(f"  {AGE_LABELS[age]:20s}: FB/CM = {ratio:.2f} ({n_fb}/{n_cm})")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════

def fig_II_1_violin_gja1(cm, gene_map, metrics_df):
    """Fig II-1: Violin plots of GJA1 expression by age + CV/Fano summary."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [3, 1, 1]})

    # Panel A: Violin plot of GJA1 expression
    ax = axes[0]
    data_by_age = []
    positions = []
    colors = []
    for i, age in enumerate(AGE_ORDER):
        mask = cm.obs['development_stage'] == age
        expr = get_gene_expr(cm[mask], 'Gja1', gene_map)
        expr_pos = expr[expr > 0]
        data_by_age.append(expr_pos)
        positions.append(i)
        colors.append(AGE_COLORS[age])

    parts = ax.violinplot(data_by_age, positions=positions, showmedians=True,
                          showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')

    # Overlay individual points (subsample for clarity)
    for i, (data, pos) in enumerate(zip(data_by_age, positions)):
        jitter = np.random.normal(0, 0.05, len(data))
        sample_idx = np.random.choice(len(data), min(200, len(data)), replace=False)
        ax.scatter(pos + jitter[sample_idx], data[sample_idx],
                   c=colors[i], alpha=0.15, s=8, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels([AGE_LABELS[a] for a in AGE_ORDER], fontsize=10)
    ax.set_ylabel('Gja1 (Connexin-43) expression', fontsize=12)
    ax.set_title('A. GJA1 Expression Across Aging', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: CV by age for key genes
    ax = axes[1]
    key_genes = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Gapdh']
    x_pos = np.arange(len(AGE_ORDER))
    for g in key_genes:
        gdf = metrics_df[metrics_df.gene == g]
        cvs = [gdf[gdf.age == a]['cv'].values[0] if len(gdf[gdf.age == a]) > 0 else np.nan
               for a in AGE_ORDER]
        style = '--' if g in HOUSEKEEPING_GENES else '-'
        lw = 1.5 if g == 'Gja1' else 1.0
        alpha = 0.5 if g in HOUSEKEEPING_GENES else 1.0
        ax.plot(x_pos, cvs, style, marker='o', label=g, linewidth=lw, alpha=alpha, markersize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['3m', '18m', '20+m'], fontsize=10)
    ax.set_ylabel('CV (Coefficient of Variation)')
    ax.set_title('B. CV Across Aging', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # Panel C: Fano factor by age
    ax = axes[2]
    for g in key_genes:
        gdf = metrics_df[metrics_df.gene == g]
        fanos = [gdf[gdf.age == a]['fano'].values[0] if len(gdf[gdf.age == a]) > 0 else np.nan
                 for a in AGE_ORDER]
        style = '--' if g in HOUSEKEEPING_GENES else '-'
        lw = 1.5 if g == 'Gja1' else 1.0
        alpha = 0.5 if g in HOUSEKEEPING_GENES else 1.0
        ax.plot(x_pos, fanos, style, marker='o', label=g, linewidth=lw, alpha=alpha, markersize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['3m', '18m', '20+m'], fontsize=10)
    ax.set_ylabel('Fano Factor (Var/Mean)')
    ax.set_title('C. Fano Factor Across Aging', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig_II_1_gja1_variability.png')
    plt.close()
    print("  Saved fig_II_1_gja1_variability.png")


def fig_II_2_bimodality(cm, gene_map):
    """Fig II-2: Density plots showing emergence of two populations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    genes_to_plot = ['Gja1', 'Scn5a', 'Cacna1c']
    titles = ['A. GJA1 (Connexin-43)', 'B. SCN5A (Nav1.5)', 'C. CACNA1C (Cav1.2)']

    for ax, gene, title in zip(axes, genes_to_plot, titles):
        for age in AGE_ORDER:
            mask = cm.obs['development_stage'] == age
            expr = get_gene_expr(cm[mask], gene, gene_map)
            expr_pos = expr[expr > 0]
            if len(expr_pos) < 10:
                continue
            # Log transform for better visualization
            expr_log = np.log1p(expr_pos)
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(expr_log, bw_method=0.3)
            x_grid = np.linspace(expr_log.min(), expr_log.max(), 200)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.3,
                            color=AGE_COLORS[age], label=AGE_LABELS[age])
            ax.plot(x_grid, kde(x_grid), color=AGE_COLORS[age], linewidth=1.5)

        ax.set_xlabel(f'log1p({gene} expression)')
        ax.set_ylabel('Density')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('Distribution Shape: Do Two Populations Emerge with Age?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig_II_2_bimodality.png')
    plt.close()
    print("  Saved fig_II_2_bimodality.png")


def fig_II_3_correlation_network(corr_by_age):
    """Fig II-3: Gene-gene correlation heatmaps young vs old."""
    ages_to_show = [AGE_ORDER[0], AGE_ORDER[-1]]  # Young vs Old
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             gridspec_kw={'width_ratios': [1, 1, 1]})

    vmin, vmax = -0.3, 0.8
    for i, age in enumerate(ages_to_show):
        ax = axes[i]
        corr = corr_by_age[age]
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=10)
        # Annotate values
        for ii in range(len(corr)):
            for jj in range(len(corr)):
                val = corr.values[ii, jj]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color)
        label = 'A' if i == 0 else 'B'
        ax.set_title(f'{label}. {AGE_LABELS[age]}', fontsize=13, fontweight='bold')

    # Panel C: Difference (Old - Young)
    ax = axes[2]
    diff = corr_by_age[AGE_ORDER[-1]] - corr_by_age[AGE_ORDER[0]]
    im = ax.imshow(diff.values, cmap='PiYG_r', vmin=-0.5, vmax=0.5, aspect='equal')
    ax.set_xticks(range(len(diff.columns)))
    ax.set_xticklabels(diff.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(diff.index)))
    ax.set_yticklabels(diff.index, fontsize=10)
    for ii in range(len(diff)):
        for jj in range(len(diff)):
            val = diff.values[ii, jj]
            color = 'white' if abs(val) > 0.3 else 'black'
            ax.text(jj, ii, f'{val:+.2f}', ha='center', va='center',
                    fontsize=8, color=color)
    ax.set_title('C. Change (Old - Young)', fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=axes[2], shrink=0.8, label='Spearman r change')
    plt.suptitle('Gene Coordination Network: Young vs Old Cardiomyocytes',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig_II_3_gene_correlation.png')
    plt.close()
    print("  Saved fig_II_3_gene_correlation.png")


def fig_II_4_fibroblast_expansion(fb_ratio, gja1_low):
    """Fig II-4: Fibroblast expansion + GJA1-low fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: FB/CM ratio by age
    ax = axes[0]
    x_pos = np.arange(len(AGE_ORDER))
    bars = ax.bar(x_pos, fb_ratio['fb_cm_ratio'].values,
                  color=[AGE_COLORS[a] for a in AGE_ORDER], alpha=0.8, edgecolor='black')
    # Add count labels
    for i, (bar, row) in enumerate(zip(bars, fb_ratio.itertuples())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'FB={row.n_fibroblasts}\nCM={row.n_cardiomyocytes}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([AGE_LABELS[a] for a in AGE_ORDER], fontsize=10)
    ax.set_ylabel('Fibroblast / Cardiomyocyte Ratio')
    ax.set_title('A. Fibroblast Expansion with Age', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: GJA1-low fraction by age
    ax = axes[1]
    bars = ax.bar(x_pos, gja1_low['frac_gja1_low'].values,
                  color=[AGE_COLORS[a] for a in AGE_ORDER], alpha=0.8, edgecolor='black')
    for i, (bar, row) in enumerate(zip(bars, gja1_low.itertuples())):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{row.n_low}/{row.n_total}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([AGE_LABELS[a] for a in AGE_ORDER], fontsize=10)
    ax.set_ylabel('Fraction GJA1-low Cardiomyocytes')
    ax.set_title('B. GJA1-low Cell Fraction', fontsize=13, fontweight='bold')
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Expected if uniform')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Structural Remodeling & Gap Junction Loss',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig_II_4_fibroblast_gja1low.png')
    plt.close()
    print("  Saved fig_II_4_fibroblast_gja1low.png")


def fig_II_5_comprehensive_metrics(metrics_df):
    """Fig II-5: Comprehensive metric summary for all target genes."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    plot_genes = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp', 'Ryr2']
    x_pos = np.arange(len(AGE_ORDER))
    x_labels = ['3m', '18m', '20+m']

    for ax, gene in zip(axes.flat, plot_genes):
        gdf = metrics_df[metrics_df.gene == gene]
        if len(gdf) == 0:
            continue

        # Plot multiple metrics
        metrics_to_plot = [
            ('cv', 'CV', 'o-', '#1976D2'),
            ('bc', 'Bimodality', 's--', '#E53935'),
            ('entropy', 'Entropy', '^:', '#43A047'),
        ]

        ax2 = ax.twinx()
        for metric, label, style, color in metrics_to_plot:
            vals = [gdf[gdf.age == a][metric].values[0] if len(gdf[gdf.age == a]) > 0 else np.nan
                    for a in AGE_ORDER]
            if metric == 'entropy':
                ax2.plot(x_pos, vals, style, color=color, label=label, markersize=6, linewidth=1.5)
                ax2.set_ylabel('Entropy', color=color, fontsize=10)
            else:
                ax.plot(x_pos, vals, style, color=color, label=label, markersize=6, linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(gene, fontsize=12, fontweight='bold')
        ax.set_ylabel('CV / Bimodality Coeff.')
        ax.grid(alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    plt.suptitle('Cell-to-Cell Variability Metrics Across Aging',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/fig_II_5_all_metrics.png')
    plt.close()
    print("  Saved fig_II_5_all_metrics.png")


def statistical_tests(metrics_df, cm, gene_map):
    """Bootstrap tests and trend analysis."""
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    results = []
    for gene in ALL_TARGET_GENES + HOUSEKEEPING_GENES:
        gdf = metrics_df[metrics_df.gene == gene]
        if len(gdf) < 3:
            continue

        # Kruskal-Wallis on raw expression across 3 age groups
        groups = []
        for age in AGE_ORDER:
            mask = cm.obs['development_stage'] == age
            expr = get_gene_expr(cm[mask], gene, gene_map)
            if expr is not None:
                groups.append(expr[expr > 0])

        if len(groups) >= 2 and all(len(g) >= 5 for g in groups):
            H, p_kw = stats.kruskal(*groups)
        else:
            H, p_kw = np.nan, np.nan

        # Mann-Whitney young vs old
        if len(groups) >= 2 and len(groups[0]) >= 5 and len(groups[-1]) >= 5:
            U, p_mw = stats.mannwhitneyu(groups[0], groups[-1], alternative='two-sided')
            # Effect size: rank-biserial correlation
            n1, n2 = len(groups[0]), len(groups[-1])
            r_rb = 1 - 2*U / (n1*n2)
        else:
            p_mw, r_rb = np.nan, np.nan

        # CV trend (young to old)
        cvs = gdf.sort_values('age')['cv'].values
        cv_change = (cvs[-1] - cvs[0]) / cvs[0] * 100 if cvs[0] > 0 else np.nan

        results.append({
            'gene': gene,
            'KW_H': H, 'KW_p': p_kw,
            'MW_p_young_vs_old': p_mw,
            'rank_biserial_r': r_rb,
            'CV_young': cvs[0] if len(cvs) > 0 else np.nan,
            'CV_old': cvs[-1] if len(cvs) > 0 else np.nan,
            'CV_change_pct': cv_change,
        })

        sig = '***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'ns'
        print(f"  {gene:12s}: KW H={H:8.1f}, p={p_kw:.2e} {sig:3s} | "
              f"MW p={p_mw:.2e} | r_rb={r_rb:+.3f} | CV change: {cv_change:+.1f}%")

    return pd.DataFrame(results)


def fig_III_1_bridge(cm, gene_map):
    """
    Fig III-1: The Cross-Scale Bridge
    Left Y: sigma_beta from PTB-XL (human ECG) vs age
    Right Y: CV(GJA1) from TMS (mouse scRNA-seq) vs age
    X: normalized age
    """
    # Load PTB-XL aging data
    beta_df = pd.read_csv('results/beta_features.csv', index_col=0)
    meta = pd.read_csv('ptb-xl/ptbxl_database.csv', index_col=0)
    merged = beta_df.join(meta[['age', 'sex', 'scp_codes']], how='inner')
    # Filter NORM
    merged = merged[merged.scp_codes.str.contains('NORM', na=False)]
    merged = merged[merged.age.between(18, 95)]

    # Compute sigma_beta per age decade
    merged['decade'] = (merged.age // 10) * 10
    sigma_by_decade = merged.groupby('decade')['beta_std'].agg(['mean', 'sem']).reset_index()

    # CV(GJA1) from TMS
    cv_gja1_by_age = []
    for age in AGE_ORDER:
        mask = cm.obs['development_stage'] == age
        expr = get_gene_expr(cm[mask], 'Gja1', gene_map)
        expr_pos = expr[expr > 0]
        if len(expr_pos) > 5:
            # Bootstrap CI
            n_boot = 1000
            boot_cvs = []
            for _ in range(n_boot):
                sample = np.random.choice(expr_pos, len(expr_pos), replace=True)
                boot_cvs.append(np.std(sample) / np.mean(sample))
            cv_gja1_by_age.append({
                'age_mouse': age,
                'age_human_equiv': AGE_HUMAN_EQUIV[age],
                'cv_mean': np.mean(boot_cvs),
                'cv_ci_lo': np.percentile(boot_cvs, 2.5),
                'cv_ci_hi': np.percentile(boot_cvs, 97.5),
            })
    cv_gja1_df = pd.DataFrame(cv_gja1_by_age)

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y: sigma_beta (human)
    color1 = '#1565C0'
    ax1.errorbar(sigma_by_decade['decade'], sigma_by_decade['mean'],
                 yerr=sigma_by_decade['sem'], fmt='s-', color=color1,
                 linewidth=2, markersize=8, capsize=4, label='σ_β (Human ECG)')
    ax1.set_xlabel('Age (years, human-equivalent for mouse)', fontsize=13)
    ax1.set_ylabel('σ_β (inter-lead SD of spectral exponent)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right Y: CV(GJA1) (mouse)
    ax2 = ax1.twinx()
    color2 = '#C62828'
    ax2.errorbar(cv_gja1_df['age_human_equiv'], cv_gja1_df['cv_mean'],
                 yerr=[cv_gja1_df['cv_mean'] - cv_gja1_df['cv_ci_lo'],
                       cv_gja1_df['cv_ci_hi'] - cv_gja1_df['cv_mean']],
                 fmt='o-', color=color2, linewidth=2, markersize=10, capsize=4,
                 label='CV(GJA1) (Mouse scRNA-seq)')
    ax2.set_ylabel('CV of GJA1 expression (cell-to-cell)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Annotations for mouse ages
    for _, row in cv_gja1_df.iterrows():
        ax2.annotate(AGE_LABELS[row['age_mouse']],
                     (row['age_human_equiv'], row['cv_mean']),
                     textcoords='offset points', xytext=(10, 10),
                     fontsize=9, color=color2, alpha=0.7)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    ax1.set_title('The Bridge: Macro (ECG) ↔ Micro (scRNA-seq) Aging Trajectories',
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Add causal arrow annotation
    ax1.annotate('', xy=(0.85, 0.7), xytext=(0.85, 0.3),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
    ax1.text(0.88, 0.5, 'Cross-scale\nconsistency', transform=ax1.transAxes,
             fontsize=10, ha='left', va='center', color='gray', fontstyle='italic')

    plt.tight_layout()
    plt.savefig('results/fig_III_1_bridge.png')
    plt.close()
    print("  Saved fig_III_1_bridge.png")


def fig_II_fibrosis_genes(cm, fb, gene_map):
    """Additional figure: Fibrosis marker genes in fibroblasts by age."""
    fb_gene_map = dict(zip(fb.var['feature_name'], fb.var.index))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, gene in zip(axes, FIBROSIS_GENES):
        for age in AGE_ORDER:
            mask = fb.obs['development_stage'] == age
            expr = get_gene_expr(fb[mask], gene, fb_gene_map)
            if expr is None:
                continue
            expr_pos = expr[expr > 0]
            if len(expr_pos) < 10:
                continue
            expr_log = np.log1p(expr_pos)
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(expr_log, bw_method=0.3)
                x_grid = np.linspace(expr_log.min(), expr_log.max(), 200)
                ax.fill_between(x_grid, kde(x_grid), alpha=0.3,
                                color=AGE_COLORS[age], label=AGE_LABELS[age])
                ax.plot(x_grid, kde(x_grid), color=AGE_COLORS[age], linewidth=1.5)
            except Exception:
                pass

        ax.set_xlabel(f'log1p({gene})')
        ax.set_ylabel('Density')
        ax.set_title(gene, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Fibrosis Marker Expression in Cardiac Fibroblasts',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig_II_6_fibrosis_markers.png')
    plt.close()
    print("  Saved fig_II_6_fibrosis_markers.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("PART II: Molecular Substrate — Tabula Muris Senis Analysis")
    print("="*70)

    # ── Load data ──
    cm, fb, gene_map = load_data()

    # ── Compute metrics ──
    print("\n--- Computing gene metrics ---")
    metrics_df = compute_all_metrics(cm, gene_map)
    metrics_df.to_csv('results/tms_gene_metrics.csv', index=False)
    print(f"  Saved metrics for {metrics_df.gene.nunique()} genes to tms_gene_metrics.csv")

    # ── GJA1-low fraction ──
    print("\n--- GJA1-low fraction ---")
    gja1_low = compute_gja1_low_fraction(cm, gene_map)

    # ── Gene-gene correlations ──
    print("\n--- Gene-gene correlations (core genes) ---")
    corr_by_age = compute_gene_correlations(cm, gene_map, genes=CORE_GENES)

    # ── Fibroblast ratio ──
    print("\n--- Fibroblast / Cardiomyocyte ratio ---")
    fb_ratio = compute_fibroblast_ratio(cm, fb)

    # ── Statistical tests ──
    stat_results = statistical_tests(metrics_df, cm, gene_map)
    stat_results.to_csv('results/tms_statistical_tests.csv', index=False)
    print("  Saved tms_statistical_tests.csv")

    # ── Figures ──
    print("\n--- Generating figures ---")
    print("  Fig II-1: GJA1 variability...")
    fig_II_1_violin_gja1(cm, gene_map, metrics_df)

    print("  Fig II-2: Bimodality...")
    fig_II_2_bimodality(cm, gene_map)

    print("  Fig II-3: Gene correlation network...")
    fig_II_3_correlation_network(corr_by_age)

    print("  Fig II-4: Fibroblast expansion & GJA1-low...")
    fig_II_4_fibroblast_expansion(fb_ratio, gja1_low)

    print("  Fig II-5: Comprehensive metrics...")
    fig_II_5_comprehensive_metrics(metrics_df)

    print("  Fig II-6: Fibrosis markers in fibroblasts...")
    fig_II_fibrosis_genes(cm, fb, gene_map)

    # ── Cross-scale bridge (Part III) ──
    print("\n--- Part III: Cross-Scale Bridge ---")
    print("  Fig III-1: The Bridge...")
    try:
        fig_III_1_bridge(cm, gene_map)
    except Exception as e:
        print(f"  WARNING: Bridge figure failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Summary ──
    print("\n" + "="*70)
    print("SUMMARY OF KEY FINDINGS")
    print("="*70)

    gja1_metrics = metrics_df[metrics_df.gene == 'Gja1']
    if len(gja1_metrics) > 0:
        cv_young = gja1_metrics[gja1_metrics.age == AGE_ORDER[0]]['cv'].values[0]
        cv_old = gja1_metrics[gja1_metrics.age == AGE_ORDER[-1]]['cv'].values[0]
        print(f"\n  GJA1 CV: {cv_young:.3f} (young) -> {cv_old:.3f} (old) "
              f"[{(cv_old-cv_young)/cv_young*100:+.1f}%]")

    gapdh_metrics = metrics_df[metrics_df.gene == 'Gapdh']
    if len(gapdh_metrics) > 0:
        cv_young_hk = gapdh_metrics[gapdh_metrics.age == AGE_ORDER[0]]['cv'].values[0]
        cv_old_hk = gapdh_metrics[gapdh_metrics.age == AGE_ORDER[-1]]['cv'].values[0]
        print(f"  GAPDH CV: {cv_young_hk:.3f} (young) -> {cv_old_hk:.3f} (old) "
              f"[{(cv_old_hk-cv_young_hk)/cv_young_hk*100:+.1f}%] (housekeeping control)")

    print(f"\n  FB/CM ratio: "
          f"{fb_ratio[fb_ratio.age==AGE_ORDER[0]]['fb_cm_ratio'].values[0]:.1f} (young) -> "
          f"{fb_ratio[fb_ratio.age==AGE_ORDER[-1]]['fb_cm_ratio'].values[0]:.1f} (old)")

    if AGE_ORDER[0] in corr_by_age and AGE_ORDER[-1] in corr_by_age:
        mask_tri = np.triu(np.ones(corr_by_age[AGE_ORDER[0]].shape, dtype=bool), k=1)
        r_young = corr_by_age[AGE_ORDER[0]].values[mask_tri].mean()
        r_old = corr_by_age[AGE_ORDER[-1]].values[mask_tri].mean()
        print(f"  Mean gene-gene correlation: {r_young:.3f} (young) -> {r_old:.3f} (old)")

    print("\n  All figures saved to results/")
    print("="*70)


if __name__ == '__main__':
    main()
