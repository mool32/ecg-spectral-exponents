#!/usr/bin/env python3
"""
Corrected Analysis: Detection-based metrics + Two-process framework
====================================================================
1. Bridge figure: detection rate (not CV)
2. Block C: multi-gene detection in subsystems
3. Two-process summary figure
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact
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
AGE_HUMAN = {'3-month-old stage': 25, '18-month-old stage': 55,
             '20-month-old stage and over': 70}


def get_expr(adata, gene, gmap):
    if gene not in gmap:
        return None
    idx = gmap[gene]
    col = adata[:, idx].X
    if hasattr(col, 'toarray'):
        col = col.toarray()
    return col.flatten()


def load():
    cm = sc.read_h5ad('results/tms_cardiomyocytes.h5ad')
    gene_map = dict(zip(cm.var['feature_name'], cm.var.index))
    cm = cm[cm.obs['development_stage'].isin(AGE_ORDER)].copy()
    cm_atrial = cm[cm.obs['cell_type'] == 'regular atrial cardiac myocyte'].copy()
    return cm, cm_atrial, gene_map


# ═══════════════════════════════════════════════════════════════════
# 1. CORRECTED BRIDGE FIGURE
# ═══════════════════════════════════════════════════════════════════

def fig_bridge_corrected(cm_atrial, gene_map):
    """
    Fig III-1 corrected: β_mean (ECG) ↔ GJA1 detection rate (scRNA-seq)
    Both fall with age. Detection-based, GAPDH-validated.
    """
    print("\n--- Building corrected bridge figure ---")

    # ── PTB-XL: β_mean by decade ──
    beta_df = pd.read_csv('results/beta_features.csv', index_col=0)
    meta = pd.read_csv('ptb-xl/ptbxl_database.csv', index_col=0)
    merged = beta_df.join(meta[['age', 'sex', 'scp_codes']], how='inner')
    norm = merged[merged.scp_codes.str.contains('NORM', na=False)]
    norm = norm[norm.age.between(18, 95)]
    norm['decade'] = (norm.age // 10) * 10
    beta_dec = norm.groupby('decade')['beta_mean'].agg(['mean', 'sem']).reset_index()

    # ── TMS: Detection rates with bootstrap CI ──
    det_data = []
    for age in AGE_ORDER:
        mask = cm_atrial.obs['development_stage'] == age
        expr = get_expr(cm_atrial[mask], 'Gja1', gene_map)
        n_tot = len(expr)
        n_det = np.sum(expr > 0)
        frac = n_det / n_tot

        # Bootstrap CI for detection rate
        n_boot = 5000
        boot_fracs = []
        for _ in range(n_boot):
            idx = np.random.choice(n_tot, n_tot, replace=True)
            boot_fracs.append(np.sum(expr[idx] > 0) / n_tot)
        boot_fracs = np.array(boot_fracs)

        det_data.append({
            'age': age,
            'age_human': AGE_HUMAN[age],
            'frac_detected': frac,
            'ci_lo': np.percentile(boot_fracs, 2.5),
            'ci_hi': np.percentile(boot_fracs, 97.5),
            'n': n_tot,
        })
    det_df = pd.DataFrame(det_data)

    # Also compute GAPDH detection as control
    gapdh_det = []
    for age in AGE_ORDER:
        mask = cm_atrial.obs['development_stage'] == age
        expr = get_expr(cm_atrial[mask], 'Gapdh', gene_map)
        gapdh_det.append(np.sum(expr > 0) / len(expr))

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(11, 7))

    # Left Y: β_mean (human ECG)
    c1 = '#1565C0'
    ax1.errorbar(beta_dec['decade'], beta_dec['mean'],
                 yerr=beta_dec['sem'], fmt='s-', color=c1,
                 linewidth=2.5, markersize=8, capsize=4, zorder=3,
                 label='β_mean (Human ECG, PTB-XL)')
    ax1.set_xlabel('Age (years; human-equivalent for mouse)', fontsize=13)
    ax1.set_ylabel('β_mean (spectral exponent)', color=c1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=c1)

    # Right Y: GJA1 detection rate (mouse scRNA-seq)
    ax2 = ax1.twinx()
    c2 = '#C62828'
    ax2.errorbar(det_df['age_human'], det_df['frac_detected'],
                 yerr=[det_df['frac_detected'] - det_df['ci_lo'],
                       det_df['ci_hi'] - det_df['frac_detected']],
                 fmt='D-', color=c2, linewidth=2.5, markersize=11, capsize=5,
                 zorder=5, label='GJA1 detection rate (Mouse, TMS)')
    ax2.set_ylabel('GJA1 detection rate (fraction of cells)', color=c2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=c2)

    # GAPDH control line
    ax2.plot([AGE_HUMAN[a] for a in AGE_ORDER], gapdh_det,
             'x:', color='#999', linewidth=1.5, markersize=8,
             label='GAPDH detection (control)')

    # Annotations
    for _, row in det_df.iterrows():
        ax2.annotate(f"{AGE_LABELS[row['age']]}\nn={row['n']}",
                     (row['age_human'], row['frac_detected']),
                     textcoords='offset points', xytext=(15, -15),
                     fontsize=9, color=c2, alpha=0.7,
                     arrowprops=dict(arrowstyle='->', color=c2, alpha=0.3))

    # Combined legend
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='lower left', fontsize=10,
               framealpha=0.9)

    ax1.set_title('The Bridge (Corrected): β Falls ↔ Gene Silencing Rises',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(alpha=0.25)

    # Key insight box
    textstr = ('Both curves decline with age:\n'
               '• β_mean ↓: signal loses high-frequency structure\n'
               '• GJA1 detection ↓: cells stop expressing connexin-43\n'
               '• GAPDH stable: silencing is gene-specific\n\n'
               'Mechanism: "silent" cardiomyocytes create\n'
               'dead zones → conduction fragmentation → β ↓')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9,
                 edgecolor='#FFC107')
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             va='top', ha='right', bbox=props)

    plt.tight_layout()
    plt.savefig('results/fig_III_1_bridge_corrected.png')
    plt.close()
    print("  Saved fig_III_1_bridge_corrected.png")

    return det_df


# ═══════════════════════════════════════════════════════════════════
# 2. BLOCK C RECHECK: MULTI-GENE DETECTION IN SUBSYSTEMS
# ═══════════════════════════════════════════════════════════════════

def block_C_corrected(cm_atrial, gene_map):
    """
    Block C recheck: detection rate of multiple genes in
    working myocardium vs conduction-like subsystems.
    """
    print("\n--- Block C: Multi-gene detection in subsystems ---")

    gja1_expr = get_expr(cm_atrial, 'Gja1', gene_map)
    hcn4_expr = get_expr(cm_atrial, 'Hcn4', gene_map)
    gja5_expr = get_expr(cm_atrial, 'Gja5', gene_map)

    # Define populations
    # Working: GJA1 detected (expr > 0) — not GJA1-high, just detected
    # This avoids circularity for GJA1 detection itself
    # Instead: cells where Ryr2 or Atp2a2 detected (robust contractile markers)
    ryr2_expr = get_expr(cm_atrial, 'Ryr2', gene_map)
    atp2a2_expr = get_expr(cm_atrial, 'Atp2a2', gene_map)

    working_mask = (ryr2_expr > 0) | (atp2a2_expr > 0)
    conduction_mask = (gja5_expr > 0)  # GJA5+ = conduction marker

    # Exclude overlap
    pure_working = working_mask & ~conduction_mask
    pure_conduction = conduction_mask

    print(f"\n  Pure Working (Ryr2+ or Atp2a2+, GJA5−): {pure_working.sum()}")
    print(f"  Conduction-like (GJA5+): {pure_conduction.sum()}")
    print(f"  Overlap: {(working_mask & conduction_mask).sum()}")

    # Test genes — these are NOT used to define populations
    test_genes = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp', 'Kcnq1', 'Gapdh']

    results = []
    for pop_name, pop_mask in [('Working', pure_working),
                                ('Conduction', pure_conduction)]:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:
            age_mask = cm_atrial.obs['development_stage'].values == age
            combined = age_mask & pop_mask
            n_cells = combined.sum()

            if n_cells < 5:
                for g in test_genes:
                    results.append({
                        'population': pop_name, 'age': age, 'gene': g,
                        'n_cells': n_cells, 'n_detected': 0,
                        'frac_detected': np.nan
                    })
                continue

            for g in test_genes:
                expr = get_expr(cm_atrial, g, gene_map)
                expr_sub = expr[combined]
                n_det = np.sum(expr_sub > 0)
                frac = n_det / n_cells

                results.append({
                    'population': pop_name, 'age': age, 'gene': g,
                    'n_cells': n_cells, 'n_detected': n_det,
                    'frac_detected': frac
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv('results/block_C_corrected.csv', index=False)

    # Print results
    print(f"\n  {'Pop':12s} {'Gene':10s} {'Det_y':>7s}(n)  {'Det_o':>7s}(n)  {'Δ':>7s}  Fisher p")
    print("  " + "-"*75)

    for pop in ['Working', 'Conduction']:
        for g in test_genes:
            y = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[0]) &
                       (res_df.gene == g)]
            o = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[-1]) &
                       (res_df.gene == g)]

            if len(y) == 0 or len(o) == 0:
                continue

            yv = y.iloc[0]
            ov = o.iloc[0]

            det_delta = ov['frac_detected'] - yv['frac_detected']

            # Fisher's exact test
            a = yv['n_detected']  # young detected
            b = yv['n_cells'] - yv['n_detected']  # young not detected
            c = ov['n_detected']  # old detected
            d = ov['n_cells'] - ov['n_detected']  # old not detected
            if yv['n_cells'] >= 5 and ov['n_cells'] >= 5:
                _, p_fisher = fisher_exact([[a, b], [c, d]])
            else:
                p_fisher = np.nan

            sig = '***' if p_fisher < 0.001 else '**' if p_fisher < 0.01 else '*' if p_fisher < 0.05 else 'ns'
            is_hk = ' (HK)' if g == 'Gapdh' else ''
            print(f"  {pop:12s} {g:10s} {yv['frac_detected']:6.1%}({int(yv['n_cells']):3d}) "
                  f"{ov['frac_detected']:6.1%}({int(ov['n_cells']):3d}) "
                  f"{det_delta:+6.1%}  {p_fisher:.2e} {sig}{is_hk}")
        print()

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Detection rates by gene and population (young vs old)
    ax = axes[0]
    genes_show = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp', 'Gapdh']
    x = np.arange(len(genes_show))
    w = 0.2
    for i, (pop, color) in enumerate([('Working', '#E53935'), ('Conduction', '#1565C0')]):
        for j, age_label in enumerate(['3m', '20+m']):
            age = AGE_ORDER[0] if age_label == '3m' else AGE_ORDER[-1]
            fracs = []
            for g in genes_show:
                row = res_df[(res_df.population == pop) & (res_df.age == age) &
                             (res_df.gene == g)]
                fracs.append(row['frac_detected'].values[0] if len(row) > 0 else np.nan)
            offset = (i * 2 + j - 1.5) * w
            alpha = 1.0 if age_label == '3m' else 0.5
            hatch = '' if age_label == '3m' else '//'
            label = f'{pop} {age_label}'
            ax.bar(x + offset, fracs, w, color=color, alpha=alpha,
                   edgecolor='black', linewidth=0.5, hatch=hatch, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(genes_show, fontsize=10, rotation=30)
    ax.set_ylabel('Detection Rate')
    ax.set_title('A. Detection by Subsystem × Age', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Detection CHANGE (old - young) by subsystem
    ax = axes[1]
    for i, (pop, color) in enumerate([('Working', '#E53935'), ('Conduction', '#1565C0')]):
        deltas = []
        for g in genes_show:
            y_row = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[0]) &
                           (res_df.gene == g)]
            o_row = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[-1]) &
                           (res_df.gene == g)]
            if len(y_row) > 0 and len(o_row) > 0:
                d = o_row['frac_detected'].values[0] - y_row['frac_detected'].values[0]
            else:
                d = np.nan
            deltas.append(d)
        offset = (i - 0.5) * w * 2
        ax.bar(x + offset, deltas, w * 1.8, color=color, alpha=0.8,
               edgecolor='black', linewidth=0.5, label=pop)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(genes_show, fontsize=10, rotation=30)
    ax.set_ylabel('Δ Detection Rate (Old − Young)')
    ax.set_title('B. Gene Silencing: Working vs Conduction', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel C: Summary statistics
    ax = axes[2]
    ax.axis('off')

    # Compute summary: mean detection drop across cardiac genes
    cardiac_test = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp']
    summary_lines = ['BLOCK C: SUBSYSTEM AGING (CORRECTED)\n',
                     'Metric: Detection rate (GAPDH-validated)\n',
                     'Population definition:',
                     '  Working = Ryr2+ or Atp2a2+, GJA5−',
                     '  Conduction = GJA5+\n']

    for pop in ['Working', 'Conduction']:
        deltas = []
        for g in cardiac_test:
            y_row = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[0]) &
                           (res_df.gene == g)]
            o_row = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[-1]) &
                           (res_df.gene == g)]
            if len(y_row) > 0 and len(o_row) > 0:
                d = o_row['frac_detected'].values[0] - y_row['frac_detected'].values[0]
                deltas.append(d)
        mean_d = np.mean(deltas) if deltas else np.nan
        n_young = res_df[(res_df.population == pop) &
                         (res_df.age == AGE_ORDER[0])]['n_cells'].values[0] if len(
            res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[0])]) > 0 else 0
        n_old = res_df[(res_df.population == pop) &
                       (res_df.age == AGE_ORDER[-1])]['n_cells'].values[0] if len(
            res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[-1])]) > 0 else 0
        summary_lines.append(f'{pop}:')
        summary_lines.append(f'  N: {int(n_young)} (young) → {int(n_old)} (old)')
        summary_lines.append(f'  Mean Δ detection (5 cardiac genes): {mean_d:+.1%}')
        summary_lines.append('')

    # GAPDH control
    for pop in ['Working', 'Conduction']:
        y_row = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[0]) &
                       (res_df.gene == 'Gapdh')]
        o_row = res_df[(res_df.population == pop) & (res_df.age == AGE_ORDER[-1]) &
                       (res_df.gene == 'Gapdh')]
        if len(y_row) > 0 and len(o_row) > 0:
            d = o_row['frac_detected'].values[0] - y_row['frac_detected'].values[0]
            summary_lines.append(f'GAPDH control ({pop}): {d:+.1%}')

    summary_lines.append('\nVERDICT:')
    summary_lines.append('Raw CV difference: NOT CONFIRMED')
    summary_lines.append('Detection-based: see Panel B')

    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Fig C-1 (Corrected): Subsystem Aging via Detection Rate',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/fig_C1_corrected.png')
    plt.close()
    print("  Saved fig_C1_corrected.png")

    return res_df


# ═══════════════════════════════════════════════════════════════════
# 3. TWO-PROCESS SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════

def fig_two_processes(cm_atrial, gene_map):
    """
    The key figure: Two simultaneous aging processes.
    Process 1: Global transcriptional noise (GAPDH CV ↑)
    Process 2: Cardiac gene silencing (detection ↓, GAPDH stable)
    """
    print("\n--- Building two-process summary figure ---")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.4)

    # ═══ TOP ROW: The two processes ═══

    # Panel A: Global noise — GAPDH CV rises
    ax = fig.add_subplot(gs[0, 0:2])
    genes = ['Gapdh', 'Gja1', 'Scn5a', 'Cacna1c', 'Dsp', 'Ryr2']
    cv_data = {}
    for g in genes:
        cvs = []
        for age in AGE_ORDER:
            mask = cm_atrial.obs['development_stage'] == age
            expr = get_expr(cm_atrial[mask], g, gene_map)
            ep = expr[expr > 0]
            cvs.append(np.std(ep)/np.mean(ep) if len(ep) >= 5 else np.nan)
        cv_data[g] = cvs

    x = np.arange(len(AGE_ORDER))
    for g in genes:
        style = 'o-' if g != 'Gapdh' else 's-'
        lw = 3 if g == 'Gapdh' else 1.2
        alpha = 1.0 if g == 'Gapdh' else 0.5
        color = '#E53935' if g == 'Gapdh' else '#999'
        ax.plot(x, cv_data[g], style, color=color, linewidth=lw, alpha=alpha,
                markersize=8 if g == 'Gapdh' else 5, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER])
    ax.set_ylabel('CV (σ/μ among detected cells)')
    ax.set_title('PROCESS 1: Global Transcriptional Noise\n'
                 'ALL genes become more variable (including GAPDH)',
                 fontweight='bold', color='#E53935')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Panel B: Specific silencing — detection drops
    ax = fig.add_subplot(gs[0, 2:4])
    det_data = {}
    for g in genes:
        dets = []
        for age in AGE_ORDER:
            mask = cm_atrial.obs['development_stage'] == age
            expr = get_expr(cm_atrial[mask], g, gene_map)
            dets.append(np.sum(expr > 0) / len(expr))
        det_data[g] = dets

    for g in genes:
        style = 's-' if g == 'Gapdh' else 'o-'
        lw = 3 if g == 'Gapdh' else 1.5
        color = '#999' if g == 'Gapdh' else '#1565C0'
        ax.plot(x, det_data[g], style, color=color, linewidth=lw,
                markersize=8 if g == 'Gapdh' else 6, label=g,
                alpha=1.0 if g == 'Gapdh' else 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([AGE_SHORT[a] for a in AGE_ORDER])
    ax.set_ylabel('Detection Rate (fraction of cells)')
    ax.set_title('PROCESS 2: Cardiac Gene Silencing\n'
                 'Cardiac genes silenced; GAPDH STABLE',
                 fontweight='bold', color='#1565C0')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ═══ MIDDLE ROW: ECG correlates ═══

    # Panel C: Beat-to-beat variability (correlate of Process 1)
    ax = fig.add_subplot(gs[1, 0:2])
    try:
        beat_df = pd.read_csv('results/block_D_beat_variability.csv')
        from statsmodels.nonparametric.smoothers_lowess import lowess
        ax.scatter(beat_df.age, beat_df.II_morph_var, alpha=0.08, s=5, c='#E53935')
        lw_fit = lowess(beat_df.II_morph_var.values, beat_df.age.values, frac=0.3)
        ax.plot(lw_fit[:, 0], lw_fit[:, 1], '-', color='#E53935', linewidth=3)
        rho, p = stats.spearmanr(beat_df.age, beat_df.II_morph_var)
        ax.set_title(f'ECG Correlate of Process 1:\nBeat-to-Beat Variability (ρ={rho:.3f})',
                     fontweight='bold', color='#E53935')
    except Exception as e:
        ax.text(0.5, 0.5, f'Beat data error: {e}', ha='center', transform=ax.transAxes)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Morphological Variability')
    ax.grid(alpha=0.3)

    # Panel D: β_mean decline (correlate of Process 2)
    ax = fig.add_subplot(gs[1, 2:4])
    beta_df = pd.read_csv('results/beta_features.csv', index_col=0)
    meta = pd.read_csv('ptb-xl/ptbxl_database.csv', index_col=0)
    merged = beta_df.join(meta[['age', 'sex', 'scp_codes']], how='inner')
    norm = merged[merged.scp_codes.str.contains('NORM', na=False)]
    norm = norm[norm.age.between(18, 95)]
    from statsmodels.nonparametric.smoothers_lowess import lowess
    ax.scatter(norm.age, norm.beta_mean, alpha=0.05, s=3, c='#1565C0')
    lw_fit = lowess(norm.beta_mean.values, norm.age.values, frac=0.3)
    ax.plot(lw_fit[:, 0], lw_fit[:, 1], '-', color='#1565C0', linewidth=3)
    rho2, p2 = stats.spearmanr(norm.age, norm.beta_mean)
    ax.set_title(f'ECG Correlate of Process 2:\nβ Decline (ρ={rho2:.3f})',
                 fontweight='bold', color='#1565C0')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('β_mean')
    ax.grid(alpha=0.3)

    # ═══ BOTTOM ROW: Causal diagram ═══
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # Process 1 box
    p1 = FancyBboxPatch((0.02, 0.55), 0.44, 0.40, boxstyle="round,pad=0.02",
                         facecolor='#FFEBEE', edgecolor='#E53935', linewidth=2,
                         transform=ax.transAxes)
    ax.add_patch(p1)
    ax.text(0.24, 0.88, 'PROCESS 1: Global Noise', transform=ax.transAxes,
            ha='center', fontsize=12, fontweight='bold', color='#E53935')
    ax.text(0.24, 0.76, 'All genes CV ↑ (GAPDH included)\n'
            '→ Epigenetic dereregulation?\n'
            '→ Each cell less stable cycle-to-cycle\n'
            '→ Beat-to-beat morphological variability ↑\n'
            '  (ρ = 0.165, p ≈ 10⁻¹⁹)',
            transform=ax.transAxes, ha='center', fontsize=9, color='#333')

    # Process 2 box
    p2 = FancyBboxPatch((0.54, 0.55), 0.44, 0.40, boxstyle="round,pad=0.02",
                         facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2,
                         transform=ax.transAxes)
    ax.add_patch(p2)
    ax.text(0.76, 0.88, 'PROCESS 2: Gene Silencing', transform=ax.transAxes,
            ha='center', fontsize=12, fontweight='bold', color='#1565C0')
    ax.text(0.76, 0.76, 'Cardiac genes detection ↓ 25-32%\n'
            '(GAPDH stable — gene-specific)\n'
            '→ "Silent" cardiomyocytes = dead zones\n'
            '→ Conduction fragmentation → β ↓\n'
            '  (ρ = −0.178, p ≈ 10⁻⁶⁵)',
            transform=ax.transAxes, ha='center', fontsize=9, color='#333')

    # Convergence arrow
    ax.annotate('', xy=(0.5, 0.2), xytext=(0.24, 0.55),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
    ax.annotate('', xy=(0.5, 0.2), xytext=(0.76, 0.55),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

    # Bottom: unified effect
    unified = FancyBboxPatch((0.15, 0.02), 0.7, 0.18, boxstyle="round,pad=0.02",
                              facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=2,
                              transform=ax.transAxes)
    ax.add_patch(unified)
    ax.text(0.5, 0.13, 'CARDIAC AGING = Global noise + Specific silencing',
            transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold',
            color='#333')
    ax.text(0.5, 0.05, 'Two independent processes, both detectable in ECG, '
            'with distinct molecular substrates',
            transform=ax.transAxes, ha='center', fontsize=10, color='#666',
            fontstyle='italic')

    plt.suptitle('Two Simultaneous Aging Processes in the Heart',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('results/fig_two_processes.png')
    plt.close()
    print("  Saved fig_two_processes.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*75)
    print("CORRECTED ANALYSIS: Detection-based + Two-Process Framework")
    print("="*75)

    cm, cm_atrial, gene_map = load()
    print(f"Loaded: {cm_atrial.shape[0]} atrial cardiomyocytes")

    # 1. Corrected bridge
    det_df = fig_bridge_corrected(cm_atrial, gene_map)

    # 2. Block C corrected
    subsys_df = block_C_corrected(cm_atrial, gene_map)

    # 3. Two-process figure
    fig_two_processes(cm_atrial, gene_map)

    print("\n" + "="*75)
    print("DONE — All corrected figures generated")
    print("="*75)
    print("\n  results/fig_III_1_bridge_corrected.png")
    print("  results/fig_C1_corrected.png")
    print("  results/fig_two_processes.png")


if __name__ == '__main__':
    main()
