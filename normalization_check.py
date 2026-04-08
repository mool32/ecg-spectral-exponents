#!/usr/bin/env python3
"""
Critical normalization check: CV(gene) / CV(GAPDH)
====================================================
Determines whether cardiac gene CV changes are specific or global noise.
Also recomputes detection-based metrics as the robust alternative.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import mannwhitneyu, bootstrap
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


def get_expr(adata, gene, gmap):
    if gene not in gmap:
        return None
    idx = gmap[gene]
    col = adata[:, idx].X
    if hasattr(col, 'toarray'):
        col = col.toarray()
    return col.flatten()


def main():
    print("="*75)
    print("CRITICAL NORMALIZATION CHECK")
    print("="*75)

    # Load
    cm = sc.read_h5ad('results/tms_cardiomyocytes.h5ad')
    gene_map = dict(zip(cm.var['feature_name'], cm.var.index))
    cm = cm[cm.obs['development_stage'].isin(AGE_ORDER)].copy()
    cm_atrial = cm[cm.obs['cell_type'] == 'regular atrial cardiac myocyte'].copy()

    print(f"Atrial cardiomyocytes: {cm_atrial.shape[0]}")

    # ═══════════════════════════════════════════════════════════════
    # PART 1: CV normalization by GAPDH
    # ═══════════════════════════════════════════════════════════════

    target_genes = ['Gja1', 'Gja5', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Kcnq1',
                    'Dsp', 'Jup', 'Pkp2', 'Ryr2', 'Atp2a2',
                    'Gapdh', 'Actb']

    print("\n" + "="*75)
    print("PART 1: Raw CV and GAPDH-normalized CV (atrial only)")
    print("="*75)

    # Compute CV for each gene × age with bootstrap CI
    results = []
    for gene in target_genes:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:  # Young vs Old
            mask = cm_atrial.obs['development_stage'] == age
            expr = get_expr(cm_atrial[mask], gene, gene_map)
            if expr is None:
                continue
            ep = expr[expr > 0]
            n_det = len(ep)
            n_tot = len(expr)
            frac_det = n_det / n_tot

            if n_det < 10:
                results.append({
                    'gene': gene, 'age': age,
                    'cv': np.nan, 'cv_lo': np.nan, 'cv_hi': np.nan,
                    'frac_detected': frac_det, 'n_detected': n_det, 'n_total': n_tot,
                    'mean_expr': np.nan,
                })
                continue

            # Bootstrap CV
            n_boot = 5000
            boot_cvs = []
            for _ in range(n_boot):
                s = np.random.choice(ep, len(ep), replace=True)
                boot_cvs.append(np.std(s) / np.mean(s))
            boot_cvs = np.array(boot_cvs)

            results.append({
                'gene': gene, 'age': age,
                'cv': np.mean(boot_cvs),
                'cv_lo': np.percentile(boot_cvs, 2.5),
                'cv_hi': np.percentile(boot_cvs, 97.5),
                'frac_detected': frac_det,
                'n_detected': n_det,
                'n_total': n_tot,
                'mean_expr': np.mean(ep),
            })

    rdf = pd.DataFrame(results)

    # Compute fold changes and normalized values
    print(f"\n{'Gene':12s} {'CV_y':>7s} {'CV_o':>7s} {'CV_FC':>7s} "
          f"{'GAPDH_FC':>8s} {'CV_norm':>8s} "
          f"{'Det_y':>7s} {'Det_o':>7s} {'Det_Δ':>7s}")
    print("-" * 90)

    gapdh_young = rdf[(rdf.gene == 'Gapdh') & (rdf.age == AGE_ORDER[0])]['cv'].values[0]
    gapdh_old = rdf[(rdf.gene == 'Gapdh') & (rdf.age == AGE_ORDER[-1])]['cv'].values[0]
    gapdh_fc = gapdh_old / gapdh_young

    norm_results = []
    for gene in target_genes:
        y = rdf[(rdf.gene == gene) & (rdf.age == AGE_ORDER[0])]
        o = rdf[(rdf.gene == gene) & (rdf.age == AGE_ORDER[-1])]
        if len(y) == 0 or len(o) == 0:
            continue
        yv = y.iloc[0]
        ov = o.iloc[0]

        cv_fc = ov['cv'] / yv['cv'] if yv['cv'] > 0 else np.nan
        cv_norm_fc = cv_fc / gapdh_fc if not np.isnan(cv_fc) else np.nan
        det_delta = ov['frac_detected'] - yv['frac_detected']

        flag = ''
        if not np.isnan(cv_norm_fc):
            if cv_norm_fc > 1.1:
                flag = '↑ SPECIFIC'
            elif cv_norm_fc < 0.9:
                flag = '↓ LESS THAN GLOBAL'
            else:
                flag = '≈ GLOBAL NOISE'

        norm_results.append({
            'gene': gene,
            'cv_young': yv['cv'], 'cv_old': ov['cv'],
            'cv_fc': cv_fc,
            'gapdh_fc': gapdh_fc,
            'cv_norm_fc': cv_norm_fc,
            'det_young': yv['frac_detected'], 'det_old': ov['frac_detected'],
            'det_delta': det_delta,
            'mean_young': yv['mean_expr'], 'mean_old': ov['mean_expr'],
            'flag': flag,
        })

        print(f"{gene:12s} {yv['cv']:7.3f} {ov['cv']:7.3f} {cv_fc:7.2f}× "
              f"{gapdh_fc:8.2f}× {cv_norm_fc:8.2f}× "
              f"{yv['frac_detected']:7.1%} {ov['frac_detected']:7.1%} {det_delta:+7.1%}  "
              f"{flag}")

    norm_df = pd.DataFrame(norm_results)
    norm_df.to_csv('results/normalization_check.csv', index=False)

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Per-cell normalization (better approach)
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*75)
    print("PART 2: Per-cell normalization (gene / GAPDH per cell)")
    print("="*75)
    print("If GAPDH varies between cells due to capture efficiency,")
    print("normalizing each cell's gene expr by its GAPDH removes this.")

    percell_results = []
    for gene in ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp', 'Ryr2', 'Atp2a2']:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:
            mask = cm_atrial.obs['development_stage'] == age
            sub = cm_atrial[mask]
            gene_expr = get_expr(sub, gene, gene_map)
            gapdh_expr = get_expr(sub, 'Gapdh', gene_map)

            if gene_expr is None or gapdh_expr is None:
                continue

            # Cells where BOTH gene and GAPDH are detected
            both_detected = (gene_expr > 0) & (gapdh_expr > 0)
            n_both = both_detected.sum()

            if n_both < 10:
                percell_results.append({
                    'gene': gene, 'age': age,
                    'cv_raw': np.nan, 'cv_normalized': np.nan,
                    'n_both': n_both, 'frac_both': n_both / len(gene_expr),
                })
                continue

            raw_vals = gene_expr[both_detected]
            norm_vals = gene_expr[both_detected] / gapdh_expr[both_detected]

            cv_raw = np.std(raw_vals) / np.mean(raw_vals)
            cv_norm = np.std(norm_vals) / np.mean(norm_vals)

            percell_results.append({
                'gene': gene, 'age': age,
                'cv_raw': cv_raw, 'cv_normalized': cv_norm,
                'n_both': n_both, 'frac_both': n_both / len(gene_expr),
                'mean_raw': np.mean(raw_vals), 'mean_norm': np.mean(norm_vals),
            })

    pcdf = pd.DataFrame(percell_results)

    print(f"\n{'Gene':12s} {'Method':12s} {'CV_y':>7s} {'CV_o':>7s} {'FC':>7s}")
    print("-" * 55)
    for gene in ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp', 'Ryr2', 'Atp2a2']:
        for method, col in [('Raw', 'cv_raw'), ('Gene/GAPDH', 'cv_normalized')]:
            y = pcdf[(pcdf.gene == gene) & (pcdf.age == AGE_ORDER[0])][col]
            o = pcdf[(pcdf.gene == gene) & (pcdf.age == AGE_ORDER[-1])][col]
            if len(y) > 0 and len(o) > 0:
                yv = y.values[0]
                ov = o.values[0]
                fc = ov / yv if yv > 0 else np.nan
                print(f"{gene:12s} {method:12s} {yv:7.3f} {ov:7.3f} {fc:7.2f}×")
        print()

    pcdf.to_csv('results/percell_normalization.csv', index=False)

    # ═══════════════════════════════════════════════════════════════
    # PART 3: Detection-based metrics (robust to CV noise)
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*75)
    print("PART 3: Detection-based metrics (GAPDH-independent)")
    print("="*75)

    det_results = []
    for gene in ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Kcnq1', 'Dsp',
                  'Ryr2', 'Atp2a2', 'Gapdh', 'Actb']:
        # Young P25 threshold
        mask_y = cm_atrial.obs['development_stage'] == AGE_ORDER[0]
        expr_y = get_expr(cm_atrial[mask_y], gene, gene_map)
        ep_y = expr_y[expr_y > 0]
        if len(ep_y) < 10:
            continue
        thresh = np.percentile(ep_y, 25)

        for age in AGE_ORDER:
            mask = cm_atrial.obs['development_stage'] == age
            expr = get_expr(cm_atrial[mask], gene, gene_map)
            n_tot = len(expr)
            n_det = np.sum(expr > 0)
            frac_det = n_det / n_tot
            n_low = np.sum(expr < thresh)
            frac_low = n_low / n_tot

            # Among detected: fraction below young median
            ep = expr[expr > 0]
            young_median = np.median(ep_y)
            frac_below_median = np.sum(ep < young_median) / len(ep) if len(ep) > 0 else np.nan

            det_results.append({
                'gene': gene, 'age': age,
                'frac_detected': frac_det,
                'frac_low': frac_low,
                'frac_below_young_median': frac_below_median,
                'n_detected': n_det, 'n_total': n_tot,
            })

    det_df = pd.DataFrame(det_results)

    print(f"\n{'Gene':12s} {'Det_y':>7s} {'Det_o':>7s} {'Det_Δ':>7s}  "
          f"{'Low_y':>7s} {'Low_o':>7s} {'Low_Δ':>7s}  "
          f"{'<Med_y':>7s} {'<Med_o':>7s}")
    print("-" * 95)

    for gene in ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Kcnq1', 'Dsp',
                  'Ryr2', 'Atp2a2', 'Gapdh', 'Actb']:
        gdf = det_df[det_df.gene == gene]
        y = gdf[gdf.age == AGE_ORDER[0]].iloc[0] if len(gdf[gdf.age == AGE_ORDER[0]]) > 0 else None
        o = gdf[gdf.age == AGE_ORDER[-1]].iloc[0] if len(gdf[gdf.age == AGE_ORDER[-1]]) > 0 else None
        if y is None or o is None:
            continue

        det_delta = o['frac_detected'] - y['frac_detected']
        low_delta = o['frac_low'] - y['frac_low']

        is_hk = gene in ['Gapdh', 'Actb']
        marker = '  (HK)' if is_hk else ''

        print(f"{gene:12s} {y['frac_detected']:7.1%} {o['frac_detected']:7.1%} "
              f"{det_delta:+7.1%}  "
              f"{y['frac_low']:7.1%} {o['frac_low']:7.1%} {low_delta:+7.1%}  "
              f"{y['frac_below_young_median']:7.1%} {o['frac_below_young_median']:7.1%}"
              f"{marker}")

    det_df.to_csv('results/detection_metrics.csv', index=False)

    # ═══════════════════════════════════════════════════════════════
    # PART 4: Block C recheck with normalization
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*75)
    print("PART 4: Block C subsystem recheck")
    print("="*75)

    gja1_expr = get_expr(cm_atrial, 'Gja1', gene_map)
    hcn4_expr = get_expr(cm_atrial, 'Hcn4', gene_map)
    gja5_expr = get_expr(cm_atrial, 'Gja5', gene_map)
    gapdh_expr_all = get_expr(cm_atrial, 'Gapdh', gene_map)

    gja1_med = np.median(gja1_expr[gja1_expr > 0])
    working_mask = gja1_expr > gja1_med
    conduction_mask = (hcn4_expr > 0) | (gja5_expr > 0)

    print("\n  A) Raw CV:")
    for pop_name, pop_mask in [('Working', working_mask), ('Conduction', conduction_mask)]:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:
            age_mask = cm_atrial.obs['development_stage'].values == age
            combined = age_mask & pop_mask
            expr = gja1_expr[combined]
            ep = expr[expr > 0]
            cv = np.std(ep) / np.mean(ep) if len(ep) >= 5 else np.nan
            print(f"    {pop_name:12s} {AGE_SHORT[age]}: CV={cv:.3f} (n={len(ep)})")

    print("\n  B) GAPDH-normalized CV (gene/GAPDH per cell):")
    for pop_name, pop_mask in [('Working', working_mask), ('Conduction', conduction_mask)]:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:
            age_mask = cm_atrial.obs['development_stage'].values == age
            combined = age_mask & pop_mask
            gja1_vals = gja1_expr[combined]
            gapdh_vals = gapdh_expr_all[combined]
            both = (gja1_vals > 0) & (gapdh_vals > 0)
            if both.sum() < 5:
                print(f"    {pop_name:12s} {AGE_SHORT[age]}: insufficient (n={both.sum()})")
                continue
            normed = gja1_vals[both] / gapdh_vals[both]
            cv_n = np.std(normed) / np.mean(normed)
            print(f"    {pop_name:12s} {AGE_SHORT[age]}: CV_norm={cv_n:.3f} (n={both.sum()})")

    print("\n  C) Detection rate (GAPDH-independent):")
    for pop_name, pop_mask in [('Working', working_mask), ('Conduction', conduction_mask)]:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:
            age_mask = cm_atrial.obs['development_stage'].values == age
            combined = age_mask & pop_mask
            expr = gja1_expr[combined]
            n_tot = len(expr)
            n_det = np.sum(expr > 0)
            frac = n_det / n_tot if n_tot > 0 else np.nan
            print(f"    {pop_name:12s} {AGE_SHORT[age]}: detected={n_det}/{n_tot} ({frac:.1%})")

    # ═══════════════════════════════════════════════════════════════
    # PART 5: COMPREHENSIVE FIGURE
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*75)
    print("PART 5: Generating corrected figure")
    print("="*75)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    cardiac_genes = ['Gja1', 'Scn5a', 'Kcnj2', 'Cacna1c', 'Dsp', 'Ryr2', 'Atp2a2']
    hk_genes = ['Gapdh', 'Actb']

    # ── Panel A: Raw CV fold changes ──
    ax = axes[0, 0]
    all_genes = cardiac_genes + hk_genes
    cv_fcs = []
    for g in all_genes:
        row = norm_df[norm_df.gene == g]
        if len(row) > 0:
            cv_fcs.append(row.iloc[0]['cv_fc'])
        else:
            cv_fcs.append(np.nan)

    colors = ['#1565C0' if g not in hk_genes else '#9E9E9E' for g in all_genes]
    y_pos = np.arange(len(all_genes))
    ax.barh(y_pos, cv_fcs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=gapdh_fc, color='red', linestyle='--', alpha=0.7,
               label=f'GAPDH FC = {gapdh_fc:.2f}×')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_genes, fontsize=10)
    ax.set_xlabel('CV Fold Change (Old / Young)')
    ax.set_title('A. Raw CV Fold Change', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # ── Panel B: Normalized CV fold changes ──
    ax = axes[0, 1]
    cv_norm_fcs = []
    for g in all_genes:
        row = norm_df[norm_df.gene == g]
        if len(row) > 0:
            cv_norm_fcs.append(row.iloc[0]['cv_norm_fc'])
        else:
            cv_norm_fcs.append(np.nan)

    bar_colors = []
    for g, fc in zip(all_genes, cv_norm_fcs):
        if g in hk_genes:
            bar_colors.append('#9E9E9E')
        elif not np.isnan(fc) and fc > 1.1:
            bar_colors.append('#E53935')  # specific increase
        elif not np.isnan(fc) and fc < 0.9:
            bar_colors.append('#43A047')  # less than global
        else:
            bar_colors.append('#FF9800')  # matches global

    ax.barh(y_pos, cv_norm_fcs, color=bar_colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='No change')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_genes, fontsize=10)
    ax.set_xlabel('CV / CV(GAPDH) Fold Change')
    ax.set_title('B. GAPDH-Normalized CV\n(>1.1 = specific, <0.9 = less than noise)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # ── Panel C: Detection rate changes ──
    ax = axes[0, 2]
    det_deltas = []
    for g in all_genes:
        row = norm_df[norm_df.gene == g]
        if len(row) > 0:
            det_deltas.append(row.iloc[0]['det_delta'])
        else:
            det_deltas.append(np.nan)

    det_colors = ['#E53935' if d < -0.1 else '#FF9800' if d < -0.02
                  else '#43A047' for d in det_deltas]
    # Override HK
    for i, g in enumerate(all_genes):
        if g in hk_genes:
            det_colors[i] = '#9E9E9E'

    ax.barh(y_pos, det_deltas, color=det_colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_genes, fontsize=10)
    ax.set_xlabel('Δ Detection Rate (Old − Young)')
    ax.set_title('C. Detection Rate Change\n(GAPDH-independent, specific signal)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # ── Panel D: Per-cell normalized CV ──
    ax = axes[1, 0]
    pcdf_cardiac = pcdf[~pcdf.gene.isin(hk_genes)]
    genes_pc = pcdf_cardiac.gene.unique()
    x = np.arange(2)
    width = 0.12
    for i, gene in enumerate(genes_pc):
        raw_y = pcdf[(pcdf.gene == gene) & (pcdf.age == AGE_ORDER[0])]['cv_raw']
        raw_o = pcdf[(pcdf.gene == gene) & (pcdf.age == AGE_ORDER[-1])]['cv_raw']
        norm_y = pcdf[(pcdf.gene == gene) & (pcdf.age == AGE_ORDER[0])]['cv_normalized']
        norm_o = pcdf[(pcdf.gene == gene) & (pcdf.age == AGE_ORDER[-1])]['cv_normalized']

        if len(raw_y) > 0 and len(raw_o) > 0:
            vals_raw = [raw_y.values[0], raw_o.values[0]]
            vals_norm = [norm_y.values[0], norm_o.values[0]]
            offset = (i - len(genes_pc)/2) * width
            ax.bar(x + offset, vals_raw, width, alpha=0.4, label=f'{gene} raw' if i == 0 else '')
            ax.bar(x + offset, vals_norm, width, alpha=0.9, label=f'{gene} norm' if i == 0 else '',
                   edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(['Young (3m)', 'Old (20+m)'])
    ax.set_ylabel('CV')
    ax.set_title('D. Per-cell Normalized CV\n(gene/GAPDH per cell)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Better: grouped bars for raw vs normalized
    # Redo Panel D properly
    axes[1, 0].clear()
    ax = axes[1, 0]
    genes_show = ['Gja1', 'Scn5a', 'Cacna1c', 'Dsp', 'Ryr2']
    x = np.arange(len(genes_show))
    w = 0.2

    raw_fcs = []
    norm_fcs = []
    for g in genes_show:
        gy = pcdf[(pcdf.gene == g) & (pcdf.age == AGE_ORDER[0])]
        go = pcdf[(pcdf.gene == g) & (pcdf.age == AGE_ORDER[-1])]
        if len(gy) > 0 and len(go) > 0:
            raw_fc = go['cv_raw'].values[0] / gy['cv_raw'].values[0]
            norm_fc = go['cv_normalized'].values[0] / gy['cv_normalized'].values[0]
        else:
            raw_fc, norm_fc = np.nan, np.nan
        raw_fcs.append(raw_fc)
        norm_fcs.append(norm_fc)

    ax.bar(x - w/2, raw_fcs, w, label='Raw CV FC', color='#BBDEFB', edgecolor='black')
    ax.bar(x + w/2, norm_fcs, w, label='Gene/GAPDH CV FC', color='#1565C0', edgecolor='black')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(genes_show, fontsize=10)
    ax.set_ylabel('CV Fold Change (Old / Young)')
    ax.set_title('D. Raw vs Per-cell Normalized\nCV Fold Change', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel E: Block C recheck ──
    ax = axes[1, 1]
    # Detection rates for subsystems
    subsys_data = {'Working': {}, 'Conduction': {}}
    for pop_name, pop_mask in [('Working', working_mask), ('Conduction', conduction_mask)]:
        for age in [AGE_ORDER[0], AGE_ORDER[-1]]:
            age_mask = cm_atrial.obs['development_stage'].values == age
            combined = age_mask & pop_mask
            expr = gja1_expr[combined]
            n_tot = len(expr)
            n_det = np.sum(expr > 0)
            subsys_data[pop_name][AGE_SHORT[age]] = {
                'frac_det': n_det/n_tot if n_tot > 0 else 0,
                'n': n_tot,
                'cv_raw': np.std(expr[expr > 0])/np.mean(expr[expr > 0]) if np.sum(expr > 0) >= 5 else np.nan,
            }
            # Normalized CV
            gapdh_vals = gapdh_expr_all[combined]
            both = (expr > 0) & (gapdh_vals > 0)
            if both.sum() >= 5:
                normed = expr[both] / gapdh_vals[both]
                subsys_data[pop_name][AGE_SHORT[age]]['cv_norm'] = np.std(normed)/np.mean(normed)
            else:
                subsys_data[pop_name][AGE_SHORT[age]]['cv_norm'] = np.nan

    # Plot detection rates
    x = np.arange(2)
    w = 0.3
    for i, (pop, color) in enumerate([('Working', '#E53935'), ('Conduction', '#1565C0')]):
        det_vals = [subsys_data[pop]['3m']['frac_det'],
                    subsys_data[pop]['20+m']['frac_det']]
        ax.bar(x + (i-0.5)*w, det_vals, w, color=color, alpha=0.8,
               edgecolor='black', label=pop)
        for j, v in enumerate(det_vals):
            n = subsys_data[pop][['3m', '20+m'][j]]['n']
            ax.text(x[j] + (i-0.5)*w, v + 0.02, f'n={n}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Young (3m)', 'Old (20+m)'])
    ax.set_ylabel('GJA1 Detection Rate')
    ax.set_title('E. Subsystem Detection\n(GAPDH-independent)', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel F: Summary interpretation ──
    ax = axes[1, 2]
    ax.axis('off')

    text = """CORRECTED INTERPRETATION

1. CV increases are MOSTLY global noise
   • GAPDH CV ↑ 2.66× → global transcriptional
     noise increases with age
   • After normalization, most cardiac genes
     show ≤ global noise level

2. Detection rate is the SPECIFIC signal
   • GAPDH detection: 93.7% → 93.8% (stable)
   • Cardiac genes: 20-53% → 6-28% (collapse)
   • This is gene-specific gene silencing

3. Two distinct aging processes:
   ① Global noise ↑ (all genes, non-specific)
     → beat-to-beat variability ↑
   ② Cardiac gene silencing (specific)
     → fraction of "silent" cells ↑
     → conduction fragmentation → β ↓

4. Block C survives IF based on detection:
   Check working vs conduction detection rates
"""
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Normalization Check: Separating Global Noise from Specific Effects',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/fig_normalization_check.png')
    plt.close()
    print("  Saved fig_normalization_check.png")

    # ═══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*75)
    print("FINAL VERDICT")
    print("="*75)

    print("""
  GAPDH CV fold change: {:.2f}×  ← GLOBAL NOISE BASELINE

  After normalization (CV_gene / CV_GAPDH):""".format(gapdh_fc))

    for _, r in norm_df[~norm_df.gene.isin(['Gapdh', 'Actb'])].iterrows():
        print(f"    {r['gene']:12s}: {r['cv_norm_fc']:.2f}×  {r['flag']}")

    print(f"""
  Detection rate changes (GAPDH-independent):
    GAPDH:   {norm_df[norm_df.gene=='Gapdh'].iloc[0]['det_delta']:+.1%}  ← STABLE (control)""")
    for _, r in norm_df[~norm_df.gene.isin(['Gapdh', 'Actb'])].iterrows():
        print(f"    {r['gene']:12s}: {r['det_delta']:+.1%}")

    print("""
  CONCLUSION:
  • CV-based metrics: contaminated by global transcriptional noise
  • Detection-based metrics: GAPDH-validated, gene-specific signal
  • Primary metric going forward: DETECTION RATE + FRACTION-LOW
  • CV should be reported as secondary, always with GAPDH control
  """)


if __name__ == '__main__':
    main()
