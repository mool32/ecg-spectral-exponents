#!/usr/bin/env python3
"""
Entropy Anticorrelation in Cardiac Aging (ПАК — Principle of Anticorrelated Coherence)
======================================================================================
Tests whether heart tissue follows the universal anticorrelation:
  E_intra (intracellular Shannon entropy) ↓  with aging
  E_inter (intercellular heterogeneity)   ↑  with aging
  ρ(ΔE_intra, ΔE_inter) < 0 across cell types

Three analyses:
  1. Reproduce whole-Heart anticorrelation from existing TMS entropy data
  2. Compute E_intra/E_inter from our downloaded cardiomyocyte h5ad
  3. Decompose: cardiac genes vs housekeeping genes → separate anticorrelations

Data sources:
  - Existing: /Users/teo/Desktop/research/oscilatory/results/entropy/
  - Our data: results/tms_cardiomyocytes.h5ad, results/tms_fibroblasts.h5ad
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.spatial.distance import pdist
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

RESULTS_DIR = Path('results')
ENTROPY_DIR = Path('/Users/teo/Desktop/research/oscilatory/results/entropy')

# ──────────────────────────────────────────────────────────
# Gene sets
# ──────────────────────────────────────────────────────────
CARDIAC_GENES = [
    # Ion channels
    'Scn5a', 'Kcnj2', 'Kcnq1', 'Kcnh2', 'Cacna1c',
    # Gap junctions
    'Gja1', 'Gja5',
    # Calcium handling
    'Ryr2', 'Atp2a2', 'Casq2', 'Slc8a1',
    # Sarcomeric
    'Myh6', 'Myh7', 'Tnnt2', 'Tnni3', 'Actc1', 'Tpm1',
    # Transcription factors
    'Nkx2-5', 'Gata4', 'Tbx5', 'Mef2c',
    # Conduction
    'Hcn4', 'Hcn1',
]

HOUSEKEEPING_GENES = [
    # Classic housekeeping
    'Gapdh', 'Actb', 'Tbp', 'Hprt', 'Pgk1',
    'Rplp0', 'Rpl13a', 'Rps18', 'Rpl4', 'Rps3',
    'Ubc', 'B2m', 'Sdha', 'Ywhaz', 'Hmbs',
    'Tubb5', 'Tuba1a', 'Eef1a1', 'Eef2', 'Ppia',
    # Ribosomal (broadly expressed)
    'Rpl7', 'Rpl8', 'Rps5', 'Rps6', 'Rps12',
    'Rpl3', 'Rpl19', 'Rpl27', 'Rps15a', 'Rpl32',
]


# ══════════════════════════════════════════════════════════
# PART 1: Existing TMS whole-Heart anticorrelation
# ══════════════════════════════════════════════════════════

def part1_existing_heart_anticorrelation():
    """
    Reproduce the Heart anticorrelation from existing TMS entropy data.
    Uses E_intra (aggregated) and E_inter from the oscilatory project.
    """
    print("=" * 70)
    print("PART 1: Existing Heart anticorrelation (from TMS full pipeline)")
    print("=" * 70)

    # Use delta_scatter.csv from the existing pipeline (canonical methodology)
    delta_scatter_path = ENTROPY_DIR / 'analysis' / 'delta_scatter.csv'
    if not delta_scatter_path.exists():
        print("  ERROR: delta_scatter.csv not found")
        return None

    delta_all = pd.read_csv(delta_scatter_path)
    heart = delta_all[delta_all.tissue == 'Heart'].copy()

    print(f"\n  Heart cell types: {len(heart)}")
    print(f"  Cell types: {sorted(heart.cell_type.values)}")

    rho, p = stats.spearmanr(heart.delta_E_intra, heart.delta_E_inter)
    print(f"\n  *** ρ(ΔE_intra, ΔE_inter) = {rho:.4f}, p = {p:.4f} ***")

    # Direction check
    n_anti = sum(1 for _, r in heart.iterrows()
                 if r.delta_E_intra * r.delta_E_inter < 0)
    print(f"  Direction: {n_anti}/{len(heart)} cell types anticorrelated")

    print("\n  Per cell type:")
    for _, row in heart.iterrows():
        arrow_i = "↓" if row.delta_E_intra < 0 else "↑"
        arrow_e = "↓" if row.delta_E_inter < 0 else "↑"
        print(f"    {row.cell_type:40s}: ΔE_intra={row.delta_E_intra:+.4f}{arrow_i}, "
              f"ΔE_inter={row.delta_E_inter:+.4f}{arrow_e}")

    delta_df = heart.rename(columns={'delta_E_intra': 'delta_E_intra',
                                      'delta_E_inter': 'delta_E_inter'})
    return delta_df, rho, p


# ══════════════════════════════════════════════════════════
# PART 2: Fresh computation from our cardiomyocyte data
# ══════════════════════════════════════════════════════════

def compute_shannon_entropy(expression_vector):
    """Compute Shannon entropy of a gene expression vector."""
    expr = expression_vector[expression_vector > 0]
    if len(expr) <= 1:
        return 0.0
    total = expr.sum()
    if total == 0:
        return 0.0
    p = expr / total
    H = -np.sum(p * np.log2(p))
    return H


def compute_pielou(expression_vector):
    """Compute Pielou evenness (normalized Shannon entropy)."""
    expr = expression_vector[expression_vector > 0]
    if len(expr) <= 1:
        return 0.0
    total = expr.sum()
    if total == 0:
        return 0.0
    p = expr / total
    H = -np.sum(p * np.log2(p))
    H_max = np.log2(len(expr))
    return H / H_max if H_max > 0 else 0.0


def compute_E_inter_jsd(X_group, n_subsample=80, n_bootstrap=50, seed=42):
    """
    Compute intercellular entropy (mean pairwise JSD) for a group of cells.
    Adds Laplace smoothing (pseudocount) for sparse gene sets.
    """
    rng = np.random.RandomState(seed)
    n_cells = X_group.shape[0]
    n_sub = min(n_subsample, n_cells)

    # Add Laplace pseudocount for sparse gene sets to avoid zero-division
    pseudocount = 1e-6

    jsd_means = []
    for b in range(n_bootstrap):
        idx = rng.choice(n_cells, n_sub, replace=(n_cells < n_sub))
        sub = X_group[idx].astype(np.float64) + pseudocount

        # Normalize to probabilities
        row_sums = sub.sum(axis=1, keepdims=True)
        P = sub / row_sums

        try:
            jsd_dists = pdist(P, metric='jensenshannon')
            jsd_sq = jsd_dists ** 2  # jensenshannon returns sqrt(JSD)
            # Filter out inf/nan
            jsd_sq = jsd_sq[np.isfinite(jsd_sq)]
            if len(jsd_sq) > 0:
                jsd_means.append(float(np.mean(jsd_sq)))
        except Exception:
            pass

    if not jsd_means:
        return np.nan, np.nan, np.nan

    return float(np.mean(jsd_means)), float(np.percentile(jsd_means, 2.5)), \
           float(np.percentile(jsd_means, 97.5))


def part2_cardiomyocyte_entropy():
    """
    Compute E_intra and E_inter from our downloaded TMS cardiac data.
    Separate analysis for cardiac genes vs housekeeping genes.
    """
    print("\n" + "=" * 70)
    print("PART 2: Fresh entropy computation from cardiac h5ad data")
    print("=" * 70)

    # Load both cell types
    cm_path = RESULTS_DIR / 'tms_cardiomyocytes.h5ad'
    fb_path = RESULTS_DIR / 'tms_fibroblasts.h5ad'

    if not cm_path.exists():
        print("  ERROR: tms_cardiomyocytes.h5ad not found")
        return None

    cm = sc.read_h5ad(cm_path)
    fb = sc.read_h5ad(fb_path)

    # Build gene name mapping
    cm_gmap = dict(zip(cm.var['feature_name'], cm.var.index))
    fb_gmap = dict(zip(fb.var['feature_name'], fb.var.index))

    print(f"\n  Cardiomyocytes: {cm.shape}")
    print(f"  Fibroblasts: {fb.shape}")

    # Find available cardiac and housekeeping genes
    cardiac_avail_cm = [g for g in CARDIAC_GENES if g in cm_gmap]
    hk_avail_cm = [g for g in HOUSEKEEPING_GENES if g in cm_gmap]
    cardiac_avail_fb = [g for g in CARDIAC_GENES if g in fb_gmap]
    hk_avail_fb = [g for g in HOUSEKEEPING_GENES if g in fb_gmap]

    print(f"\n  CM: {len(cardiac_avail_cm)}/{len(CARDIAC_GENES)} cardiac genes, "
          f"{len(hk_avail_cm)}/{len(HOUSEKEEPING_GENES)} HK genes")
    print(f"  FB: {len(cardiac_avail_fb)}/{len(CARDIAC_GENES)} cardiac genes, "
          f"{len(hk_avail_fb)}/{len(HOUSEKEEPING_GENES)} HK genes")

    results = {}

    # Determine age column name
    AGE_COL = 'development_stage'
    AGE_ORDER_MAP = {
        '3-month-old stage': 0,
        '18-month-old stage': 1,
        '20-month-old stage and over': 2,
    }
    AGE_SHORT_MAP = {
        '3-month-old stage': '3m',
        '18-month-old stage': '18m',
        '20-month-old stage and over': '20+m',
    }

    for label, adata, gmap, cardiac_g, hk_g in [
        ('Cardiomyocyte', cm, cm_gmap, cardiac_avail_cm, hk_avail_cm),
        ('Fibroblast', fb, fb_gmap, cardiac_avail_fb, hk_avail_fb),
    ]:
        print(f"\n  --- {label} ---")

        # Get age column
        age_col = None
        for candidate in ['development_stage', 'age', 'age_months']:
            if candidate in adata.obs.columns:
                age_col = candidate
                break
        if age_col is None:
            print(f"    ERROR: no age column found. Available: {list(adata.obs.columns)}")
            continue

        ages = sorted(adata.obs[age_col].unique(), key=lambda x: AGE_ORDER_MAP.get(str(x), 99))
        print(f"    Ages: {ages}")

        # Get expression matrices for gene subsets
        cardiac_idx = [gmap[g] for g in cardiac_g]
        hk_idx = [gmap[g] for g in hk_g]

        # All genes for full E_intra
        X_full = adata.X
        if hasattr(X_full, 'toarray'):
            X_full = X_full.toarray()

        # Subset matrices
        cardiac_col_idx = [list(adata.var.index).index(idx) for idx in cardiac_idx]
        hk_col_idx = [list(adata.var.index).index(idx) for idx in hk_idx]

        X_cardiac = X_full[:, cardiac_col_idx]
        X_hk = X_full[:, hk_col_idx]

        n_cells = X_full.shape[0]

        # Compute E_intra per cell for each gene set
        E_intra_full = np.array([compute_shannon_entropy(X_full[i]) for i in range(n_cells)])
        E_intra_cardiac = np.array([compute_shannon_entropy(X_cardiac[i]) for i in range(n_cells)])
        E_intra_hk = np.array([compute_shannon_entropy(X_hk[i]) for i in range(n_cells)])

        # Pielou evenness
        J_full = np.array([compute_pielou(X_full[i]) for i in range(n_cells)])
        J_cardiac = np.array([compute_pielou(X_cardiac[i]) for i in range(n_cells)])
        J_hk = np.array([compute_pielou(X_hk[i]) for i in range(n_cells)])

        # Per-age statistics
        age_results = []
        for age in ages:
            mask = (adata.obs[age_col] == age).values
            n = mask.sum()

            # E_intra means
            entry = {
                'cell_type': label,
                'age': age,
                'n_cells': n,
                'E_intra_full': np.mean(E_intra_full[mask]),
                'E_intra_cardiac': np.mean(E_intra_cardiac[mask]),
                'E_intra_hk': np.mean(E_intra_hk[mask]),
                'J_full': np.mean(J_full[mask]),
                'J_cardiac': np.mean(J_cardiac[mask]),
                'J_hk': np.mean(J_hk[mask]),
            }

            # E_inter (JSD) for each gene set
            X_age_full = X_full[mask]
            X_age_cardiac = X_cardiac[mask]
            X_age_hk = X_hk[mask]

            if n >= 10:
                entry['E_inter_full'], _, _ = compute_E_inter_jsd(X_age_full, seed=42)
                entry['E_inter_cardiac'], _, _ = compute_E_inter_jsd(X_age_cardiac, seed=42)
                entry['E_inter_hk'], _, _ = compute_E_inter_jsd(X_age_hk, seed=42)
            else:
                entry['E_inter_full'] = np.nan
                entry['E_inter_cardiac'] = np.nan
                entry['E_inter_hk'] = np.nan

            age_results.append(entry)
            print(f"    Age {str(age)[:30]:30s}: n={n:4d}, "
                  f"E_intra_full={entry['E_intra_full']:.3f}, "
                  f"E_intra_cardiac={entry['E_intra_cardiac']:.3f}, "
                  f"E_intra_hk={entry['E_intra_hk']:.3f}")
            print(f"      {'':30s}  E_inter_full={entry.get('E_inter_full', np.nan):.4f}, "
                  f"cardiac={entry.get('E_inter_cardiac', np.nan):.4f}, "
                  f"hk={entry.get('E_inter_hk', np.nan):.4f}")

        results[label] = pd.DataFrame(age_results)

    return results


# ══════════════════════════════════════════════════════════
# PART 3: Cross-cell-type anticorrelation (the ПАК test)
# ══════════════════════════════════════════════════════════

def part3_anticorrelation_test(results_dict):
    """
    Compute ρ(ΔE_intra, ΔE_inter) across cell types.
    The ПАК prediction: ρ < 0.
    """
    print("\n" + "=" * 70)
    print("PART 3: Anticorrelation test (ПАК)")
    print("=" * 70)

    if results_dict is None:
        print("  No results to test")
        return None

    # For each gene set, compute Δ(old - young)
    AGE_ORDER_MAP = {
        '3-month-old stage': 0,
        '18-month-old stage': 1,
        '20-month-old stage and over': 2,
    }
    age_key_young = None
    age_key_old = None

    # Determine young/old ages from data
    for ct_label, df in results_dict.items():
        ages = sorted(df.age.unique(), key=lambda x: AGE_ORDER_MAP.get(str(x), 99))
        if len(ages) >= 2:
            age_key_young = ages[0]   # youngest
            age_key_old = ages[-1]    # oldest
            break

    if age_key_young is None:
        print("  Cannot determine age groups")
        return None

    print(f"\n  Young: {age_key_young}")
    print(f"  Old: {age_key_old}")

    deltas_all = []
    for ct_label, df in results_dict.items():
        young = df[df.age == age_key_young]
        old = df[df.age == age_key_old]

        if len(young) == 0 or len(old) == 0:
            continue

        for gene_set, intra_col, inter_col in [
            ('full', 'E_intra_full', 'E_inter_full'),
            ('cardiac', 'E_intra_cardiac', 'E_inter_cardiac'),
            ('housekeeping', 'E_intra_hk', 'E_inter_hk'),
        ]:
            d_intra = old[intra_col].values[0] - young[intra_col].values[0]
            d_inter = old[inter_col].values[0] - young[inter_col].values[0]

            deltas_all.append({
                'cell_type': ct_label,
                'gene_set': gene_set,
                'delta_E_intra': d_intra,
                'delta_E_inter': d_inter,
            })

    delta_df = pd.DataFrame(deltas_all)

    print("\n  Deltas (old - young):")
    for _, row in delta_df.iterrows():
        print(f"    {row.cell_type:15s} [{row.gene_set:12s}]: "
              f"ΔE_intra = {row.delta_E_intra:+.4f}, "
              f"ΔE_inter = {row.delta_E_inter:+.4f}")

    # With only 2 cell types, Spearman ρ is trivial (±1 or 0)
    # But we can still report the direction
    for gs in ['full', 'cardiac', 'housekeeping']:
        sub = delta_df[delta_df.gene_set == gs]
        if len(sub) >= 2:
            # Even with 2 points, report the sign
            signs_match = (sub.delta_E_intra.values[0] * sub.delta_E_inter.values[0]) < 0
            print(f"\n  Gene set '{gs}':")
            print(f"    N cell types: {len(sub)}")
            if len(sub) >= 3:
                rho, p = stats.spearmanr(sub.delta_E_intra, sub.delta_E_inter)
                print(f"    ρ(ΔE_intra, ΔE_inter) = {rho:.4f}, p = {p:.4f}")
            else:
                # Pearson for 2 points
                r = np.corrcoef(sub.delta_E_intra.values, sub.delta_E_inter.values)[0, 1]
                print(f"    r(ΔE_intra, ΔE_inter) = {r:.4f} (only {len(sub)} points)")
                for _, row in sub.iterrows():
                    sign_intra = "↓" if row.delta_E_intra < 0 else "↑"
                    sign_inter = "↓" if row.delta_E_inter < 0 else "↑"
                    anticorr = "✓" if (row.delta_E_intra * row.delta_E_inter) < 0 else "✗"
                    print(f"      {row.cell_type}: E_intra{sign_intra} E_inter{sign_inter} {anticorr}")

    return delta_df


# ══════════════════════════════════════════════════════════
# PART 4: Combined figure
# ══════════════════════════════════════════════════════════

def make_figure(part1_data, part2_results, delta_df):
    """Generate combined entropy anticorrelation figure."""
    print("\n" + "=" * 70)
    print("PART 4: Generating figure")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Panel A: Existing Heart ΔE_intra vs ΔE_inter ──
    ax = axes[0, 0]
    if part1_data is not None:
        p1_df, rho1, p1 = part1_data
        ax.scatter(p1_df.delta_E_intra, p1_df.delta_E_inter,
                   s=80, c='#E53935', zorder=5, edgecolors='k', linewidth=0.5)
        for _, row in p1_df.iterrows():
            ax.annotate(row.cell_type.replace(' of cardiac tissue', '').replace('endothelial cell of coronary artery', 'coronary EC'),
                        (row.delta_E_intra, row.delta_E_inter),
                        fontsize=7, ha='center', va='bottom', xytext=(0, 5),
                        textcoords='offset points')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('ΔE_intra (old − young)')
        ax.set_ylabel('ΔE_inter (old − young)')
        ax.set_title(f'A. Heart ΔE anticorrelation (TMS)\n'
                     f'ρ = {rho1:.3f}, p = {p1:.3f}, n = {len(p1_df)} cell types',
                     fontweight='bold')
    ax.grid(alpha=0.3)

    # ── Panel B: E_intra trajectories by age (CM) ──
    ax = axes[0, 1]
    if part2_results and 'Cardiomyocyte' in part2_results:
        cm_df = part2_results['Cardiomyocyte']
        # Map age labels to numeric for plotting
        age_numeric = {age: i for i, age in enumerate(sorted(cm_df.age.unique(), key=str))}
        x = [age_numeric[a] for a in cm_df.age]
        age_labels = [str(a)[:20] for a in sorted(cm_df.age.unique(), key=str)]

        for col, color, label in [
            ('E_intra_full', '#1565C0', 'All genes'),
            ('E_intra_cardiac', '#E53935', 'Cardiac genes'),
            ('E_intra_hk', '#43A047', 'Housekeeping genes'),
        ]:
            ax.plot(range(len(age_labels)), cm_df.sort_values('age', key=lambda x: x.astype(str))[col].values,
                    'o-', color=color, linewidth=2, markersize=8, label=label)

        ax.set_xticks(range(len(age_labels)))
        ax.set_xticklabels(age_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('E_intra (Shannon entropy)')
        ax.set_title('B. Cardiomyocyte E_intra by age', fontweight='bold')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel C: E_intra trajectories (FB) ──
    ax = axes[0, 2]
    if part2_results and 'Fibroblast' in part2_results:
        fb_df = part2_results['Fibroblast']
        age_labels_fb = [str(a)[:20] for a in sorted(fb_df.age.unique(), key=str)]

        for col, color, label in [
            ('E_intra_full', '#1565C0', 'All genes'),
            ('E_intra_cardiac', '#E53935', 'Cardiac genes'),
            ('E_intra_hk', '#43A047', 'Housekeeping genes'),
        ]:
            ax.plot(range(len(age_labels_fb)),
                    fb_df.sort_values('age', key=lambda x: x.astype(str))[col].values,
                    'o-', color=color, linewidth=2, markersize=8, label=label)

        ax.set_xticks(range(len(age_labels_fb)))
        ax.set_xticklabels(age_labels_fb, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('E_intra (Shannon entropy)')
        ax.set_title('C. Fibroblast E_intra by age', fontweight='bold')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel D: E_inter trajectories (CM) ──
    ax = axes[1, 0]
    if part2_results and 'Cardiomyocyte' in part2_results:
        cm_df = part2_results['Cardiomyocyte']
        age_labels_cm = [str(a)[:20] for a in sorted(cm_df.age.unique(), key=str)]

        for col, color, label in [
            ('E_inter_full', '#1565C0', 'All genes'),
            ('E_inter_cardiac', '#E53935', 'Cardiac genes'),
            ('E_inter_hk', '#43A047', 'Housekeeping genes'),
        ]:
            vals = cm_df.sort_values('age', key=lambda x: x.astype(str))[col].values
            ax.plot(range(len(age_labels_cm)), vals,
                    's--', color=color, linewidth=2, markersize=8, label=label)

        ax.set_xticks(range(len(age_labels_cm)))
        ax.set_xticklabels(age_labels_cm, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('E_inter (mean JSD)')
        ax.set_title('D. Cardiomyocyte E_inter by age', fontweight='bold')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel E: E_inter trajectories (FB) ──
    ax = axes[1, 1]
    if part2_results and 'Fibroblast' in part2_results:
        fb_df = part2_results['Fibroblast']
        age_labels_fb = [str(a)[:20] for a in sorted(fb_df.age.unique(), key=str)]

        for col, color, label in [
            ('E_inter_full', '#1565C0', 'All genes'),
            ('E_inter_cardiac', '#E53935', 'Cardiac genes'),
            ('E_inter_hk', '#43A047', 'Housekeeping genes'),
        ]:
            vals = fb_df.sort_values('age', key=lambda x: x.astype(str))[col].values
            ax.plot(range(len(age_labels_fb)), vals,
                    's--', color=color, linewidth=2, markersize=8, label=label)

        ax.set_xticks(range(len(age_labels_fb)))
        ax.set_xticklabels(age_labels_fb, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('E_inter (mean JSD)')
        ax.set_title('E. Fibroblast E_inter by age', fontweight='bold')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel F: Summary anticorrelation table ──
    ax = axes[1, 2]
    ax.axis('off')

    # Prepare summary text
    lines = [
        "Anticorrelation Summary (ПАК)",
        "─" * 40,
        "",
        "Universal principle:",
        "  E_intra ↓ + E_inter ↑ with aging",
        "  ρ(ΔE_intra, ΔE_inter) < 0",
        "",
        "─" * 40,
    ]

    if part1_data is not None:
        p1_df, rho1, p1 = part1_data
        lines.append(f"TMS Heart (6 cell types):")
        lines.append(f"  ρ = {rho1:.3f}, p = {p1:.3f}")
        lines.append("")

    if delta_df is not None:
        lines.append("Per gene set (CM + FB):")
        for gs in ['full', 'cardiac', 'housekeeping']:
            sub = delta_df[delta_df.gene_set == gs]
            if len(sub) >= 2:
                # Check direction consistency
                consistent = sum(1 for _, r in sub.iterrows()
                                 if r.delta_E_intra * r.delta_E_inter < 0)
                lines.append(f"  {gs:12s}: {consistent}/{len(sub)} anticorrelated")

    lines.extend([
        "",
        "─" * 40,
        "Reference ρ values:",
        "  Mouse TMS (all tissues): ρ ≈ −0.54",
        "  Human CZI (all tissues): ρ ≈ −0.55",
        "  TMS Heart (6 ct):        ρ = −0.486",
    ])

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Entropy Anticorrelation in Cardiac Aging (ПАК)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'fig_entropy_anticorrelation.png')
    plt.close()
    print("  Saved fig_entropy_anticorrelation.png")


# ══════════════════════════════════════════════════════════
# PART 5: Detailed per-cell entropy distributions
# ══════════════════════════════════════════════════════════

def part5_distributions(part2_results):
    """
    Plot per-cell E_intra distributions for cardiac vs housekeeping genes,
    split by age. This shows the cellular-level entropy shift.
    """
    print("\n" + "=" * 70)
    print("PART 5: Per-cell entropy distributions")
    print("=" * 70)

    if part2_results is None:
        return

    # Reload data to get per-cell values
    cm = sc.read_h5ad(RESULTS_DIR / 'tms_cardiomyocytes.h5ad')
    cm_gmap = dict(zip(cm.var['feature_name'], cm.var.index))

    cardiac_avail = [g for g in CARDIAC_GENES if g in cm_gmap]
    hk_avail = [g for g in HOUSEKEEPING_GENES if g in cm_gmap]

    X = cm.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    cardiac_col_idx = [list(cm.var.index).index(cm_gmap[g]) for g in cardiac_avail]
    hk_col_idx = [list(cm.var.index).index(cm_gmap[g]) for g in hk_avail]

    X_cardiac = X[:, cardiac_col_idx]
    X_hk = X[:, hk_col_idx]

    n_cells = X.shape[0]
    E_cardiac = np.array([compute_shannon_entropy(X_cardiac[i]) for i in range(n_cells)])
    E_hk = np.array([compute_shannon_entropy(X_hk[i]) for i in range(n_cells)])

    cm.obs['E_intra_cardiac'] = E_cardiac
    cm.obs['E_intra_hk'] = E_hk

    # Determine age groups
    age_col = 'development_stage' if 'development_stage' in cm.obs.columns else 'age'
    AGE_ORDER_MAP = {
        '3-month-old stage': 0,
        '18-month-old stage': 1,
        '20-month-old stage and over': 2,
    }
    ages = sorted(cm.obs[age_col].unique(), key=lambda x: AGE_ORDER_MAP.get(str(x), 99))
    age_short = {a: str(a).replace('-month-old stage', 'm').replace(' and over', '+') for a in ages}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Cardiac gene entropy by age
    ax = axes[0]
    for age, color in zip(ages, ['#2196F3', '#FF9800', '#E53935']):
        mask = cm.obs[age_col] == age
        vals = E_cardiac[mask.values]
        ax.hist(vals, bins=40, alpha=0.5, color=color,
                label=f'{age_short[age]} (μ={np.mean(vals):.2f})', density=True)
    ax.set_xlabel('E_intra (cardiac genes)')
    ax.set_ylabel('Density')
    ax.set_title('A. Cardiac gene entropy', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: Housekeeping gene entropy by age
    ax = axes[1]
    for age, color in zip(ages, ['#2196F3', '#FF9800', '#E53935']):
        mask = cm.obs[age_col] == age
        vals = E_hk[mask.values]
        ax.hist(vals, bins=40, alpha=0.5, color=color,
                label=f'{age_short[age]} (μ={np.mean(vals):.2f})', density=True)
    ax.set_xlabel('E_intra (housekeeping genes)')
    ax.set_ylabel('Density')
    ax.set_title('B. Housekeeping gene entropy', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: Cardiac vs HK entropy scatter (per cell)
    ax = axes[2]
    for age, color in zip(ages, ['#2196F3', '#FF9800', '#E53935']):
        mask = cm.obs[age_col] == age
        ax.scatter(E_hk[mask.values], E_cardiac[mask.values],
                   alpha=0.1, s=5, c=color, label=age_short[age])

    # Per-age correlation
    for age, color in zip(ages, ['#2196F3', '#FF9800', '#E53935']):
        mask = cm.obs[age_col] == age
        rho, p = stats.spearmanr(E_hk[mask.values], E_cardiac[mask.values])
        print(f"  {age_short[age]}: ρ(E_cardiac, E_hk) = {rho:.3f}, p = {p:.2e}")

    ax.set_xlabel('E_intra (housekeeping)')
    ax.set_ylabel('E_intra (cardiac)')
    ax.set_title('C. Cardiac vs HK entropy per cell', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle('Per-Cell Entropy Distributions: Cardiac vs Housekeeping',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'fig_entropy_distributions.png')
    plt.close()
    print("  Saved fig_entropy_distributions.png")


# ══════════════════════════════════════════════════════════
# PART 6: Print final summary table
# ══════════════════════════════════════════════════════════

def print_summary(part1_data, part2_results, delta_df):
    """Print the final summary for inclusion in the paper."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Entropy Anticorrelation in Heart")
    print("=" * 70)

    print("\n1. Existing TMS Heart (from oscillatory project):")
    if part1_data is not None:
        p1_df, rho1, p1 = part1_data
        print(f"   ρ(ΔE_intra, ΔE_inter) = {rho1:.4f}, p = {p1:.4f}")
        print(f"   N cell types = {len(p1_df)}")
        print(f"   Cell types: {', '.join(p1_df.cell_type.values)}")

    AGE_ORDER_MAP = {
        '3-month-old stage': 0,
        '18-month-old stage': 1,
        '20-month-old stage and over': 2,
    }
    print("\n2. Cardiac-specific entropy (our data):")
    if part2_results:
        for ct, df in part2_results.items():
            ages = sorted(df.age.unique(), key=lambda x: AGE_ORDER_MAP.get(str(x), 99))
            young = df[df.age == ages[0]]
            old = df[df.age == ages[-1]]
            if len(young) > 0 and len(old) > 0:
                print(f"\n   {ct}:")
                for gene_set, intra_col, inter_col in [
                    ('Full', 'E_intra_full', 'E_inter_full'),
                    ('Cardiac', 'E_intra_cardiac', 'E_inter_cardiac'),
                    ('Housekeeping', 'E_intra_hk', 'E_inter_hk'),
                ]:
                    d_intra = old[intra_col].values[0] - young[intra_col].values[0]
                    d_inter = old[inter_col].values[0] - young[inter_col].values[0]
                    sign = "ANTI" if d_intra * d_inter < 0 else "CORR"
                    arrow_i = "↓" if d_intra < 0 else "↑"
                    arrow_e = "↓" if d_inter < 0 else "↑"
                    print(f"     {gene_set:12s}: E_intra{arrow_i}({d_intra:+.3f}) "
                          f"E_inter{arrow_e}({d_inter:+.4f}) → {sign}")

    print("\n3. Reference table:")
    print("   ┌──────────────────────────────┬─────────┬────────────┐")
    print("   │ Dataset                      │    ρ    │ N cell types│")
    print("   ├──────────────────────────────┼─────────┼────────────┤")
    print("   │ Mouse TMS (all tissues)      │  −0.54  │    ~80     │")
    print("   │ Human CZI (all tissues)      │  −0.55  │    ~60     │")
    if part1_data is not None:
        _, rho1, _ = part1_data
        print(f"   │ TMS Heart (all cell types)   │  {rho1:+.3f} │     6      │")
    print("   └──────────────────────────────┴─────────┴────────────┘")

    # Save results
    all_results = []
    if part2_results:
        for ct, df in part2_results.items():
            for _, row in df.iterrows():
                all_results.append(row.to_dict())
    if all_results:
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'entropy_cardiac_results.csv', index=False)
        print("\n  Saved entropy_cardiac_results.csv")


def main():
    print("=" * 70)
    print("ENTROPY ANTICORRELATION IN CARDIAC AGING")
    print("=" * 70)

    # Part 1: Existing data
    part1_data = part1_existing_heart_anticorrelation()

    # Part 2: Fresh computation
    part2_results = part2_cardiomyocyte_entropy()

    # Part 3: Anticorrelation test
    delta_df = part3_anticorrelation_test(part2_results)

    # Part 4: Figure
    make_figure(part1_data, part2_results, delta_df)

    # Part 5: Distributions
    part5_distributions(part2_results)

    # Part 6: Summary
    print_summary(part1_data, part2_results, delta_df)


if __name__ == '__main__':
    main()
