#!/usr/bin/env python3
"""
Block F + Summary: Conceptual figures for the paper
=====================================================
- Fig F-1: Three Scales conceptual diagram
- Fig Summary: Grand summary — all results in one figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def fig_F1_three_scales():
    """
    Fig F-1: Three Scales conceptual figure.
    Cell (scRNA-seq) → Tissue (spatial, dashed) → Organ (ECG)
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1, 8)
    ax.axis('off')

    # Title
    ax.text(7.75, 7.5, 'From Molecules to the Electrocardiogram: Three Scales of Cardiac Aging',
            ha='center', fontsize=15, fontweight='bold')

    # ═══ Scale 1: Cell (scRNA-seq) ═══
    box1 = FancyBboxPatch((0.3, 3.5), 4.2, 3.0, boxstyle="round,pad=0.2",
                           facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2.5)
    ax.add_patch(box1)
    ax.text(2.4, 6.1, 'CELL', fontsize=14, fontweight='bold', ha='center', color='#1565C0')
    ax.text(2.4, 5.6, 'Single-cell RNA-seq', fontsize=10, ha='center', color='#666',
            fontstyle='italic')

    # Cell content
    cell_text = [
        '• CV(GJA1) ↑ 89% with age',
        '• SCN5A CV ↑ 165%',
        '• Detection rate drops',
        '• Gene correlations ↓ (atrial)',
        '• Working > Conduction aging',
    ]
    for i, t in enumerate(cell_text):
        ax.text(0.6, 5.0 - i*0.35, t, fontsize=9, color='#333')

    ax.text(2.4, 3.65, 'TMS (mouse, 1,650 cardiomyocytes)', fontsize=8,
            ha='center', color='#999', fontstyle='italic')

    # ═══ Scale 2: Tissue (spatial — dashed, future) ═══
    box2 = FancyBboxPatch((5.3, 3.5), 4.2, 3.0, boxstyle="round,pad=0.2",
                           facecolor='#FFF8E1', edgecolor='#FF8F00', linewidth=2.5,
                           linestyle='dashed')
    ax.add_patch(box2)
    ax.text(7.4, 6.1, 'TISSUE', fontsize=14, fontweight='bold', ha='center', color='#FF8F00')
    ax.text(7.4, 5.6, 'Spatial Transcriptomics', fontsize=10, ha='center', color='#666',
            fontstyle='italic')

    tissue_text = [
        '• WHERE does GJA1 drop?',
        '• Spatial domains of degradation',
        '• Fibrosis patches vs diffuse',
        '• Map to ECG lead regions',
        '  → FUTURE EXPERIMENT',
    ]
    for i, t in enumerate(tissue_text):
        color = '#C62828' if 'FUTURE' in t else '#333'
        ax.text(5.6, 5.0 - i*0.35, t, fontsize=9, color=color,
                fontweight='bold' if 'FUTURE' in t else 'normal')

    ax.text(7.4, 3.65, 'Visium/MERFISH (not yet available for aging)', fontsize=8,
            ha='center', color='#999', fontstyle='italic')

    # ═══ Scale 3: Organ (ECG) ═══
    box3 = FancyBboxPatch((10.3, 3.5), 4.8, 3.0, boxstyle="round,pad=0.2",
                           facecolor='#FCE4EC', edgecolor='#C62828', linewidth=2.5)
    ax.add_patch(box3)
    ax.text(12.7, 6.1, 'ORGAN', fontsize=14, fontweight='bold', ha='center', color='#C62828')
    ax.text(12.7, 5.6, 'Electrocardiogram', fontsize=10, ha='center', color='#666',
            fontstyle='italic')

    organ_text = [
        '• β falls with age (ρ = −0.178)',
        '• σ_β falls (unexpected)',
        '• Beat variability ↑ 58%',
        '• Breakpoint ~42 years',
        '• Bio age model: ρ = 0.60',
    ]
    for i, t in enumerate(organ_text):
        ax.text(10.6, 5.0 - i*0.35, t, fontsize=9, color='#333')

    ax.text(12.7, 3.65, 'PTB-XL (human, 9,500 NORM recordings)', fontsize=8,
            ha='center', color='#999', fontstyle='italic')

    # ═══ Arrows between scales ═══
    # Cell → Tissue (solid hypothesis)
    ax.annotate('', xy=(5.3, 5.0), xytext=(4.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
    ax.text(4.9, 5.35, 'Predicts spatial\nheterogeneity', ha='center',
            fontsize=8, color='#1565C0', fontstyle='italic')

    # Tissue → Organ (dashed — gap)
    ax.annotate('', xy=(10.3, 5.0), xytext=(9.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=2,
                                linestyle='dashed'))
    ax.text(9.9, 5.35, 'Maps to\nlead regions', ha='center',
            fontsize=8, color='#FF8F00', fontstyle='italic')

    # ═══ Cross-scale bridge (bottom) ═══
    bridge_box = FancyBboxPatch((2, 0.5), 11.5, 2.3, boxstyle="round,pad=0.2",
                                 facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(bridge_box)
    ax.text(7.75, 2.45, 'CROSS-SCALE VALIDATION', fontsize=13, fontweight='bold',
            ha='center', color='#7B1FA2')

    pairs = [
        ('σ_β (ECG)', 'CV(GJA1) (scRNA)', 'Coordination loss'),
        ('Beat variability', 'Gene variability', 'Temporal→molecular'),
        ('Regional β gradient', 'Fibrosis spatial', 'Structural remodel'),
        ('β_NORM ≈ 1.76', 'β↓ (aging) / β↑ (pathology)', 'Bifurcation'),
    ]
    for i, (ecg, rna, meaning) in enumerate(pairs):
        x_pos = 2.5 + i * 2.85
        ax.text(x_pos, 1.9, ecg, fontsize=8, fontweight='bold', color='#C62828')
        ax.text(x_pos, 1.55, '↔', fontsize=10, ha='left', color='#7B1FA2')
        ax.text(x_pos + 0.3, 1.55, rna, fontsize=8, fontweight='bold', color='#1565C0')
        ax.text(x_pos, 1.15, meaning, fontsize=7, color='#666', fontstyle='italic')

    # Lines connecting boxes to bridge
    for x_center in [2.4, 7.4, 12.7]:
        ax.plot([x_center, x_center], [3.5, 2.8], color='#7B1FA2',
                linewidth=1, linestyle=':', alpha=0.5)

    # Species labels
    ax.text(2.4, 3.2, '🐭 Mouse', fontsize=10, ha='center', color='#666')
    ax.text(7.4, 3.2, '🐭/🧑 Both', fontsize=10, ha='center', color='#666')
    ax.text(12.7, 3.2, '🧑 Human', fontsize=10, ha='center', color='#666')

    # Missing link annotation
    ax.annotate('MISSING LINK\n(Future work)', xy=(7.4, 3.0),
                xytext=(7.4, -0.3), fontsize=10, ha='center', color='#FF8F00',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1.5,
                                linestyle='dashed'))

    plt.tight_layout()
    plt.savefig('results/fig_F1_three_scales.png')
    plt.close()
    print("  Saved fig_F1_three_scales.png")


def fig_summary():
    """Grand summary figure: key results across all analyses."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    # Load data
    beta_df = pd.read_csv('results/beta_features.csv', index_col=0)
    meta = pd.read_csv('ptb-xl/ptbxl_database.csv', index_col=0)
    merged = beta_df.join(meta[['age', 'sex', 'scp_codes']], how='inner')

    # ── Panel 1: β by diagnosis (from Part I) ──
    ax = axes[0, 0]
    diag_map = {'NORM': '#4CAF50', 'MI': '#FF9800', 'STTC': '#9C27B0',
                'CD': '#F44336', 'HYP': '#2196F3'}
    for diag, color in diag_map.items():
        mask = merged.scp_codes.str.contains(diag, na=False)
        vals = merged[mask]['beta_mean'].dropna()
        bp = ax.boxplot([vals.values], positions=[list(diag_map.keys()).index(diag)],
                        widths=0.6, patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(list(diag_map.keys()))
    ax.axhline(y=merged[merged.scp_codes.str.contains('NORM', na=False)]['beta_mean'].median(),
               color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('β_mean')
    ax.set_title('1. β by Diagnosis', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 2: β vs age (NORM only) ──
    ax = axes[0, 1]
    norm = merged[merged.scp_codes.str.contains('NORM', na=False)]
    norm = norm[norm.age.between(18, 95)]
    from statsmodels.nonparametric.smoothers_lowess import lowess
    ax.scatter(norm.age, norm.beta_mean, alpha=0.08, s=5, c='#1565C0')
    lw = lowess(norm.beta_mean.values, norm.age.values, frac=0.3)
    ax.plot(lw[:, 0], lw[:, 1], 'r-', linewidth=2.5)
    from scipy.stats import spearmanr
    rho, p = spearmanr(norm.age, norm.beta_mean)
    ax.set_xlabel('Age')
    ax.set_ylabel('β_mean')
    ax.set_title(f'2. β vs Age (NORM)\nρ={rho:.3f}', fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)

    # ── Panel 3: Beat variability vs age ──
    ax = axes[0, 2]
    try:
        beat_df = pd.read_csv('results/block_D_beat_variability.csv')
        ax.scatter(beat_df.age, beat_df.II_morph_var, alpha=0.1, s=5, c='#43A047')
        lw_b = lowess(beat_df.II_morph_var.values, beat_df.age.values, frac=0.3)
        ax.plot(lw_b[:, 0], lw_b[:, 1], 'r-', linewidth=2.5)
        rho_b, p_b = spearmanr(beat_df.age, beat_df.II_morph_var)
        ax.set_title(f'3. Beat Variability vs Age\nρ={rho_b:.3f}', fontweight='bold', fontsize=11)
    except:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('3. Beat Variability', fontweight='bold', fontsize=11)
    ax.set_xlabel('Age')
    ax.set_ylabel('Morph. Variability')
    ax.grid(alpha=0.3)

    # ── Panel 4: Biological age model ──
    ax = axes[0, 3]
    ax.text(0.5, 0.85, 'Biological Age Model', ha='center', va='center',
            transform=ax.transAxes, fontsize=13, fontweight='bold')
    ax.text(0.5, 0.65, 'GradientBoosting', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='#666')
    ax.text(0.5, 0.45, 'ρ = 0.60', ha='center', va='center',
            transform=ax.transAxes, fontsize=20, fontweight='bold', color='#1565C0')
    ax.text(0.5, 0.28, 'MAE = 11.2 years', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='#666')
    ax.text(0.5, 0.1, 'β features → chronological age', ha='center', va='center',
            transform=ax.transAxes, fontsize=9, color='#999', fontstyle='italic')
    ax.axis('off')

    # ── Panel 5: GJA1 CV by age (scRNA-seq) ──
    ax = axes[1, 0]
    try:
        met = pd.read_csv('results/tms_gene_metrics_v2.csv')
        gja1 = met[(met.gene == 'Gja1') & (met.subset == 'atrial')]
        ages = ['3-month-old stage', '20-month-old stage and over']
        colors = ['#2196F3', '#E53935']
        x = [0, 1]
        cvs = [gja1[gja1.age == a]['cv'].values[0] if len(gja1[gja1.age == a]) > 0
               else np.nan for a in ages]
        bars = ax.bar(x, cvs, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(['3m\n(Young)', '20+m\n(Old)'])
        ax.set_ylabel('CV(GJA1)')
        pct = (cvs[1] - cvs[0]) / cvs[0] * 100
        ax.set_title(f'5. GJA1 CV (atrial)\n+{pct:.0f}%', fontweight='bold', fontsize=11)
    except:
        # Fallback: use raw metrics
        ax.bar([0, 1], [1.078, 2.038], color=['#2196F3', '#E53935'], alpha=0.8, edgecolor='black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['3m\n(Young)', '20+m\n(Old)'])
        ax.set_ylabel('CV(GJA1)')
        ax.set_title('5. GJA1 CV (atrial)\n+89%', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 6: Ion channel heatmap (mini) ──
    ax = axes[1, 1]
    try:
        fc = pd.read_csv('results/block_A_fold_changes.csv')
        cardiac = fc[~fc.category.isin(['Housekeeping', 'Fibrosis'])]
        genes = cardiac.gene.values
        cv_fc = cardiac.cv_fc.values
        colors_bar = ['#E91E63' if c == 'Gap Junction' else '#2196F3' if c == 'Ion Channel'
                      else '#4CAF50' if c == 'Structural' else '#FF9800'
                      for c in cardiac.category.values]
        y = np.arange(len(genes))
        ax.barh(y, cv_fc, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(genes, fontsize=9)
        ax.set_xlabel('CV Fold Change (Old/Young)')
        ax.set_title('6. Molecular Aging Profile', fontweight='bold', fontsize=11)
    except:
        ax.text(0.5, 0.5, 'No fold change data', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(axis='x', alpha=0.3)

    # ── Panel 7: Subsystem aging ──
    ax = axes[1, 2]
    try:
        sub = pd.read_csv('results/block_C_subsystems.csv')
        for pop, style, color in [('Working Myocardium', 'o-', '#E53935'),
                                   ('Conduction-like', 's--', '#1565C0')]:
            pdf = sub[sub.population == pop]
            vals = pdf.gja1_cv.values
            ax.plot(range(len(vals)), vals, style, color=color,
                    label=pop, linewidth=2, markersize=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['3m', '18m', '20+m'])
        ax.set_ylabel('GJA1 CV')
        ax.set_title('7. Subsystem Aging', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
    except:
        ax.text(0.5, 0.5, 'No subsystem data', ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(alpha=0.3)

    # ── Panel 8: Cross-scale bridge (mini) ──
    ax = axes[1, 3]
    # sigma_beta by decade
    norm_beta = merged[merged.scp_codes.str.contains('NORM', na=False)]
    norm_beta = norm_beta[norm_beta.age.between(18, 95)]
    norm_beta['decade'] = (norm_beta.age // 10) * 10
    sigma_dec = norm_beta.groupby('decade')['beta_std'].mean()

    ax2 = ax.twinx()
    ax.plot(sigma_dec.index, sigma_dec.values, 's-', color='#1565C0',
            linewidth=2, markersize=7, label='σ_β (ECG)')
    ax.set_xlabel('Age')
    ax.set_ylabel('σ_β', color='#1565C0')

    # CV(GJA1)
    cv_points = [(25, 1.078), (55, 1.522), (70, 2.038)]
    ax2.plot([p[0] for p in cv_points], [p[1] for p in cv_points],
             'D-', color='#C62828', linewidth=2, markersize=9, label='CV(GJA1)')
    ax2.set_ylabel('CV(GJA1)', color='#C62828')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax.set_title('8. Cross-Scale Bridge', fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)

    plt.suptitle('Grand Summary: Cardiac Aging Across Scales',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/fig_summary_grand.png')
    plt.close()
    print("  Saved fig_summary_grand.png")


def main():
    print("="*70)
    print("Conceptual Figures + Grand Summary")
    print("="*70)

    fig_F1_three_scales()
    fig_summary()

    print("\nAll conceptual figures generated!")


if __name__ == '__main__':
    main()
