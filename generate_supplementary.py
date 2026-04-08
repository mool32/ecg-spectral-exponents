"""
Generate supplementary tables and figures for the preprint.
"""
import numpy as np, pandas as pd, ast, warnings, os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

OUT = 'paper/figures'
os.makedirs(OUT, exist_ok=True)

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# ── Load data ──
beta_df = pd.read_csv('results/beta_features.csv')
meta = pd.read_csv('ptb-xl/ptbxl_database.csv', index_col='ecg_id')
scp_map = pd.read_csv('ptb-xl/scp_statements.csv', index_col=0)
meta['scp_dict'] = meta.scp_codes.apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else {})

def get_superclass(d):
    out = set()
    for c, v in d.items():
        if c in scp_map.index:
            dc = scp_map.loc[c, 'diagnostic_class']
            if pd.notna(dc): out.add(dc)
    return list(out)

meta['superclasses'] = meta.scp_dict.apply(get_superclass)
meta['clean_superclass'] = meta.superclasses.apply(lambda x: x[0] if len(x)==1 else None)
df = meta.join(beta_df.set_index('ecg_id'), how='inner')
df = df[df.beta_mean.notna()].copy()

r2_cols = [f'r2_ir_{l}' for l in LEAD_NAMES]
lead_betas = [f'beta_ir_{l}' for l in LEAD_NAMES]

norm = df[(df.clean_superclass=='NORM')&(df.age>=18)&(df.age<=95)].copy()

# ================================================================
# TABLE S1: Pairwise superclass comparisons
# ================================================================
print("=== TABLE S1 ===")
classes = ['NORM','MI','STTC','HYP','CD']
rows = []
for i, c1 in enumerate(classes):
    for j, c2 in enumerate(classes):
        if j <= i: continue
        v1 = df[df.clean_superclass==c1].beta_mean.dropna()
        v2 = df[df.clean_superclass==c2].beta_mean.dropna()
        u, p = stats.mannwhitneyu(v1, v2)
        pooled = np.sqrt(((len(v1)-1)*v1.std()**2+(len(v2)-1)*v2.std()**2)/(len(v1)+len(v2)-2))
        d = (v2.mean()-v1.mean())/pooled if pooled>0 else 0
        p_bonf = min(p * 10, 1.0)  # 10 comparisons
        rows.append({'Comparison': f'{c1} vs {c2}', 'n1': len(v1), 'n2': len(v2),
                     'mean1': f'{v1.mean():.3f}', 'mean2': f'{v2.mean():.3f}',
                     'Cohen_d': f'{d:.3f}', 'p_raw': f'{p:.2e}', 'p_Bonferroni': f'{p_bonf:.2e}'})
tab_s1 = pd.DataFrame(rows)
tab_s1.to_csv(f'{OUT}/table_s1_pairwise.csv', index=False)
print(tab_s1.to_string(index=False))

# ================================================================
# TABLE S2: Per-subtype CD classification
# ================================================================
print("\n=== TABLE S2 ===")
cd_subtypes = ['CLBBB','CRBBB','IRBBB','LAFB','1AVB','IVCD']
rows2 = []
norm_v = df[df.clean_superclass=='NORM'].beta_mean
for sub in cd_subtypes:
    s = df[df.scp_dict.apply(lambda x: sub in x and x.get(sub,0)>=80)]
    if len(s) == 0: continue
    pooled = np.sqrt(((len(norm_v)-1)*norm_v.std()**2+(len(s)-1)*s.beta_mean.std()**2)/(len(norm_v)+len(s)-2))
    d = (s.beta_mean.mean()-norm_v.mean())/pooled
    rows2.append({
        'Subtype': sub, 'n': len(s),
        'beta_mean': f'{s.beta_mean.mean():.3f}',
        'beta_SD': f'{s.beta_mean.std():.3f}',
        'Cohen_d_vs_NORM': f'{d:.3f}',
    })
tab_s2 = pd.DataFrame(rows2)
tab_s2.to_csv(f'{OUT}/table_s2_cd_subtypes.csv', index=False)
print(tab_s2.to_string(index=False))

# ================================================================
# TABLE S3: Segmented regression parameters
# ================================================================
print("\n=== TABLE S3 ===")
metrics = [('beta_mean', 'β_mean'), ('beta_std', 'σ_β')]
for col, label in metrics:
    if col not in norm.columns: continue
    age_v = norm.age.values.astype(float)
    y_v = norm[col].values.astype(float)
    valid = np.isfinite(age_v) & np.isfinite(y_v)
    age_v, y_v = age_v[valid], y_v[valid]

    def seg_rss(bp):
        l=age_v<=bp; r=age_v>bp
        if l.sum()<50 or r.sum()<50: return 1e10
        rss=0
        for m in [l,r]:
            s,i,_,_,_=stats.linregress(age_v[m],y_v[m])
            rss+=np.sum((y_v[m]-(i+s*age_v[m]))**2)
        return rss

    bpc = np.arange(30,65,0.5)
    rss_v = [seg_rss(b) for b in bpc]
    bp = bpc[np.argmin(rss_v)]

    lm=age_v<=bp; rm=age_v>bp
    sl,il,rl,pl,sel=stats.linregress(age_v[lm],y_v[lm])
    sr,ir,rr,pr,ser=stats.linregress(age_v[rm],y_v[rm])

    rss_full = seg_rss(bp)
    s_null,i_null,_,_,_ = stats.linregress(age_v, y_v)
    rss_null = np.sum((y_v - (i_null+s_null*age_v))**2)
    n = len(age_v); df_null = n-2; df_full = n-4
    F = ((rss_null-rss_full)/(df_null-df_full)) / (rss_full/df_full)
    from scipy.stats import f as fdist
    p_f = 1 - fdist.cdf(F, df_null-df_full, df_full)

    print(f'{label}: breakpoint={bp:.0f}y, slope_left={sl:.5f}, slope_right={sr:.5f}, F={F:.1f}, p={p_f:.2e}')

# ================================================================
# TABLE S5: Cross-dataset β by lead
# ================================================================
print("\n=== TABLE S5 ===")
# PTB-XL NORM medians per lead
ptb_norm = df[df.clean_superclass=='NORM']
print("PTB-XL NORM median β per lead:")
for l in LEAD_NAMES:
    print(f"  {l}: {ptb_norm[f'beta_ir_{l}'].median():.3f}")

# Load Chapman and CODE-15 data
try:
    chap = pd.read_csv('results/chapman_beta_features.csv')
    chap_norm = chap[chap.label=='NORM']
    print("\nChapman NORM median β per lead:")
    for l in LEAD_NAMES:
        col = f'beta_ir_{l}'
        if col in chap_norm.columns:
            print(f"  {l}: {chap_norm[col].median():.3f}")
except Exception as e:
    print(f"Chapman data not available: {e}")

try:
    code = pd.read_csv('results/code15_beta_features.csv')
    code_norm = code[code.label=='NORM'] if 'label' in code.columns else code[code.get('diagnosis','')=='NORM'] if 'diagnosis' in code.columns else None
    if code_norm is not None and len(code_norm) > 0:
        print("\nCODE-15 NORM median β per lead:")
        for l in LEAD_NAMES:
            col = f'beta_ir_{l}'
            if col in code_norm.columns:
                print(f"  {l}: {code_norm[col].median():.3f}")
except Exception as e:
    print(f"CODE data: {e}")

# ================================================================
# FIGURE S1: R² distribution
# ================================================================
print("\nGenerating Figure S1...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# A: R² histogram
ax = axes[0]
r2_all = df[r2_cols].values.flatten()
r2_all = r2_all[np.isfinite(r2_all)]
ax.hist(r2_all, bins=100, color='#3498db', alpha=0.7, edgecolor='white', lw=0.3)
ax.axvline(0.9, color='red', ls='--', lw=2, label='R² = 0.9')
ax.axvline(np.median(r2_all), color='black', ls='-', lw=2, label=f'Median = {np.median(r2_all):.3f}')
ax.set_xlabel('R² (IRASA fit quality)'); ax.set_ylabel('Count')
ax.set_title('a', fontweight='bold', loc='left', fontsize=14)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# B: R² by lead
ax = axes[1]
r2_by_lead = [df[c].dropna().values for c in r2_cols]
bp = ax.boxplot(r2_by_lead, labels=LEAD_NAMES, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db'); patch.set_alpha(0.5)
ax.axhline(0.9, color='red', ls='--', alpha=0.5)
ax.set_ylabel('R²'); ax.set_title('b', fontweight='bold', loc='left', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig_s1_r2_distribution.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_s1_r2_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig_s1 saved")

# ================================================================
# FIGURE S2: Sensitivity analyses
# ================================================================
print("Generating Figure S2...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# A: Effect of R² threshold
ax = axes[0]
thresholds = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
for thr in thresholds:
    sub = df[df[r2_cols].mean(axis=1) >= thr]
    n_v = sub[sub.clean_superclass=='NORM'].beta_mean
    c_v = sub[sub.clean_superclass=='CD'].beta_mean
    if len(n_v) > 10 and len(c_v) > 10:
        pooled = np.sqrt(((len(n_v)-1)*n_v.std()**2+(len(c_v)-1)*c_v.std()**2)/(len(n_v)+len(c_v)-2))
        d = (c_v.mean()-n_v.mean())/pooled
        ax.bar(str(thr), d, alpha=0.7, color='#E53935', edgecolor='black')
        ax.text(thresholds.index(thr), d+0.02, f'n={len(sub):,}', ha='center', fontsize=8)
ax.set_xlabel('Minimum mean R² threshold'); ax.set_ylabel("Cohen's d (CD vs NORM)")
ax.set_title('a   R² threshold sensitivity', fontweight='bold', loc='left', fontsize=14)
ax.grid(axis='y', alpha=0.3)

# B: Age-matched analysis
ax = axes[1]
norm_s = df[df.clean_superclass=='NORM']
cd_s = df[df.clean_superclass=='CD']
age_bins_match = [(20,40),(40,60),(60,80)]
for lo, hi in age_bins_match:
    n_v = norm_s[(norm_s.age>=lo)&(norm_s.age<hi)].beta_mean
    c_v = cd_s[(cd_s.age>=lo)&(cd_s.age<hi)].beta_mean
    if len(n_v)>10 and len(c_v)>10:
        pooled = np.sqrt(((len(n_v)-1)*n_v.std()**2+(len(c_v)-1)*c_v.std()**2)/(len(n_v)+len(c_v)-2))
        d = (c_v.mean()-n_v.mean())/pooled
        ax.bar(f'{lo}-{hi}', d, alpha=0.7, color='#E53935', edgecolor='black')
        ax.text(age_bins_match.index((lo,hi)), d+0.02, f'd={d:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Age group'); ax.set_ylabel("Cohen's d (CD vs NORM)")
ax.set_title('b   Age-matched effect sizes', fontweight='bold', loc='left', fontsize=14)
ax.grid(axis='y', alpha=0.3)

# C: Sex-stratified aging
ax = axes[2]
for sex_val, sex_label, color in [(0,'Male','#2196F3'),(1,'Female','#E91E63')]:
    s = norm[norm.sex==sex_val]
    bins = np.arange(20,91,5)
    s_copy = s.copy()
    s_copy['age_bin'] = pd.cut(s_copy.age, bins=bins)
    binned = s_copy.groupby('age_bin').agg(age_mid=('age','mean'),beta_avg=('beta_mean','mean'),
        beta_se=('beta_mean',lambda x: x.std()/np.sqrt(len(x)))).dropna()
    ax.errorbar(binned.age_mid, binned.beta_avg, yerr=binned.beta_se*1.96, fmt='o-',
                color=color, ms=5, lw=1.5, capsize=3, label=f'{sex_label} (n={len(s):,})')
    rho, p = stats.spearmanr(s.age, s.beta_mean)
    ax.text(0.95, 0.95 if sex_val==0 else 0.88, f'{sex_label}: ρ={rho:.3f}',
            transform=ax.transAxes, ha='right', fontsize=10, color=color, fontweight='bold')
ax.set_xlabel('Age (years)'); ax.set_ylabel(r'$\beta_\text{mean}$')
ax.set_title('c   Sex-stratified aging', fontweight='bold', loc='left', fontsize=14)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig_s2_sensitivity.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_s2_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig_s2 saved")

# ================================================================
# Demographics summary
# ================================================================
print("\n=== DEMOGRAPHICS TABLE ===")
valid_meta = meta[meta.age <= 120]
print(f"PTB-XL: N={len(meta)}, Male={int((meta.sex==0).sum())} ({(meta.sex==0).mean()*100:.1f}%), "
      f"Age={valid_meta.age.mean():.1f}±{valid_meta.age.std():.1f}, median={valid_meta.age.median():.0f}")
for sc in ['NORM','MI','STTC','CD','HYP']:
    sub = meta[meta.superclasses.apply(lambda x: sc in x)]
    sub_v = sub[sub.age<=120]
    print(f"  {sc}: n={len(sub)}, Male={int((sub.sex==0).sum())} ({(sub.sex==0).mean()*100:.1f}%), "
          f"Age={sub_v.age.mean():.1f}±{sub_v.age.std():.1f}")

print("\nDone!")
