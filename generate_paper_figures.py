"""
Generate all 5 main figures for the preprint.
Uses cached β-features + metadata.
"""
import numpy as np, pandas as pd, ast, warnings, os
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# ── Paths ──
OUT = 'paper/figures'
os.makedirs(OUT, exist_ok=True)

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
COLORS = {'NORM':'#4CAF50','MI':'#2196F3','STTC':'#FF9800','CD':'#E53935','HYP':'#9C27B0'}
CD_COLORS = {'CLBBB':'#E53935','CRBBB':'#1565C0','IRBBB':'#00BCD4',
             'LAFB':'#FF9800','1AVB':'#9C27B0','IVCD':'#607D8B'}

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

# Regional averages
for reg, leads in [('anterior',['V1','V2','V3','V4']),('lateral',['I','aVL','V5','V6']),('inferior',['II','III','aVF'])]:
    df[f'beta_{reg}'] = df[[f'beta_ir_{l}' for l in leads]].mean(axis=1)
df['beta_regional_div'] = df[['beta_anterior','beta_lateral','beta_inferior']].max(axis=1) - df[['beta_anterior','beta_lateral','beta_inferior']].min(axis=1)

lead_betas = [f'beta_ir_{l}' for l in LEAD_NAMES]
r2_cols = [f'r2_ir_{l}' for l in LEAD_NAMES]
df['beta_iqr'] = df[lead_betas].quantile(0.75,axis=1) - df[lead_betas].quantile(0.25,axis=1)
df['beta_cv'] = df['beta_std'] / df['beta_mean'].abs()
df['beta_skew'] = df[lead_betas].skew(axis=1)
df['beta_range'] = df[lead_betas].max(axis=1) - df[lead_betas].min(axis=1)
df['r2_std'] = df[r2_cols].std(axis=1)
df['sex_num'] = df['sex'].astype(float)
df['age_x_beta'] = df['age'] * df['beta_mean']
df['r2_mean'] = df[r2_cols].mean(axis=1)
df['delta'] = abs(df['beta_mean'] - 1.0)
df['beta_median'] = df[lead_betas].median(axis=1)

# SP columns if exist
sp_cols = [f'beta_sp_{l}' for l in LEAD_NAMES]
sp_cols = [c for c in sp_cols if c in df.columns]

norm = df[(df.clean_superclass=='NORM')&(df.age>=18)&(df.age<=95)].copy()

# CD subtypes
cd_subtypes_of_interest = ['CLBBB','CRBBB','LAFB','1AVB','IVCD','IRBBB']
cd_sub_list = []
for ecg_id, row in df.iterrows():
    codes = row['scp_dict']
    for sub in cd_subtypes_of_interest:
        if sub in codes and codes[sub] >= 80:
            cd_sub_list.append({'ecg_id': ecg_id, 'cd_subtype': sub})
            break
cd_sub_df = pd.DataFrame(cd_sub_list).drop_duplicates(subset='ecg_id').set_index('ecg_id')
df = df.join(cd_sub_df, how='left')
cd_data = df[df.cd_subtype.notna()].copy()

# Feature columns
feat_cols = ['age','sex_num','beta_mean','beta_std','beta_median','delta',
             'beta_anterior','beta_lateral','beta_inferior','beta_regional_div',
             'r2_mean','r2_std','beta_iqr','beta_cv','beta_skew','beta_range',
             'age_x_beta'] + lead_betas + r2_cols + sp_cols
if 'beta_sp_mean' in df.columns:
    feat_cols.append('beta_sp_mean')
avail = [f for f in feat_cols if f in df.columns]

print(f"Data loaded: {len(df)} records, {len(norm)} NORM, {len(cd_data)} CD subtypes")

# ================================================================
# FIGURE 1: β-Landscape
# ================================================================
print("Generating Figure 1...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
clean = df[df.clean_superclass.isin(['NORM','STTC','MI','HYP','CD'])]
order = ['NORM','STTC','MI','HYP','CD']

# A: Violins
ax = axes[0]
for i, cls in enumerate(order):
    vals = clean[clean.clean_superclass==cls].beta_mean.dropna()
    parts = ax.violinplot(vals, positions=[i], showmeans=False, showmedians=True, showextrema=False)
    for pc in parts['bodies']: pc.set_facecolor(COLORS[cls]); pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.text(i, ax.get_ylim()[0]+0.05 if i==0 else vals.min()-0.1,
            f'n={len(vals):,}', ha='center', fontsize=8, color='gray')
ax.axhline(norm.beta_mean.median(), color='gray', ls='--', alpha=0.5)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
ax.set_ylabel(r'$\beta$ (spectral exponent)'); ax.set_title('a', fontweight='bold', loc='left', fontsize=14)
ax.grid(axis='y', alpha=0.3)

# B: Cohen's d
ax = axes[1]
norm_v = clean[clean.clean_superclass=='NORM'].beta_mean.dropna()
n1, s1 = len(norm_v), norm_v.std()
es = []
for cls in ['MI','STTC','HYP','CD']:
    v = clean[clean.clean_superclass==cls].beta_mean.dropna()
    pooled = np.sqrt(((n1-1)*s1**2+(len(v)-1)*v.std()**2)/(n1+len(v)-2))
    d = (v.mean()-norm_v.mean())/pooled
    es.append({'class':cls,'d':d})
es_df = pd.DataFrame(es).sort_values('d')
ax.barh(range(len(es_df)), es_df.d.values, color=[COLORS[c] for c in es_df['class']], alpha=0.8)
ax.set_yticks(range(len(es_df)))
ax.set_yticklabels([f"{r['class']}  (d={r['d']:.2f})" for _,r in es_df.iterrows()])
ax.axvline(0.5, color='gray', ls=':', alpha=0.5); ax.axvline(0.8, color='gray', ls='--', alpha=0.5)
ax.set_xlabel("Cohen's d (vs NORM)"); ax.set_title('b', fontweight='bold', loc='left', fontsize=14)
ax.grid(axis='x', alpha=0.3)

# C: Spatial fingerprint
ax = axes[2]
clbbb = df[df.scp_dict.apply(lambda x: 'CLBBB' in x and x.get('CLBBB',0)>=80)]
crbbb = df[df.scp_dict.apply(lambda x: 'CRBBB' in x and x.get('CRBBB',0)>=80)]
for label, sub, color, marker in [('NORM',norm,'#4CAF50','o'),('CLBBB',clbbb,'#E53935','s'),('CRBBB',crbbb,'#1565C0','^')]:
    meds = [sub[f'beta_ir_{l}'].median() for l in LEAD_NAMES]
    q25 = [sub[f'beta_ir_{l}'].quantile(0.25) for l in LEAD_NAMES]
    q75 = [sub[f'beta_ir_{l}'].quantile(0.75) for l in LEAD_NAMES]
    ax.plot(range(12), meds, f'{marker}-', color=color, lw=2, ms=6, label=f'{label} (n={len(sub):,})')
    ax.fill_between(range(12), q25, q75, alpha=0.12, color=color)
ax.axvspan(6, 9, alpha=0.06, color='red')
ax.set_xticks(range(12)); ax.set_xticklabels(LEAD_NAMES)
ax.set_ylabel(r'$\beta$ (median)'); ax.set_title('c', fontweight='bold', loc='left', fontsize=14)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_beta_landscape.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig1_beta_landscape.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig1 saved")

# ================================================================
# FIGURE 2: Spectral Anatomy
# ================================================================
print("Generating Figure 2...")
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3, width_ratios=[1.3,1,1])

# Classification
bbb = cd_data[cd_data.cd_subtype.isin(['CLBBB','CRBBB'])].copy()
bbb['label'] = (bbb.cd_subtype=='CLBBB').astype(int)
bbb_tr = bbb[bbb.strat_fold.isin(range(1,9))]
bbb_te = bbb[bbb.strat_fold.isin([9,10])]
tr = bbb_tr[avail+['label']].dropna(); te = bbb_te[avail+['label']].dropna()
X_tr, y_tr = tr[avail].values, tr['label'].values
X_te, y_te = te[avail].values, te['label'].values
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
gbm.fit(X_tr_s, y_tr)
y_prob = gbm.predict_proba(X_te_s)[:,1]
bbb_auc = roc_auc_score(y_te, y_prob)
bbb_fpr, bbb_tpr, _ = roc_curve(y_te, y_prob)
print(f"  CLBBB vs CRBBB AUC = {bbb_auc:.3f}")

# 6-class
cd_tr6 = cd_data[cd_data.strat_fold.isin(range(1,9))]
cd_te6 = cd_data[cd_data.strat_fold.isin([9,10])]
tr6 = cd_tr6[avail+['cd_subtype']].dropna(); te6 = cd_te6[avail+['cd_subtype']].dropna()
X_tr6, y_tr6 = tr6[avail].values, tr6['cd_subtype'].values
X_te6, y_te6 = te6[avail].values, te6['cd_subtype'].values
sc6 = StandardScaler(); X_tr6s = sc6.fit_transform(X_tr6); X_te6s = sc6.transform(X_te6)
gbm6 = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
gbm6.fit(X_tr6s, y_tr6)
y_prob6 = gbm6.predict_proba(X_te6s)
classes6 = gbm6.classes_
y_te6_bin = label_binarize(y_te6, classes=classes6)
per_auc = {}
for i, c in enumerate(classes6):
    try: per_auc[c] = roc_auc_score(y_te6_bin[:,i], y_prob6[:,i])
    except: per_auc[c] = np.nan
macro = np.nanmean(list(per_auc.values()))
print(f"  6-class macro AUC = {macro:.3f}")

# A: Spatial fingerprint
ax = fig.add_subplot(gs[0,:2])
norm_meds = [norm[f'beta_ir_{l}'].median() for l in LEAD_NAMES]
ax.plot(range(12), norm_meds, 'o--', color=COLORS['NORM'], lw=1.5, ms=4, alpha=0.5, label=f'NORM (n={len(norm):,})')
for sub in ['CLBBB','CRBBB','IRBBB','LAFB','1AVB','IVCD']:
    s = cd_data[cd_data.cd_subtype==sub]
    if len(s)<10: continue
    meds = [s[f'beta_ir_{l}'].median() for l in LEAD_NAMES]
    q25 = [s[f'beta_ir_{l}'].quantile(0.25) for l in LEAD_NAMES]
    q75 = [s[f'beta_ir_{l}'].quantile(0.75) for l in LEAD_NAMES]
    ax.plot(range(12), meds, 'o-', color=CD_COLORS[sub], lw=2, ms=5, label=f'{sub} (n={len(s):,})')
    ax.fill_between(range(12), q25, q75, alpha=0.08, color=CD_COLORS[sub])
ax.axvspan(6,9, alpha=0.06, color='red')
ax.set_xticks(range(12)); ax.set_xticklabels(LEAD_NAMES)
ax.set_ylabel(r'$\beta$ (median)'); ax.set_title('a', fontweight='bold', loc='left', fontsize=14)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# B: Per-subtype AUC
ax = fig.add_subplot(gs[0,2])
sorted_s = sorted(per_auc.keys(), key=lambda k: per_auc.get(k,0), reverse=True)
bars = ax.bar(range(len(sorted_s)), [per_auc[c] for c in sorted_s],
              color=[CD_COLORS.get(c,'gray') for c in sorted_s], alpha=0.85, edgecolor='black', lw=0.5)
ax.set_xticks(range(len(sorted_s))); ax.set_xticklabels(sorted_s, fontsize=9)
ax.set_ylabel('AUC (one-vs-rest)'); ax.set_title('b', fontweight='bold', loc='left', fontsize=14)
ax.axhline(0.5, color='gray', ls=':', alpha=0.5); ax.set_ylim(0.4,1.05)
for bar, v in zip(bars, [per_auc[c] for c in sorted_s]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

# C: ROC
ax = fig.add_subplot(gs[1,0])
ax.plot(bbb_fpr, bbb_tpr, lw=2.5, color='#C62828', label=f'GBM (AUC = {bbb_auc:.3f})')
ax.plot([0,1],[0,1],'k:',alpha=0.3)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('c', fontweight='bold', loc='left', fontsize=14)
ax.legend(fontsize=10, loc='lower right'); ax.grid(alpha=0.3)

# D: Feature importance
ax = fig.add_subplot(gs[1,1])
imp = pd.DataFrame({'feature':avail,'importance':gbm.feature_importances_}).sort_values('importance',ascending=True).tail(12)
colors_imp = ['#E53935' if 'beta_ir_V' in f else '#FF9800' if 'beta_ir_' in f else '#2196F3' if 'r2_ir_' in f else '#9E9E9E' for f in imp.feature]
ax.barh(range(len(imp)), imp.importance.values, color=colors_imp, alpha=0.85)
ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.feature.values, fontsize=9)
ax.set_xlabel('Feature Importance'); ax.set_title('d', fontweight='bold', loc='left', fontsize=14)

# E: Radar
ax = fig.add_subplot(gs[1,2], projection='polar')
angles = np.linspace(0, 2*np.pi, 12, endpoint=False).tolist() + [0]
for label, sub, color in [('NORM',norm,'#4CAF50'),('CLBBB',cd_data[cd_data.cd_subtype=='CLBBB'],'#E53935'),('CRBBB',cd_data[cd_data.cd_subtype=='CRBBB'],'#1565C0')]:
    meds = [sub[f'beta_ir_{l}'].median() for l in LEAD_NAMES] + [sub[f'beta_ir_I'].median()]
    ax.plot(angles, meds, 'o-', color=color, lw=2, ms=4, label=f'{label}')
    ax.fill(angles, meds, color=color, alpha=0.08)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(LEAD_NAMES, fontsize=7)
ax.set_title('e', fontweight='bold', loc='left', fontsize=14, pad=20)
ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3,1.1))

plt.savefig(f'{OUT}/fig2_spectral_anatomy.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig2_spectral_anatomy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig2 saved")

# ================================================================
# FIGURE 3: Aging Bifurcation
# ================================================================
print("Generating Figure 3...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# A: β vs age
ax = axes[0]
bins = np.arange(20,91,5)
norm['age_bin'] = pd.cut(norm.age, bins=bins)
binned = norm.groupby('age_bin').agg(age_mid=('age','mean'),beta_avg=('beta_mean','mean'),
    beta_se=('beta_mean',lambda x: x.std()/np.sqrt(len(x)))).dropna()
rng = np.random.RandomState(42)
idx = rng.choice(len(norm), min(3000,len(norm)), replace=False)
ax.scatter(norm.age.values[idx], norm.beta_mean.values[idx], alpha=0.08, s=8, c=COLORS['NORM'], rasterized=True)
ax.errorbar(binned.age_mid, binned.beta_avg, yerr=binned.beta_se*1.96, fmt='ko-', ms=5, lw=1.5, capsize=3, zorder=5)
rho_age, p_age = stats.spearmanr(norm.age, norm.beta_mean)
ax.set_xlabel('Age (years)'); ax.set_ylabel(r'$\beta$ (mean across leads)')
ax.set_title(f'a   $\\rho$ = {rho_age:.3f}', fontweight='bold', loc='left', fontsize=14)
ax.grid(alpha=0.3)

# B: Breakpoint
ax = axes[1]
age_v = norm.age.values.astype(float); bstd_v = norm.beta_std.values.astype(float)
valid = np.isfinite(age_v) & np.isfinite(bstd_v); age_v, bstd_v = age_v[valid], bstd_v[valid]
def seg_rss(bp):
    l=age_v<=bp; r=age_v>bp
    if l.sum()<50 or r.sum()<50: return 1e10
    rss=0
    for m in [l,r]:
        s,i,_,_,_=stats.linregress(age_v[m],bstd_v[m])
        rss+=np.sum((bstd_v[m]-(i+s*age_v[m]))**2)
    return rss
bpc = np.arange(30,65,0.5); rss_v = [seg_rss(b) for b in bpc]; bp = bpc[np.argmin(rss_v)]
lm=age_v<=bp; rm=age_v>bp
sl,il,_,_,_=stats.linregress(age_v[lm],bstd_v[lm])
sr,ir,_,_,_=stats.linregress(age_v[rm],bstd_v[rm])
binned_bp = norm.groupby('age_bin').agg(age_mid=('age','mean'),bstd_avg=('beta_std','mean'),
    bstd_se=('beta_std',lambda x:x.std()/np.sqrt(len(x)))).dropna()
ax.errorbar(binned_bp.age_mid, binned_bp.bstd_avg, yerr=binned_bp.bstd_se*1.96, fmt='ko-', ms=5, lw=1.5, capsize=3)
ax.plot(np.linspace(18,bp,50), il+sl*np.linspace(18,bp,50), 'b-', lw=2)
ax.plot(np.linspace(bp,90,50), ir+sr*np.linspace(bp,90,50), 'r-', lw=2)
ax.axvline(bp, color='gray', ls=':', alpha=0.7, label=f'Breakpoint: {bp:.0f}y')
ax.set_xlabel('Age (years)'); ax.set_ylabel(r'$\sigma_\beta$ (spatial heterogeneity)')
ax.set_title('b', fontweight='bold', loc='left', fontsize=14); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# C: Bifurcation
ax = axes[2]
decades = [(20,35,'Young'),(36,50,'Middle'),(51,65,'Senior'),(66,90,'Elderly')]
xa, ya = [], []
for lo,hi,lab in decades:
    s = norm[(norm.age>=lo)&(norm.age<=hi)]
    xa.append(s.age.mean()); ya.append(s.beta_mean.median())
ax.plot(xa, ya, 'o-', color=COLORS['NORM'], lw=2.5, ms=10, label='Aging (NORM)', zorder=5)
for cls in ['STTC','MI','HYP','CD']:
    s = df[df.clean_superclass==cls]
    ax.scatter(s.age.mean(), s.beta_mean.median(), s=200, c=COLORS[cls], marker='D', edgecolors='black', lw=0.5, zorder=6, label=cls)
ax.annotate('', xy=(80,ya[-1]-0.01), xytext=(30,ya[0]+0.01), arrowprops=dict(arrowstyle='->',color=COLORS['NORM'],lw=2))
ax.text(55, min(ya)-0.06, r'$\beta \downarrow$ with age', color=COLORS['NORM'], fontsize=10, ha='center', fontweight='bold')
ax.annotate('', xy=(55,2.05), xytext=(55,1.85), arrowprops=dict(arrowstyle='->',color=COLORS['CD'],lw=2))
ax.text(58, 1.96, r'$\beta \uparrow$ in disease', color=COLORS['CD'], fontsize=10, ha='left', fontweight='bold')
ax.set_xlabel('Mean age (years)'); ax.set_ylabel(r'$\beta$ (median)')
ax.set_title('c', fontweight='bold', loc='left', fontsize=14); ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig3_aging_bifurcation.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig3_aging_bifurcation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig3 saved")

# ================================================================
# FIGURE 4: Subclinical Detection
# ================================================================
print("Generating Figure 4...")
subclinical_cd = {'IRBBB','IVCD','1AVB','LAFB','LPFB','ILBBB'}
ga_ids, gb_ids = [], []
for eid, row in df.iterrows():
    codes = row['scp_dict']
    if 'NORM' not in codes or codes.get('NORM',0)<80: continue
    cd_p = {c for c in codes if c in subclinical_cd and codes[c]>0}
    if len(cd_p)==0 and len(codes)==1: ga_ids.append(eid)
    elif len(cd_p)>0: gb_ids.append(eid)
ga = df.loc[df.index.isin(ga_ids)].copy(); ga['sub_group']=0
gb = df.loc[df.index.isin(gb_ids)].copy(); gb['sub_group']=1
combined = pd.concat([ga, gb])

# Effects
key_feats = ['beta_mean','beta_std','beta_anterior','beta_lateral','beta_inferior','beta_regional_div','r2_mean']
eff_res = []
for feat in key_feats:
    va, vb = ga[feat].dropna(), gb[feat].dropna()
    _, p = stats.mannwhitneyu(va, vb)
    d = (vb.mean()-va.mean())/np.sqrt(((len(va)-1)*va.std()**2+(len(vb)-1)*vb.std()**2)/(len(va)+len(vb)-2))
    eff_res.append({'feature':feat,'d':d,'p':p})

# Classification
sub_tr = combined[combined.strat_fold.isin(range(1,9))]
sub_te = combined[combined.strat_fold.isin([9,10])]
sub_avail = [f for f in avail if f in combined.columns]
trs = sub_tr[sub_avail+['sub_group']].dropna(); tes = sub_te[sub_avail+['sub_group']].dropna()
X_trs, y_trs = trs[sub_avail].values, trs['sub_group'].values
X_tes, y_tes = tes[sub_avail].values, tes['sub_group'].values
scs = StandardScaler(); X_trs_s = scs.fit_transform(X_trs); X_tes_s = scs.transform(X_tes)
gbm_s = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)
gbm_s.fit(X_trs_s, y_trs)
yps = gbm_s.predict_proba(X_tes_s)[:,1]
sub_auc = roc_auc_score(y_tes, yps)
sub_fpr, sub_tpr, _ = roc_curve(y_tes, yps)
print(f"  Subclinical AUC = {sub_auc:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
# A: Distributions
ax = axes[0]
ax.hist(ga.beta_mean.dropna(), bins=60, alpha=0.6, density=True, color=COLORS['NORM'], label=f'Pure NORM (n={len(ga):,})', edgecolor='white', lw=0.3)
ax.hist(gb.beta_mean.dropna(), bins=40, alpha=0.6, density=True, color='#E53935', label=f'NORM + subclinical (n={len(gb):,})', edgecolor='white', lw=0.3)
ax.axvline(ga.beta_mean.median(), color=COLORS['NORM'], ls='--', lw=2)
ax.axvline(gb.beta_mean.median(), color='#E53935', ls='--', lw=2)
ax.set_xlabel(r'$\beta$'); ax.set_ylabel('Density')
ax.set_title('a', fontweight='bold', loc='left', fontsize=14); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# B: Effect sizes
ax = axes[1]
eff_df = pd.DataFrame(eff_res).sort_values('d',ascending=True)
colors_e = ['#E53935' if abs(d)>0.3 else '#FF9800' if abs(d)>0.15 else '#9E9E9E' for d in eff_df.d]
ax.barh(range(len(eff_df)), eff_df.d.values, color=colors_e, alpha=0.85)
ax.set_yticks(range(len(eff_df))); ax.set_yticklabels(eff_df.feature.values, fontsize=9)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel("Cohen's d"); ax.set_title('b', fontweight='bold', loc='left', fontsize=14)
for i, (_,r) in enumerate(eff_df.iterrows()):
    star = '***' if r['p']<0.001 else '**' if r['p']<0.01 else '*' if r['p']<0.05 else ''
    if star: ax.text(r['d']+(0.01 if r['d']>=0 else -0.06), i, star, va='center', fontsize=10, fontweight='bold', color='#C62828')

# C: ROC
ax = axes[2]
ax.plot(sub_fpr, sub_tpr, lw=2.5, color='#C62828', label=f'GBM (AUC = {sub_auc:.3f})')
ax.plot([0,1],[0,1],'k:',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('c', fontweight='bold', loc='left', fontsize=14); ax.legend(fontsize=10, loc='lower right'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_subclinical.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig4_subclinical.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig4 saved")

# ================================================================
# FIGURE 5: External Validation (copy existing + enhance)
# ================================================================
print("Generating Figure 5...")
# Use existing external validation figures and create a combined one
from matplotlib.image import imread

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# A: Three-continent AUC bars
ax = axes[0]
datasets = ['PTB-XL\n(Germany)','Chapman\n(China)','CODE-15%\n(Brazil)']
aucs = [bbb_auc, 0.982, 0.979]  # Use actual PTB-XL AUC computed here
colors_bar = ['#3498db','#e74c3c','#2ecc71']
bars = ax.bar(datasets, aucs, color=colors_bar, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.003, f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0.90, 1.01); ax.set_ylabel('AUC (CLBBB vs CRBBB)')
ax.set_title('a', fontweight='bold', loc='left', fontsize=14)
ax.axhline(0.95, color='gray', ls=':', alpha=0.3)

# B: β_NORM comparison
ax = axes[1]
b_norms = [1.76, 1.52, 2.75]
bars2 = ax.bar(datasets, b_norms, color=colors_bar, alpha=0.8, edgecolor='black')
for bar, val in zip(bars2, b_norms):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.03, f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$\beta_{\rm NORM}$ (mean)'); ax.set_ylim(0, 3.2)
ax.set_title('b', fontweight='bold', loc='left', fontsize=14)

# C: Summary table
ax = axes[2]
ax.axis('off')
table_data = [
    ['', 'PTB-XL', 'Chapman', 'CODE-15%'],
    ['Country', 'Germany', 'China', 'Brazil'],
    ['N (total)', '21,799', '45,152', '345,779'],
    [r'$\beta_{\rm NORM}$', '1.76', '1.52', '2.75'],
    ['CLBBB / CRBBB', '536 / 537', '213 / 1,096', '414 / 557'],
    ['AUC', f'{bbb_auc:.3f}', '0.982', '0.979'],
    ['5-fold CV', '—', '0.980±0.009', '0.977±0.006'],
]
table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
for j in range(4):
    table[0,j].set_facecolor('#2c3e50'); table[0,j].set_text_props(color='white', fontweight='bold')
table[5,1].set_facecolor('#d5f4e6'); table[5,2].set_facecolor('#d5f4e6'); table[5,3].set_facecolor('#d5f4e6')
ax.set_title('c', fontweight='bold', loc='left', fontsize=14)

plt.tight_layout()
plt.savefig(f'{OUT}/fig5_external_validation.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig5_external_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → fig5 saved")

# === Print verified numbers ===
print(f"\n{'='*60}")
print("VERIFIED NUMBERS FOR PAPER")
print(f"{'='*60}")
print(f"CLBBB vs CRBBB AUC (PTB-XL): {bbb_auc:.3f}")
print(f"6-class macro AUC: {macro:.3f}")
for c in cd_subtypes_of_interest:
    if c in per_auc: print(f"  {c}: {per_auc[c]:.3f}")
print(f"Aging ρ: {rho_age:.4f}")
print(f"Breakpoint: {bp:.0f}")
print(f"Subclinical AUC: {sub_auc:.3f}")
print(f"Pure NORM: {len(ga)}, Subclinical: {len(gb)}")
b_ant_d = [r for r in eff_res if r['feature']=='beta_anterior'][0]['d']
print(f"beta_anterior Cohen's d: {b_ant_d:.3f}")
print("Done!")
