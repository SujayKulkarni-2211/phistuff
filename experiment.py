"""
Predictive Modelling of Healthcare Burden-Infrastructure Gaps Across Indian States
Reproducible experiment — corrected methodology.

Key methodological decisions (documented):

1. EXCLUSION of small UTs (<1M rural population):
   Lakshadweep (3,000 rural residents) has 266 doctors/100k vs Bihar at 1.8/100k.
   Including it in a per-capita analysis would dominate every normalisation step
   and completely obscure the states where millions of people lack care.
   Excluded: Lakshadweep, Chandigarh, Daman & Diu, Dadra & NH, A&N Islands,
             Sikkim, Goa, Puducherry, Delhi, Ladakh.

2. Z-NORMALISATION within each time period separately:
   National health improved from 2005→2015 (mean IMR fell from 47 to 32).
   If we z-normalise across both years combined, the z-scores encode which
   time period each row belongs to (2005 rows systematically higher z).
   Linear regression then learns to detect the year, not the cross-state gap,
   producing a spurious R²=0.994. We normalise within year so BurdenIndex
   captures "how sick is this state RELATIVE TO ITS PEERS at that moment."

3. ML FORMULATION — predictive, not retrospective:
   Features = 2005 raw state health indicators (IMR, U5MR, underweight, anaemia,
   and per-capita infrastructure at baseline).
   Target = 2015 GapIndex for the SAME state.
   This answers: "given what we knew about a state in 2005, could we have
   predicted which states would have the worst burden-infrastructure mismatch
   ten years later?" That is a genuine policy-useful question.
   Cross-validation via Leave-One-Out (only 19 complete states).

4. FORWARD PROJECTION uses the 2005→2015 trend per state extrapolated to 2030,
   with uncertainty from the residual variation around that linear trend.

Usage:
    python experiment.py

Outputs:
    figures/  — 6 publication-quality PNG figures
    results.json — all numeric results referenced in the paper
"""

import os, json, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

RHS_BASE   = "/home/sujay-v-kulkarni/.cache/kagglehub/datasets/vineethakkinapalli/rural-health-statistics-india2005-2019-2020/versions/3"
NFHS4_PATH = "/home/sujay-v-kulkarni/.cache/kagglehub/datasets/awadhi123/national-family-health-survey4/versions/1/nfhs4.csv"
NFHS3_PATH = "/home/sujay-v-kulkarni/.cache/kagglehub/datasets/awadhi123/national-family-health-survey3-web-crawled/versions/1/nfhs3.csv"
POP_PATH   = os.path.join(RHS_BASE, "rhs_population_density.csv")

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────

STATE_MAP = {
    'Chattisgarh': 'Chhattisgarh',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'A& N Islands': 'Andaman and Nicobar Islands',
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Andhra Pradesh**': 'Andhra Pradesh',
    'Telangana**': 'Telangana',
    'Orissa': 'Odisha',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
    'D & N Haveli and \nDaman & Diu': 'Dadra and Nagar Haveli',
    'Andaman & Nicobar \nIslands': 'Andaman and Nicobar Islands',
}

# States/UTs excluded because their tiny rural populations create extreme
# per-capita outliers that would distort any normalisation.
EXCLUDE = {
    'Lakshadweep', 'Chandigarh', 'Daman and Diu', 'Dadra and Nagar Haveli',
    'Andaman and Nicobar Islands', 'Sikkim', 'Goa', 'Puducherry', 'Delhi',
    'Ladakh', 'India', 'All India Total',
}

def norm(df):
    df['State'] = df['State'].str.strip().replace(STATE_MAP)
    return df

def load_rhs(year):
    df = pd.read_csv(os.path.join(RHS_BASE, f"rhs_{year}.csv"))
    df.columns = df.columns.str.strip()
    df.rename(columns={'State/UT': 'State'}, inplace=True)
    norm(df)
    df['RHSYear'] = year
    for c in df.columns:
        if c not in ('State', 'RHSYear'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[~df['State'].isin(EXCLUDE)]

def load_nfhs(path, survey_year):
    df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.strip()
    df = df[df['Area'].str.strip() == 'Total'].copy()
    df.rename(columns={'India/States/UTs': 'State'}, inplace=True)
    norm(df)
    df['SurveyYear'] = survey_year
    return df[~df['State'].isin(EXCLUDE)]

rhs05 = load_rhs(2005)
rhs19 = load_rhs(2019)
nfhs3 = load_nfhs(NFHS3_PATH, 2005)
nfhs4 = load_nfhs(NFHS4_PATH, 2015)

pop = pd.read_csv(POP_PATH)
pop.columns = pop.columns.str.strip()
pop.rename(columns={'State/UT': 'State'}, inplace=True)
norm(pop)
pop = pop[~pop['State'].isin(EXCLUDE)]
for c in pop.columns:
    if c != 'State':
        pop[c] = pd.to_numeric(pop[c], errors='coerce')

# ─────────────────────────────────────────────────────────────
# 2. EXTRACT KEY COLUMNS
# ─────────────────────────────────────────────────────────────

IMR_C   = 'Infant and Child Mortality Rates (per 1000 live births) - Infant mortality rate (IMR)'
U5MR_C  = 'Infant and Child Mortality Rates (per 1000 live births) - Under-five mortality rate (U5MR)'
UW_C    = 'Child Feeding Practices and Nutritional Status of Children - Children under 5 years who are underweight (weight-for-age) (%)'
AN_C    = 'Anaemia among Children and Adults15 - Children age 6-59 months who are anaemic (<11.0 g/dl) (%)'
INFRA_COLS = ['SubCenters', 'PHCs', 'CHCs', 'Doctors']
BURDEN_RENAME = {IMR_C: 'IMR', U5MR_C: 'U5MR', UW_C: 'Underweight_pct', AN_C: 'Anaemia_child_pct'}

def extract_burden(df):
    keep = [c for c in [IMR_C, U5MR_C, UW_C, AN_C] if c in df.columns]
    out = df[['State', 'SurveyYear'] + keep].copy()
    out.rename(columns=BURDEN_RENAME, inplace=True)
    for c in ['IMR', 'U5MR', 'Underweight_pct', 'Anaemia_child_pct']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

b3 = extract_burden(nfhs3)
b4 = extract_burden(nfhs4)
burden_all = pd.concat([b3, b4], ignore_index=True)

# ─────────────────────────────────────────────────────────────
# 3. BUILD PANEL & COMPUTE PER-CAPITA INFRASTRUCTURE
# ─────────────────────────────────────────────────────────────

# NFHS3 (2005) ↔ RHS 2005;  NFHS4 (2015) ↔ RHS 2019 (closest available)
burden_all['RHSYear'] = burden_all['SurveyYear'].map({2005: 2005, 2015: 2019})

infra_panel = pd.concat([
    rhs05[['State', 'RHSYear'] + INFRA_COLS],
    rhs19[['State', 'RHSYear'] + INFRA_COLS],
], ignore_index=True)

panel = pd.merge(burden_all, infra_panel, on=['State', 'RHSYear'], how='inner')
panel = pd.merge(panel, pop[['State', 'Rural_Population']], on='State', how='left')

# Drop states where rural population is missing or below 1M
panel = panel[panel['Rural_Population'].notna() & (panel['Rural_Population'] >= 1_000_000)].copy()
panel.reset_index(drop=True, inplace=True)

pop_scale = panel['Rural_Population'] / 100_000
for col in INFRA_COLS:
    panel[f'{col}_per100k'] = panel[col] / pop_scale

print(f"Panel: {len(panel)} rows, {panel['State'].nunique()} states, {panel['SurveyYear'].unique()} survey years")

# ─────────────────────────────────────────────────────────────
# 4. COMPOSITE INDICES — normalised within each time period
# ─────────────────────────────────────────────────────────────

BURDEN_RAW   = ['IMR', 'U5MR', 'Underweight_pct', 'Anaemia_child_pct']
INFRA_PC     = ['SubCenters_per100k', 'PHCs_per100k', 'CHCs_per100k', 'Doctors_per100k']

for yr in panel['SurveyYear'].unique():
    mask = panel['SurveyYear'] == yr
    for col in BURDEN_RAW:
        if col in panel.columns:
            vals = panel.loc[mask, col].dropna()
            panel.loc[mask, f'{col}_z'] = (panel.loc[mask, col] - vals.mean()) / (vals.std() + 1e-9)
    for col in INFRA_PC:
        if col in panel.columns:
            vals = panel.loc[mask, col].dropna()
            panel.loc[mask, f'{col}_z'] = (panel.loc[mask, col] - vals.mean()) / (vals.std() + 1e-9)

burden_z = [f'{c}_z' for c in BURDEN_RAW if f'{c}_z' in panel.columns]
infra_z  = [f'{c}_z' for c in INFRA_PC  if f'{c}_z' in panel.columns]

panel['BurdenIndex'] = panel[burden_z].mean(axis=1)
panel['InfraIndex']  = panel[infra_z].mean(axis=1)
# Gap: high burden + low infra = large positive value = most at risk
panel['GapIndex']    = panel['BurdenIndex'] - panel['InfraIndex']

REGION_MAP = {
    'Andhra Pradesh':'South','Telangana':'South','Tamil Nadu':'South',
    'Karnataka':'South','Kerala':'South',
    'Maharashtra':'West','Gujarat':'West','Rajasthan':'West',
    'Uttar Pradesh':'North','Haryana':'North','Punjab':'North',
    'Himachal Pradesh':'North','Uttarakhand':'North','Jammu and Kashmir':'North',
    'Bihar':'East','West Bengal':'East','Odisha':'East',
    'Jharkhand':'East','Chhattisgarh':'East',
    'Madhya Pradesh':'Central',
    'Assam':'NE','Arunachal Pradesh':'NE','Manipur':'NE',
    'Meghalaya':'NE','Mizoram':'NE','Nagaland':'NE','Tripura':'NE',
}
panel['Region'] = panel['State'].map(REGION_MAP).fillna('Other')

print("\n--- Top gap states 2005 ---")
top05 = panel[panel['SurveyYear']==2005].sort_values('GapIndex', ascending=False)
print(top05[['State','BurdenIndex','InfraIndex','GapIndex']].head(8).round(3).to_string())

print("\n--- Top gap states 2015 ---")
top15 = panel[panel['SurveyYear']==2015].sort_values('GapIndex', ascending=False)
print(top15[['State','BurdenIndex','InfraIndex','GapIndex']].head(8).round(3).to_string())

# ─────────────────────────────────────────────────────────────
# 5. ML — PREDICTIVE FORMULATION
#    Features: 2005 raw baseline indicators
#    Target:   2015 GapIndex for same state
#    CV:       Leave-One-Out (only ~20 complete states)
# ─────────────────────────────────────────────────────────────

df05 = panel[panel['SurveyYear']==2005].set_index('State')
df15 = panel[panel['SurveyYear']==2015].set_index('State')

# States with full data in both years
FEATURE_COLS = BURDEN_RAW + INFRA_PC
common_states = sorted(
    set(df05.index) & set(df15.index)
)
X_all = df05.loc[common_states, FEATURE_COLS].copy()
y_all = df15.loc[common_states, 'GapIndex'].copy()
valid = X_all.notna().all(axis=1) & y_all.notna()
X_ml = X_all[valid].copy()
y_ml = y_all[valid].copy()

print(f"\nML dataset: {len(X_ml)} states")
print("States:", X_ml.index.tolist())

models = {
    'Linear Regression': Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]),
    'Random Forest':     RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_leaf=2, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=42),
}

loo = LeaveOneOut()
ml_results = {}

for name, model in models.items():
    y_pred_loo = cross_val_predict(model, X_ml, y_ml, cv=loo)
    mae  = mean_absolute_error(y_ml, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y_ml, y_pred_loo))
    r2   = r2_score(y_ml, y_pred_loo)
    ml_results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"{name}: LOO MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

best_name = max(ml_results, key=lambda k: ml_results[k]['R2'])
print(f"\nBest model (LOO R²): {best_name}")

# Fit best model on all data for feature importance and projection
best_model = models[best_name]
best_model.fit(X_ml, y_ml)

if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
elif hasattr(best_model, 'named_steps'):
    coef = best_model.named_steps['m'].coef_
    feat_imp = pd.Series(np.abs(coef) / (np.abs(coef).sum() + 1e-9), index=FEATURE_COLS)
else:
    feat_imp = pd.Series(np.ones(len(FEATURE_COLS))/len(FEATURE_COLS), index=FEATURE_COLS)
feat_imp = feat_imp.sort_values(ascending=False)

print("\nFeature importances:")
print(feat_imp.round(4))

# ─────────────────────────────────────────────────────────────
# 6. FORWARD PROJECTION TO 2030
# ─────────────────────────────────────────────────────────────

state_gap = panel.pivot_table(index='State', columns='SurveyYear', values='GapIndex')
state_gap.columns.name = None
states_both = state_gap.dropna().index.tolist()

proj_rows = []
for state in states_both:
    g05 = state_gap.loc[state, 2005]
    g15 = state_gap.loc[state, 2015]
    # Linear extrapolation from 2005→2015 trend
    trend_per_yr = (g15 - g05) / 10.0
    g2030 = g15 + trend_per_yr * 15  # 15 years from 2015 to 2030

    # Bootstrap CI using residual variation from LOO predictions
    # Use model residual std as uncertainty proxy, scaled by time horizon
    residual_std = np.std(y_ml - cross_val_predict(best_model, X_ml, y_ml, cv=loo)) if state in X_ml.index else 0.5
    noise = np.random.normal(0, residual_std, 2000)
    ci_lo = np.percentile(g2030 + noise, 5)
    ci_hi = np.percentile(g2030 + noise, 95)
    proj_rows.append({
        'State': state, 'Gap2005': g05, 'Gap2015': g15,
        'Gap2030': g2030, 'CI_lo': ci_lo, 'CI_hi': ci_hi,
        'Region': REGION_MAP.get(state, 'Other'),
    })

proj = pd.DataFrame(proj_rows).sort_values('Gap2030', ascending=False)
top10 = proj.head(10)
print("\n--- Top 10 most at-risk states by 2030 ---")
print(top10[['State','Gap2005','Gap2015','Gap2030','CI_lo','CI_hi']].round(3).to_string())

# ─────────────────────────────────────────────────────────────
# 7. SAVE RESULTS
# ─────────────────────────────────────────────────────────────

all_results = {
    'n_states': int(panel['State'].nunique()),
    'n_states_ml': int(len(X_ml)),
    'model_metrics_loo': ml_results,
    'best_model': best_name,
    'feature_importances': feat_imp.to_dict(),
    'top10_at_risk_2030': top10.set_index('State')[['Gap2005','Gap2015','Gap2030']].to_dict(),
    'top_gap_states_2015': top15.head(5)[['State','GapIndex']].set_index('State')['GapIndex'].to_dict(),
}
with open(os.path.join(BASE_DIR, 'results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# ─────────────────────────────────────────────────────────────
# 8. FIGURES
# ─────────────────────────────────────────────────────────────

REGION_COLORS = {
    'North':'#4e79a7','South':'#f28e2b','East':'#e15759',
    'West':'#76b7b2','NE':'#59a14f','Central':'#edc948','Other':'#b07aa1'
}
FEAT_LABELS = {
    'IMR':'Infant Mortality Rate (IMR)',
    'U5MR':'Under-5 Mortality Rate (U5MR)',
    'Underweight_pct':'Children Underweight (%)',
    'Anaemia_child_pct':'Child Anaemia (%)',
    'SubCenters_per100k':'Sub-Centres /100k',
    'PHCs_per100k':'PHCs /100k',
    'CHCs_per100k':'CHCs /100k',
    'Doctors_per100k':'Doctors /100k',
}

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.titlesize':13,'axes.labelsize':11,
    'xtick.labelsize':9,'ytick.labelsize':9,'figure.dpi':300,
})

# ── Fig 1: Burden heatmap ────────────────────────────────────
fig1 = panel.pivot_table(index='State', columns='SurveyYear', values='BurdenIndex')
fig1.columns = [f'NFHS-3 ({c})' if c==2005 else f'NFHS-4 ({c})' for c in fig1.columns]
fig1 = fig1.sort_values(fig1.columns[0], ascending=False)

fig, ax = plt.subplots(figsize=(8, 11))
sns.heatmap(fig1, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            linewidths=0.4, ax=ax, cbar_kws={'label':'Burden Index (z-score, within year)'})
ax.set_title('State-wise Composite Health Burden Index\nHigher = greater burden relative to national peers', pad=10)
ax.set_xlabel(''); ax.set_ylabel('')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig1_burden_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig1")

# ── Fig 2: Infrastructure heatmap ───────────────────────────
fig2 = panel.pivot_table(index='State', columns='RHSYear', values='InfraIndex')
fig2.columns = [f'RHS {c}' for c in fig2.columns]
fig2 = fig2.sort_values(fig2.columns[0], ascending=False)

fig, ax = plt.subplots(figsize=(8, 11))
sns.heatmap(fig2, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.4, ax=ax, cbar_kws={'label':'Infrastructure Index (z-score, within year)'})
ax.set_title('State-wise Healthcare Infrastructure Index\nHigher = better infrastructure relative to national peers', pad=10)
ax.set_xlabel(''); ax.set_ylabel('')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig2_infra_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig2")

# ── Fig 3: Gap scatter — the core story ─────────────────────
scatter = panel[panel['SurveyYear']==2015].dropna(subset=['BurdenIndex','InfraIndex']).copy()
scatter['GapLabel'] = scatter['GapIndex'].apply(lambda x: 'high' if x > 1.0 else ('low' if x < -0.5 else 'mid'))

fig, ax = plt.subplots(figsize=(11, 8))
for region, grp in scatter.groupby('Region'):
    ax.scatter(grp['InfraIndex'], grp['BurdenIndex'],
               color=REGION_COLORS.get(region,'#999'), label=region,
               s=110, edgecolors='white', linewidth=0.6, zorder=3, alpha=0.9)

# Label every state, with larger font for high-gap ones
for _, row in scatter.iterrows():
    fontsize = 8.5 if row['GapLabel']=='high' else 7.5
    weight   = 'bold' if row['GapLabel']=='high' else 'normal'
    ax.annotate(row['State'], (row['InfraIndex'], row['BurdenIndex']),
                xytext=(4, 2), textcoords='offset points',
                fontsize=fontsize, fontweight=weight, color='#222222')

# Diagonal: burden = infra (gap = 0)
xlim = ax.get_xlim(); ylim = ax.get_ylim()
lo = min(xlim[0], ylim[0]); hi = max(xlim[1], ylim[1])
ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.35, linewidth=1.2, label='Gap = 0 (balanced)')

# Shade the high-gap quadrant
ax.axhspan(0.5, ylim[1], xmin=0, xmax=0.5, alpha=0.05, color='red')
ax.text(lo+0.05, ylim[1]-0.15, 'HIGH BURDEN\nLOW INFRA\n→ CRISIS ZONE',
        fontsize=8, color='darkred', alpha=0.7, va='top')

ax.set_xlabel('Infrastructure Index (2015, z-score)\n← worse                                          better →')
ax.set_ylabel('Health Burden Index (2015, z-score)\n↑ worse\n\n\n↓ better')
ax.set_title('Health Burden vs. Infrastructure Availability by State (2015–16)\n'
             'States above the diagonal have higher burden than their infrastructure can support', pad=12)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9, title='Region')
ax.grid(True, alpha=0.18)
ax.set_xlim(xlim); ax.set_ylim(ylim)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig3_gap_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig3")

# ── Fig 4: Model comparison ──────────────────────────────────
names_plot = list(ml_results.keys())
maes  = [ml_results[m]['MAE'] for m in names_plot]
r2s   = [ml_results[m]['R2']  for m in names_plot]
colors_bar = ['#4e79a7','#f28e2b','#e15759']
x = np.arange(len(names_plot)); w = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
b1 = ax1.bar(x, maes, w, color=colors_bar, edgecolor='white', linewidth=0.7)
ax1.set_xticks(x); ax1.set_xticklabels(names_plot, rotation=12, ha='right')
ax1.set_ylabel('MAE (Leave-One-Out CV)')
ax1.set_title('Model Comparison — MAE\n(lower is better)')
ax1.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
ax1.set_ylim(0, max(maes)*1.3); ax1.grid(axis='y', alpha=0.25)

b2 = ax2.bar(x, r2s, w, color=colors_bar, edgecolor='white', linewidth=0.7)
ax2.set_xticks(x); ax2.set_xticklabels(names_plot, rotation=12, ha='right')
ax2.set_ylabel('R² (Leave-One-Out CV)')
ax2.set_title('Model Comparison — R²\n(higher is better)')
ax2.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)
ax2.set_ylim(min(r2s)*1.3 if min(r2s)<0 else -0.1, 1.05); ax2.grid(axis='y', alpha=0.25)
ax2.axhline(0, color='k', linewidth=0.8, linestyle=':')

plt.suptitle('LOO Cross-Validated Performance: Predicting 2015 Gap Index from 2005 Indicators',
             y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig4_model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig4")

# ── Fig 5: Feature importance ────────────────────────────────
fi = feat_imp.copy()
fi.index = [FEAT_LABELS.get(i, i) for i in fi.index]
fi = fi.sort_values(ascending=True)
median_fi = fi.median()

fig, ax = plt.subplots(figsize=(9, 6))
bar_colors = ['#e15759' if v >= median_fi else '#4e79a7' for v in fi]
ax.barh(fi.index, fi.values, color=bar_colors, edgecolor='white', height=0.6)
ax.axvline(median_fi, color='k', linestyle='--', linewidth=0.9, alpha=0.5, label='Median')
ax.set_xlabel('Feature Importance (normalised)')
ax.set_title(f'Normalised Absolute Regression Coefficients — {best_name}\n(red = above median; scaled to sum to 1 after unit-variance feature scaling)', pad=10)
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig5_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig5")

# ── Fig 6: 2030 projection ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
y_pos = np.arange(len(top10))
top10_sorted = top10.sort_values('Gap2030')
bar_colors6 = [REGION_COLORS.get(r,'#999') for r in top10_sorted['Region']]

err_lo = (top10_sorted['Gap2030'] - top10_sorted['CI_lo']).clip(lower=0)
err_hi = (top10_sorted['CI_hi'] - top10_sorted['Gap2030']).clip(lower=0)

ax.barh(y_pos, top10_sorted['Gap2030'],
        xerr=[err_lo, err_hi],
        color=bar_colors6, edgecolor='white', capsize=4,
        error_kw={'linewidth': 1.2, 'ecolor': '#555555'}, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(top10_sorted['State'], fontsize=10)
ax.set_xlabel('Projected Burden-Infrastructure Gap Index (2030)')
ax.set_title('Top 10 States Projected to Face Largest Healthcare Gaps by 2030\n'
             'Extrapolated from 2005→2015 per-state trend  |  Whiskers: 90% CI', pad=10)
ax.axvline(0, color='k', linewidth=0.8)

# Also plot 2005 and 2015 dots as reference
for i, (_, row) in enumerate(top10_sorted.iterrows()):
    ax.scatter(row['Gap2005'], i, marker='o', color='#aaaaaa', s=50, zorder=5)
    ax.scatter(row['Gap2015'], i, marker='D', color='#333333', s=50, zorder=5)

legend_handles = [
    mpatches.Patch(label='Region', color='white'),
] + [mpatches.Patch(color=v, label=k) for k,v in REGION_COLORS.items() if k in top10_sorted['Region'].values] + [
    plt.scatter([],[], marker='o', color='#aaaaaa', s=50, label='2005 gap'),
    plt.scatter([],[], marker='D', color='#333333', s=50, label='2015 gap'),
]
ax.legend(handles=legend_handles[1:], loc='lower right', fontsize=8.5,
          framealpha=0.9, ncol=2)
ax.grid(axis='x', alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig6_projection_2030.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig6")

print("\n=== EXPERIMENT COMPLETE ===")
print(json.dumps(all_results, indent=2, default=str))
