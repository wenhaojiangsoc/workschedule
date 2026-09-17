"""Assemble the deliverable crosswalk + diagnostics."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
OUT="/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility/processed data/linkage"
import os; os.makedirs(OUT, exist_ok=True)

P = pd.read_parquet(f"{S}/pairs_scored.parquet")
O = pd.read_parquet(f"{S}/orbis_master.parquet")
G = pd.read_parquet(f"{S}/gd_canon2.parquet")

K = P[P['rank']<=3].merge(
      O[['bvd','name_legal','name_internat','country','etype','city','us_state','website','ticker',
         'listed','n_years','yr_min','yr_max']], on='bvd', how='left').merge(
      G[['cid','name','hq','reviews','salaries','dom','tick','gtype','gind','n_index']],
      on='cid', how='left', suffixes=('','_gd'))

K = K.rename(columns={'bvd':'orbis_id','name_legal':'orbis_name','name_internat':'orbis_name_intl',
    'country':'orbis_country','etype':'orbis_entity_type','city':'orbis_city','us_state':'orbis_state',
    'website':'orbis_website','ticker':'orbis_ticker','listed':'orbis_listed','cid':'gd_id',
    'name':'gd_name','hq':'gd_hq','reviews':'gd_reviews','salaries':'gd_salaries','dom':'gd_domain',
    'tick':'gd_ticker','gtype':'gd_type','gind':'gd_industry','n_index':'gd_n_indices',
    'p_pair':'p_match','p_best':'p_top','p_none':'p_no_match','tier':'evidence','block_src':'blocked_by'})

K['decision'] = np.select(
    [(K['rank']==1)&(K['p_top']>=0.90), (K['rank']==1)&(K['p_top']>=0.50)],
    ['auto_accept','review'], default=np.where(K['rank']==1,'weak','alternate'))

# how many Orbis firms take this Glassdoor employer as their accepted rank-1 match
acc_mask = (K['rank']==1) & (K['p_top']>=0.90)
claims = K.loc[acc_mask].groupby('gd_id')['orbis_id'].nunique()
K['n_orbis_claiming'] = K['gd_id'].map(claims).fillna(0).astype(int)
K['link_type'] = np.where(K['n_orbis_claiming']<=1,'one_to_one',
                 np.where(K['n_orbis_claiming']<=4,'small_group','group_shared'))
K['is_fund_vehicle'] = K['orbis_entity_type'].isin(['E','F'])

cols = ['orbis_id','orbis_name','orbis_name_intl','orbis_country','orbis_entity_type','orbis_city',
        'orbis_state','orbis_website','orbis_ticker','orbis_listed','n_years','yr_min','yr_max',
        'rank','gd_id','gd_name','gd_hq','gd_reviews','gd_salaries','gd_domain','gd_ticker','gd_type',
        'gd_industry','gd_n_indices','p_match','p_top','p_no_match','evidence','blocked_by','decision',
        'n_orbis_claiming','link_type','is_fund_vehicle',
        'sim_max','sim_brand','nkey_eq','dom_eq','tick_eq','ctry','state','city']
K = K[[c for c in cols if c in K.columns]].sort_values(['orbis_id','rank'])
K.to_csv(f"{OUT}/orbis_glassdoor_candidates.csv", index=False)

best = K[K['rank']==1].copy()
best.to_csv(f"{OUT}/orbis_glassdoor_bestmatch.csv", index=False)

print("=== LINKAGE SUMMARY")
print(f"Orbis firms in sample          : {len(O):,}")
print(f"  with >=1 candidate           : {K['orbis_id'].nunique():,}")
print(f"  no candidate at all          : {len(O)-K['orbis_id'].nunique():,}")
print()
for lab, m in [('p_top >= 0.95', best['p_top']>=.95), ('p_top 0.90-0.95', (best['p_top']>=.90)&(best['p_top']<.95)),
               ('p_top 0.50-0.90', (best['p_top']>=.50)&(best['p_top']<.90)), ('p_top < 0.50', best['p_top']<.50)]:
    print(f"  {lab:22s} {m.sum():6,}  ({m.mean():5.1%} of firms with candidates)")
print()
print("evidence tier of the rank-1 candidate:")
print(best['evidence'].value_counts().to_string())
print()
print("decision:"); print(best['decision'].value_counts().to_string())
print()
acc = best[best['decision']=='auto_accept']
print(f"auto-accepted firms: {len(acc):,}")
print(f"  median Glassdoor reviews: {acc['gd_reviews'].median():,.0f}")
print(f"  with >=25 reviews: {(acc['gd_reviews']>=25).mean():.1%}   >=100: {(acc['gd_reviews']>=100).mean():.1%}")
print(f"  total reviews covered: {acc['gd_reviews'].sum():,.0f}")
print()
print("by Orbis country (top 10, auto_accept rate):")
t = best.groupby('orbis_country').agg(n=('orbis_id','size'), auto=('decision', lambda s:(s=='auto_accept').mean()))
print(t.sort_values('n',ascending=False).head(10).assign(auto=lambda d:(100*d['auto']).round(1)).to_string())
print()
print("by Orbis entity type:")
t2 = best.groupby('orbis_entity_type').agg(n=('orbis_id','size'), auto=('decision', lambda s:(s=='auto_accept').mean()))
print(t2.sort_values('n',ascending=False).assign(auto=lambda d:(100*d['auto']).round(1)).to_string())

# many-to-one diagnostics
dup = best[best['decision']=='auto_accept'].groupby('gd_id').size()
print(f"\nGlassdoor employers claimed by >1 Orbis firm (auto_accept): {(dup>1).sum():,}")
print(f"  one-to-one links      : {(best['link_type']=='one_to_one').sum():,}")
print(f"  small group (2-4)     : {(best['link_type']=='small_group').sum():,}")
print(f"  group shared (5+)     : {(best['link_type']=='group_shared').sum():,}")
print(f"  accepted links that are fund/financial vehicles (Orbis type E/F): {acc['is_fund_vehicle'].sum():,}")
print(f"files written to {OUT}")
