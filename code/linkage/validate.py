"""Stage 4: validation diagnostics — blocking recall, calibration, audit sample."""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
OUT="/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility/processed data/linkage"
os.makedirs(OUT, exist_ok=True)
P = pd.read_parquet(f"{S}/pairs_scored.parquet")
O = pd.read_parquet(f"{S}/orbis_master.parquet")
G = pd.read_parquet(f"{S}/gd_pool.parquet")

print("="*74); print("A. BLOCKING RECALL (can name/ticker blocking find the website-verified match?)")
anc = P[P['anchor']==1][['bvd','cid','block_src','ts','tset','idf_cov','nkey_eq','tick_eq']].copy()
anc['found_wo_domain'] = anc['block_src'].str.replace('dom','',regex=False).str.strip('|').str.len()>0
anc['via'] = anc['block_src']
print(f"  website-anchored firms                     : {len(anc):,}")
print(f"  anchor also retrieved WITHOUT the domain key: {anc['found_wo_domain'].mean():.1%}")
print("  which non-domain key retrieved it:")
for k in ['fuzzy','nkey','ckey','tick','domstem']:
    print(f"    {k:9s} {anc['block_src'].str.contains(k).mean():6.1%}")

print("\n"+"="*74); print("B. CALIBRATION OF THE RANK-1 POSTERIOR (held-out anchored firms)")
H = pd.read_parquet(f"{S}/holdout_scored.parquet")
r1 = H[H['rank']==1].assign(correct=lambda d: d['y']==1)
bins=[(0,.5),(.5,.7),(.7,.8),(.8,.9),(.9,.95),(.95,.99),(.99,1.01)]
print(f"  {'p_top bin':>14} {'n':>7} {'share correct':>14}")
for lo,hi in bins:
    m=(r1['p_best']>=lo)&(r1['p_best']<hi)
    if m.sum(): print(f"  [{lo:.2f},{hi:.2f}) {m.sum():7,} {r1.loc[m,'correct'].mean():13.1%}")
print("  Held-out firms only; model score with NO website/ticker tier override, so this is")
print("  the honest accuracy for a firm whose website Orbis does not record.")

print("\n"+"="*74); print("C. AMBIGUITY STRUCTURE")
b = P[P['rank']==1]
print(f"  firms with candidates      : {P['bvd'].nunique():,}")
print(f"  p_top >= .95               : {(b['p_best']>=.95).sum():,}")
print(f"  p_top .90-.95              : {((b['p_best']>=.90)&(b['p_best']<.95)).sum():,}")
print(f"  p_top .50-.90              : {((b['p_best']>=.50)&(b['p_best']<.90)).sum():,}")
print(f"  p_top < .50                : {(b['p_best']<.50).sum():,}")
m2 = P[P['rank']==2].set_index('bvd')['p_best']
b = b.set_index('bvd').assign(margin=lambda d: d['p_best']-m2.reindex(d.index).fillna(0))
print(f"  median margin (p1 - p2)    : {b['margin'].median():.3f}")
print(f"  firms where p2 > .25       : {(m2>.25).sum():,}")

print("\n"+"="*74); print("D. MANY-TO-ONE")
acc = P[(P['rank']==1)&(P['p_best']>=.9)]
d = acc.groupby('cid').size()
print(f"  accepted links                          : {len(acc):,}")
print(f"  Glassdoor employers used by >1 Orbis firm: {(d>1).sum():,}")
if (d>1).sum():
    top = d[d>1].sort_values(ascending=False).head(5)
    nm = G.set_index('cid')['name']
    for cid,k in top.items(): print(f"     {nm.get(cid,'?')[:40]:42s} claimed by {k} Orbis ids")

# stratified audit sample
rng = np.random.default_rng(11)
aud=[]
for lo,hi,n in [(.99,1.01,30),(.9,.99,30),(.7,.9,25),(.5,.7,20),(.2,.5,20),(0,.2,15)]:
    s = P[(P['rank']==1)&(P['p_best']>=lo)&(P['p_best']<hi)]
    if len(s): aud.append(s.sample(min(n,len(s)), random_state=int(lo*100)))
Oa = O[['bvd','name_legal','country','city','website','ticker']].rename(
        columns={'city':'orbis_city','website':'orbis_website','ticker':'orbis_ticker'})
Ga = G[['cid','name','hq','dom','tick','reviews','gind']].rename(
        columns={'name':'gd_name','hq':'gd_hq','dom':'gd_domain','tick':'gd_ticker',
                 'reviews':'gd_reviews','gind':'gd_industry'})
A = pd.concat(aud).merge(Oa,on='bvd',how='left').merge(Ga,on='cid',how='left')
A = A[['bvd','name_legal','country','orbis_city','orbis_website','orbis_ticker','cid','gd_name',
       'gd_hq','gd_domain','gd_ticker','gd_reviews','gd_industry','p_pair','p_best','p_none','tier',
       'block_src','ts','tset','idf_cov']]
A.insert(0,'verdict','')
A.sort_values('p_best',ascending=False).to_csv(f"{OUT}/audit_sample.csv", index=False)
print(f"\n  stratified audit sample ({len(A)} pairs) -> {OUT}/audit_sample.csv")
