"""Second-pass collapse: merge residual duplicate Glassdoor indices of the same employer.

Glassdoor issues many indices per company; the Apollo cache resolves most to a canonical id,
but indices never covered by the retained pickles survive as singletons. Duplicates of one
employer share an identical normalised name and near-identical review counts, so within a
name group we cluster on (review proximity, headquarters compatibility) and keep the
domain-bearing / lowest-index member as representative.  Applied only to employers with >=5
reviews, i.e. the range where an Orbis match is plausible."""
import pandas as pd, numpy as np, time
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
t0=time.time()
g = pd.read_parquet(f"{S}/gd_canon.parquet")
g['cid_num'] = pd.to_numeric(g['cid'], errors='coerce').fillna(9e18)
g['has_dom'] = (g['dom'].fillna('')!='').astype(int)
LO = 5
hot = g[g['reviews']>=LO].copy()
cold = g[g['reviews']<LO].copy()
print(f"collapsing {len(hot):,} employers with >={LO} reviews ({len(cold):,} left as-is)", flush=True)

hot = hot.sort_values(['nname','has_dom','reviews','cid_num'], ascending=[True,False,False,True])
rep_of = {}
for nm, sub in hot.groupby('nname', sort=False):
    if len(sub)==1:
        rep_of[sub['cid'].iloc[0]] = sub['cid'].iloc[0]; continue
    reps = []   # (cid, reviews, hq)
    for cid, rev, hq in zip(sub['cid'], sub['reviews'], sub['hq'].fillna('')):
        hit = None
        for rcid, rrev, rhq in reps:
            close = abs(rev-rrev) <= 0.05*max(rev,rrev,1)
            hq_ok = (not hq) or (not rhq) or (hq==rhq)
            if close and hq_ok: hit = rcid; break
        if hit is None: reps.append((cid,rev,hq)); rep_of[cid]=cid
        else: rep_of[cid]=hit
print(f"  clustered in {time.time()-t0:.0f}s", flush=True)

hot['rep'] = hot['cid'].map(rep_of)
pd.DataFrame({'cid':list(rep_of), 'rep':[rep_of[k] for k in rep_of]}).to_parquet(f"{S}/collapse_map.parquet", index=False)
n_before = len(hot); n_after = hot['rep'].nunique()
print(f"  {n_before:,} -> {n_after:,} employers ({n_before-n_after:,} duplicates absorbed)", flush=True)

first = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
firstnz = lambda s: next((x for x in s if isinstance(x,str) and x), '')
A = hot.groupby('rep').agg(
    name=('name', first), nname=('nname','first'), cname=('cname','first'),
    nkey=('nkey','first'), ckey=('ckey','first'),
    dom=('dom', firstnz), dom_stem=('dom_stem', firstnz), tick=('tick', firstnz),
    gtype=('gtype', first), gsize=('gsize', first), grevenue=('grevenue', first),
    hq=('hq', first), hq_city=('hq_city', firstnz), hq_state=('hq_state', firstnz),
    hq_country_txt=('hq_country_txt', firstnz), is_us=('is_us','max'),
    gind=('gind', first), gdesc=('gdesc', first), reviews=('reviews','max'),
    salaries=('salaries','max'), yearFounded=('yearFounded', first),
    parentId=('parentId', first), parentName=('parentName', first),
    competitors=('competitors', first), n_index=('n_index','sum'),
    activeStatus=('activeStatus', first), n_merged=('cid','size')).reset_index().rename(columns={'rep':'cid'})
cold['n_merged']=1
out = pd.concat([A, cold[[c for c in A.columns if c in cold.columns]]], ignore_index=True)
out.to_parquet(f"{S}/gd_canon2.parquet", index=False)
print(f"\nFINAL universe {len(out):,} employers", flush=True)
for lo,hi in [(5,24),(25,99),(100,999),(1000,10**9)]:
    s=out[(out['reviews']>=lo)&(out['reviews']<=hi)]
    print(f"  {lo:>5}-{hi:<11} n={len(s):>9,}  dom={100*(s['dom'].fillna('')!='').mean():5.1f}%  tick={100*(s['tick'].fillna('')!='').mean():5.1f}%  hq={100*s['hq'].notna().mean():5.1f}%")
print("\ntop:"); print(out.nlargest(6,'reviews')[['cid','name','reviews','dom','tick','hq','n_merged']].to_string(index=False))
