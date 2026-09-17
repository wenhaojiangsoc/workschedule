"""Per-review key: reviewID -> scraped index -> canonical Glassdoor employer -> year.

Deliberately stops at the Glassdoor employer. Join this to orbis_glassdoor_bestmatch.csv on
gd_id to reach Orbis; keeping the two steps apart means the analyst decides what to do about
the corporate groups where several Orbis entities share one employer page."""
import time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
BOX="/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility"
RAW=f"{BOX}/raw data/Glassdoor review"; OUT=f"{BOX}/processed data/linkage"
FILES=[("domestic_clean", f"{BOX}/processed data/review/glassdoor_reviews_clean.csv"),
       ("foreign_00",     f"{RAW}/df_reviews_foreign_array_00.csv"),
       ("foreign_01",     f"{RAW}/df_reviews_foreign_array_01.csv"),
       ("topup_2024_2026",f"{RAW}/reviews_2024_2026.csv")]
ap = pd.read_csv(f"{S}/gd_apollo.csv", dtype=str, usecols=['queried_id','canon_id'])
ap = ap[ap['queried_id'].notna() & ap['canon_id'].notna()].drop_duplicates('queried_id')
idx2canon = dict(zip(ap['queried_id'], ap['canon_id']))
cm = pd.read_parquet(f"{S}/collapse_map.parquet")
canon2rep = dict(zip(cm['cid'].astype(str), cm['rep'].astype(str)))
def to_gd(s):
    c = s.map(idx2canon).fillna(s)
    return c.map(canon2rep).fillna(c)

t0=time.time(); parts=[]; seen=set()
for tag,path in FILES:
    for ch in pd.read_csv(path, usecols=['reviewID','companyID','date'], dtype=str,
                          chunksize=1_000_000, on_bad_lines='skip',
                          encoding='utf-8', encoding_errors='replace'):
        ch=ch[ch['companyID'].notna() & ch['reviewID'].notna()]
        rid=ch['reviewID'].str.replace(r'\.0$','',regex=True)
        keep=~rid.isin(seen); seen.update(rid[keep])
        ch=ch[keep.values]
        if not len(ch): continue
        idx=ch['companyID'].str.replace(r'\.0$','',regex=True)
        parts.append(pd.DataFrame({
            'reviewID': rid[keep.values].values,
            'review_companyID': idx.values,
            'gd_id': to_gd(idx).values,
            'year': pd.to_numeric(ch['date'].astype(str).str.slice(0,4), errors='coerce').astype('Int16').values,
            'source': tag}))
    print(f"  {tag:16s} cumulative {sum(len(p) for p in parts):>10,}  [{time.time()-t0:.0f}s]", flush=True)
R=pd.concat(parts, ignore_index=True)
# the running `seen` set only catches duplicates across chunks; a reviewID repeated inside one
# chunk survives it, so de-duplicate once more on the assembled frame
n0=len(R); R=R.drop_duplicates('reviewID', keep='first').reset_index(drop=True)
print(f"  removed {n0-len(R):,} within-chunk duplicate reviewIDs", flush=True)
R['source']=R['source'].astype('category')
R.to_parquet(f"{OUT}/review_id_to_employer.parquet", index=False, compression='zstd')
print(f"\nreview_id_to_employer.parquet: {len(R):,} unique reviews, "
      f"{R['gd_id'].nunique():,} canonical employers  [{time.time()-t0:.0f}s]", flush=True)
