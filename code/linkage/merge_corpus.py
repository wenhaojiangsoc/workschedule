"""One canonical Glassdoor review corpus from all four scrape products.

Sources: the cleaned domestic file, the two foreign files it omits, and the collaborator's
2024-2026 top-up. De-duplicated on reviewID, harmonised to the union of columns, and carrying the
resolved canonical employer id so downstream joins cannot hit the scraped-index trap.

Written as Parquet: 18.6M rows of review text is several times smaller and far quicker to read back
than CSV, and pandas reads it with read_parquet. The de-duplication decision is taken from the key
file's own `source` column, so no growing membership set is needed."""
import time, warnings, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
warnings.filterwarnings('ignore')
BOX="/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility"
RAW=f"{BOX}/raw data/Glassdoor review"
DEST=f"{BOX}/processed data/review/glassdoor_reviews_merged.parquet"
FILES=[("domestic_clean",  f"{BOX}/processed data/review/glassdoor_reviews_clean.csv"),
       ("foreign_00",      f"{RAW}/df_reviews_foreign_array_00.csv"),
       ("foreign_01",      f"{RAW}/df_reviews_foreign_array_01.csv"),
       ("topup_2024_2026", f"{RAW}/reviews_2024_2026.csv")]
COLS=['reviewID','date','year','companyID','gd_id','companyname','source',
      'rating_overall','rating_business_outlook','rating_worklife_balance','rating_culture_values',
      'rating_diversity_inclusion','rating_leadership','rating_recommend_friend',
      'rating_career_opportunity','rating_compensation','ceo_name','ceo_photo','is_current_job',
      'length_employment','employment_status','job_title','job_location','job_location_type',
      'review_pros','review_cons','review_summary','review_advice','count_helpful',
      'count_nothelpful','language','len_review_pros','len_review_cons','len_review_summary',
      'len_review_advice']
# reviewID and gd_id are fully numeric, so the lookup is three int64 arrays plus a binary search
# rather than two 18.6M-entry Python dicts, which otherwise cost ~8 GB and thrash the machine
key = pd.read_parquet(f"{BOX}/processed data/linkage/review_id_to_employer.parquet",
                      columns=['reviewID','gd_id','source'])
TAGS=[t for t,_ in FILES]
K_RID = key['reviewID'].to_numpy(dtype=np.int64)
K_GD  = key['gd_id'].to_numpy(dtype=np.int64)
K_SRC = key['source'].astype(str).map({t:i for i,t in enumerate(TAGS)}).to_numpy(dtype=np.int8)
order = np.argsort(K_RID, kind='stable')
K_RID, K_GD, K_SRC = K_RID[order], K_GD[order], K_SRC[order]
del key, order
print(f"key universe {len(K_RID):,} reviews, lookup arrays {K_RID.nbytes/2**20:.0f} MB", flush=True)

EMITTED = np.zeros(len(K_RID), dtype=bool)   # one bit per key row: has this review been written?

def lookup(rid_str, tag_code):
    """-> (mask of rows belonging to this source and not yet written, their gd_id)"""
    rid = pd.to_numeric(rid_str, errors='coerce')
    ok = rid.notna().to_numpy()
    r = rid.fillna(-1).to_numpy(dtype=np.int64)
    pos = np.searchsorted(K_RID, r)
    pos_c = np.clip(pos, 0, len(K_RID)-1)
    hit = ok & (K_RID[pos_c] == r)
    keep = hit & (K_SRC[pos_c] == tag_code) & ~EMITTED[pos_c]
    # EMITTED is read before it is written, so a reviewID repeated inside THIS chunk would pass
    # twice; keep only the first occurrence of each key position
    idx = np.nonzero(keep)[0]
    if len(idx):
        _, first = np.unique(pos_c[idx], return_index=True)
        keep = np.zeros_like(keep); keep[idx[first]] = True
    EMITTED[pos_c[keep]] = True        # a reviewID repeated later in the file is then skipped
    gd = np.where(keep, K_GD[pos_c], -1)
    return keep, gd

t0=time.time(); writer=None; written=0
for tag_code,(tag,path) in enumerate(FILES):
    for ch in pd.read_csv(path, dtype=str, chunksize=500_000, on_bad_lines='skip',
                          encoding='utf-8', encoding_errors='replace'):
        ch['reviewID']=ch['reviewID'].astype(str).str.replace(r'\.0$','',regex=True)
        # a reviewID belongs to exactly one source in the key file; that decides the duplicate
        keep, gd = lookup(ch['reviewID'], tag_code)
        ch=ch[keep].copy(); gd=gd[keep]
        if not len(ch): continue
        ch['companyID']=ch['companyID'].astype(str).str.replace(r'\.0$','',regex=True)
        ch['gd_id']=pd.Series(gd, index=ch.index).astype('int64').astype(str)
        ch['year']=pd.to_numeric(ch['date'].astype(str).str.slice(0,4), errors='coerce').astype('Int16')
        ch['source']=tag
        for c in ['review_pros','review_cons','review_summary','review_advice']:
            lc=f"len_{c}"
            if lc not in ch.columns or ch[lc].isna().all():
                ch[lc]=ch[c].fillna('').astype(str).str.split().str.len()
        for c in COLS:
            if c not in ch.columns: ch[c]=pd.NA
        out=ch[COLS].astype({c:'string' for c in COLS if c not in ('year',)})
        tbl=pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer=pq.ParquetWriter(DEST, tbl.schema, compression='zstd')
        writer.write_table(tbl); written+=len(out)
    print(f"  {tag:16s} cumulative {written:>10,}  [{time.time()-t0:.0f}s]", flush=True)
writer.close()
print(f"\nwrote {DEST}\n  {written:,} unique reviews  [{time.time()-t0:.0f}s]", flush=True)
assert written == int(EMITTED.sum()) == len(K_RID), (written, int(EMITTED.sum()), len(K_RID))
print("  row count matches the key universe exactly", flush=True)
