"""Crosswalk from the review files' scraped companyID to canonical employer and Orbis firm,
plus the Orbis firm-year review-count panel.

The review CSVs key on the index that was queried, not the employer that was returned, so a
direct join on companyID splits a firm across indices and mismatches most large employers.
`review_index_to_orbis.csv` carries the full chain; `orbis_year_review_panel.csv` is the
firm-year cell count the classifier output will aggregate into."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
BOX="/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility"
OUT=f"{BOX}/processed data/linkage"

ap = pd.read_csv(f"{S}/gd_apollo.csv", dtype=str, usecols=['queried_id','canon_id'])
ap = ap[ap['queried_id'].notna() & ap['canon_id'].notna()].drop_duplicates('queried_id')
idx2canon = dict(zip(ap['queried_id'], ap['canon_id']))
cm = pd.read_parquet(f"{S}/collapse_map.parquet")
canon2rep = dict(zip(cm['cid'].astype(str), cm['rep'].astype(str)))

# counts come from the de-duplicated per-review key, so the two files cannot disagree
R = pd.read_parquet(f"{OUT}/review_id_to_employer.parquet")
rc = (R.groupby([R['review_companyID'].rename('idx'), R['year'].astype('float')], as_index=False)
        .size().rename(columns={'size':'n'}))
rc['idx']=rc['idx'].astype(str)
tot = rc.groupby('idx')['n'].sum()

# ---- one row per scraped index ----
X = pd.DataFrame({'review_companyID': pd.Index(rc['idx'].unique())})
X['canon_id'] = X['review_companyID'].map(idx2canon).fillna(X['review_companyID'])
X['gd_id']    = X['canon_id'].map(canon2rep).fillna(X['canon_id'])
X['resolved_by_apollo'] = X['review_companyID'].isin(idx2canon)
X['reviews_at_index'] = X['review_companyID'].map(tot).fillna(0).astype(int)
N_IDX, N_REV = len(X), int(X['reviews_at_index'].sum())

K = pd.read_csv(f"{OUT}/orbis_glassdoor_bestmatch.csv", dtype={'gd_id':str},
                usecols=['orbis_id','orbis_name','orbis_country','orbis_entity_type','gd_id',
                         'gd_name','p_top','evidence','decision','link_type','is_fund_vehicle'])
# several Orbis entities can share one Glassdoor page (corporate groups): keep every pairing,
# but record the fan-out so nobody double counts reviews by accident
K['n_orbis_on_this_gd_id'] = K.groupby('gd_id')['orbis_id'].transform('nunique')
X = X.merge(K, on='gd_id', how='left').sort_values('reviews_at_index', ascending=False)
X.to_csv(f"{OUT}/review_index_to_orbis.csv", index=False)

acc = X[X['decision']=='auto_accept']
acc_idx = acc.drop_duplicates('review_companyID')
print(f"review_index_to_orbis.csv  {len(X):,} rows over {N_IDX:,} scraped indices")
print(f"  index resolved to a canonical employer via Apollo : {X.drop_duplicates('review_companyID')['resolved_by_apollo'].mean():6.1%}")
print(f"  indices reaching an accepted Orbis link           : {acc_idx['review_companyID'].nunique():,}")
print(f"  reviews under an accepted link                    : {acc_idx['reviews_at_index'].sum():,} of {N_REV:,} "
      f"({acc_idx['reviews_at_index'].sum()/N_REV:.1%})")
fan = acc.groupby('orbis_id')['review_companyID'].nunique()
print(f"  Orbis firms whose reviews span >1 scraped index   : {(fan>1).sum():,} (max {fan.max()})")
print(f"  Orbis firms reachable from review text            : {acc['orbis_id'].nunique():,} of 5,475 accepted links")

# ---- Orbis firm-year panel ----
m = X[X['decision']=='auto_accept'][['review_companyID','orbis_id','orbis_name','orbis_country',
                                     'orbis_entity_type','gd_id','p_top','link_type','is_fund_vehicle']]
P = rc.merge(m, left_on='idx', right_on='review_companyID', how='inner')
P = (P.groupby(['orbis_id','orbis_name','orbis_country','orbis_entity_type','gd_id','p_top',
                'link_type','is_fund_vehicle','year'], dropna=False, as_index=False)['n'].sum()
       .rename(columns={'n':'n_reviews'}))
P['year']=P['year'].astype('Int64')

big3 = pd.read_csv(f"{BOX}/processed data/Orbis/big3_shares_allyears_merged.csv")
big3 = big3.rename(columns={'company_id':'orbis_id'})[['orbis_id','year','blackrock_pct',
                                                       'vanguard_pct','statestreet_pct']]
P = P.merge(big3, on=['orbis_id','year'], how='left')
P['in_ownership_panel'] = P['blackrock_pct'].notna()
P.sort_values(['orbis_id','year']).to_csv(f"{OUT}/orbis_year_review_panel.csv", index=False)

print(f"\norbis_year_review_panel.csv  {len(P):,} firm-year cells, {P['orbis_id'].nunique():,} firms, "
      f"{P['n_reviews'].sum():,} reviews")
ov = P[P['in_ownership_panel']]
print(f"  cells that also exist in the Big-3 ownership panel : {len(ov):,} ({len(ov)/len(P):.1%})")
print(f"  of 60,416 ownership company-years, matched to reviews: {len(ov):,} ({len(ov)/60416:.1%})")
print("\n  estimable cells by minimum reviews per firm-year:")
for k in [1,3,5,10,25,50]:
    s=ov[ov['n_reviews']>=k]
    print(f"    >= {k:>2d} reviews : {len(s):>7,} cells   {s['orbis_id'].nunique():>5,} firms")
print("\n  reviews per firm-year in the ownership panel:")
print("   ", ov['n_reviews'].describe(percentiles=[.25,.5,.75,.9]).round(1).to_dict())
