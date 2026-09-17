"""Firms with no candidate or only a weak one, flagged for non-operating vehicles."""
import re, numpy as np, pandas as pd
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
OUT="/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility/processed data/linkage"
P=pd.read_parquet(f"{S}/pairs_scored.parquet"); O=pd.read_parquet(f"{S}/orbis_master.parquet")
O=O.merge(P[P['rank']==1][['bvd','p_best']], on='bvd', how='left')
O['status']=np.where(O['p_best'].isna(),'no_candidate',
            np.where(O['p_best']>=.90,'auto_accept',np.where(O['p_best']>=.50,'review','weak')))
SPV=r'\b(?:FUND|TRUST|SICAV|SPV|NO\.? ?\d|INVESTMENT|CAPITAL PARTNERS|L\.?P\.?$|PROPERTY|REALTY|PENSION|NOMINEE|TRUSTEE|SECURITIES|ISSUER)'
O['looks_shell']=O['name_legal'].fillna('').str.upper().str.contains(SPV,regex=True)
u=O[O['status'].isin(['weak','no_candidate'])]
u[['bvd','name_legal','name_internat','country','etype','city','website','ticker','listed',
   'status','p_best','looks_shell','n_years','yr_min','yr_max']]\
 .rename(columns={'bvd':'orbis_id','name_legal':'orbis_name','p_best':'p_top'})\
 .sort_values(['status','orbis_id']).to_csv(f"{OUT}/unmatched_firms.csv",index=False)
print(f"unmatched_firms.csv: {len(u):,} firms "
      f"({(u['status']=='no_candidate').sum():,} no candidate, {(u['status']=='weak').sum():,} weak); "
      f"{u['looks_shell'].mean():.1%} carry fund/SPV-style names; {(u['dom']=='').mean():.1%} have no website")
