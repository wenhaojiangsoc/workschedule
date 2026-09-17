"""Re-apply evidence floors and posteriors to saved model scores (no refit)."""
import os, numpy as np, pandas as pd
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
P=pd.read_parquet(f"{S}/pairs_scored.parquet")
P['p_pair']=np.where(P['tier']=='website',        np.maximum(P['p_model'],0.97),
            np.where(P['tier']=='ticker',         np.maximum(P['p_model'],0.93),
            np.where(P['tier']=='website_shared', np.maximum(P['p_model'],0.35), P['p_model'])))
od=P['p_pair'].clip(1e-6,1-1e-6); P['odds']=od/(1-od)
den=P.groupby('bvd')['odds'].transform('sum')+1.0
P['p_best']=P['odds']/den; P['p_none']=1.0/den
P=P.sort_values(['bvd','p_best'],ascending=[True,False])
P['rank']=P.groupby('bvd').cumcount()+1
P.to_parquet(f"{S}/pairs_scored.parquet", index=False)
b=P[P['rank']==1]
print(f"p_top>=.95 {(b['p_best']>=.95).sum():,} | .90-.95 {((b['p_best']>=.90)&(b['p_best']<.95)).sum():,} "
      f"| .50-.90 {((b['p_best']>=.50)&(b['p_best']<.90)).sum():,} | <.50 {(b['p_best']<.50).sum():,}")
