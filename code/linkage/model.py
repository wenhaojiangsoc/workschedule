"""Stage 3: distant-supervision model + calibrated posteriors."""
import sys, os, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

P = pd.read_parquet(os.environ.get("PAIRS", f"{S}/pairs.parquet"))
O = pd.read_parquet(f"{S}/orbis_master.parquet")
G = pd.read_parquet(f"{S}/gd_pool.parquet")
log(f"pairs {P.shape}, orbis {len(O)}, gd pool {len(G)}")

# ---------- anchors: unique corporate website, most-reviewed employer on that domain ----------
odom_n = O.loc[O['dom']!='', 'dom'].value_counts()
gdom_n = G.loc[G['dom']!='', 'dom'].value_counts()
P['odom'] = P['bvd'].map(O.set_index('bvd')['dom'].to_dict()).fillna('')
# Website evidence = exact registrable domain OR shared domain stem (walmart.com / walmart.ca).
# Several Glassdoor pages often share one corporate domain (divisional pages such as
# "Amazon Flex"); the firm-level employer is the one carrying the review mass, so we rank
# eligible candidates by review count only.  No name information enters the anchor, so the
# name features stay uncontaminated by the labelling rule.
# A shared domain STEM (walmart.com / walmart.ca) usually means the same group, but the stem
# also collides across unrelated firms on different TLDs (mmc.co.jp vs mmc.org, wish.com vs
# wish.org).  Stem evidence therefore counts only with a corroborating name match; exact-domain
# evidence needs no such guard.
STEM_OK = P['stem_eq'] & ((P['ts'] >= 0.70) | P['nkey_eq'] | P['ckey_eq'])
uniq = P['odom'].map(lambda d: odom_n.get(d,9)<=2) & P['odom'].map(lambda d: gdom_n.get(d,9)<=5)
elig = (P['dom_eq'] | STEM_OK) & uniq
cand = P[elig].sort_values('lrev', ascending=False)
anchor = cand.groupby('bvd').head(1)[['bvd','cid']].assign(anchor=1)
P = P.merge(anchor, on=['bvd','cid'], how='left')
P['anchor'] = P['anchor'].fillna(0).astype(int)
pos_firms = set(P.loc[P['anchor']==1,'bvd'])
log(f"anchored firms: {len(pos_firms)}")

lab = P[P['bvd'].isin(pos_firms)].copy()
lab = lab[(lab['anchor']==1) | (~lab['dom_eq'])]
lab['y'] = lab['anchor']
log(f"training pairs {len(lab)}  pos {lab['y'].sum()}  neg {(lab['y']==0).sum()}")

FEATS = ['ts','tset','wr','jw','ts_brand','idf_jac','idf_cov','nkey_eq','ckey_eq','tick_eq',
         'ctry','state','city','pub_agree','lrev','lsal','lidx','size_ord']
X = lambda d: np.clip(np.nan_to_num(d[FEATS].astype(float).to_numpy(),
                                    nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

firms = np.array(sorted(pos_firms)); np.random.default_rng(20260912).shuffle(firms)
cut = int(0.7*len(firms)); tr_f = set(firms[:cut]); te_f = set(firms[cut:])
tr, te = lab[lab['bvd'].isin(tr_f)], lab[lab['bvd'].isin(te_f)]

def fit(train):
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.06,
                                    subsample=0.8, random_state=0)
    m = CalibratedClassifierCV(gb, method='isotonic', cv=4)
    m.fit(X(train), train['y'].to_numpy()); return m

m = fit(tr); pte = m.predict_proba(X(te))[:,1]
log(f"held-out AUC {roc_auc_score(te['y'],pte):.4f}  AP {average_precision_score(te['y'],pte):.4f}")
te = te.assign(p=pte)
top1 = te.sort_values('p',ascending=False).groupby('bvd').head(1)
log(f"held-out top-1 accuracy {top1['y'].mean():.4f} over {len(top1)} firms")
for lo,hi in [(0,.5),(.5,.8),(.8,.9),(.9,.95),(.95,1.01)]:
    msk=(top1['p']>=lo)&(top1['p']<hi)
    if msk.sum(): log(f"   p in [{lo},{hi}): n={msk.sum():5d} precision={top1.loc[msk,'y'].mean():.3f}")

# honest held-out posterior: model only, no deterministic tier override
ho = te.copy()
od = np.clip(ho['p'],1e-6,1-1e-6); ho['odds'] = od/(1-od)
ho['p_best'] = ho['odds']/ho.groupby('bvd')['odds'].transform('sum').add(1.0)
ho['p_none'] = 1.0/ho.groupby('bvd')['odds'].transform('sum').add(1.0)
ho = ho.sort_values(['bvd','p_best'],ascending=[True,False])
ho['rank'] = ho.groupby('bvd').cumcount()+1
ho.to_parquet(f"{S}/holdout_scored.parquet", index=False)
log("held-out posterior calibration (rank-1, model only):")
h1 = ho[ho['rank']==1]
for lo,hi in [(0,.5),(.5,.7),(.7,.8),(.8,.9),(.9,.95),(.95,.99),(.99,1.01)]:
    msk=(h1['p_best']>=lo)&(h1['p_best']<hi)
    if msk.sum(): log(f"   p_best [{lo:.2f},{hi:.2f}) n={msk.sum():5d} correct={h1.loc[msk,'y'].mean():.3f}")

mf = fit(lab)
P['p_model'] = mf.predict_proba(X(P))[:,1]
imp = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(X(lab), lab['y'])
log("importances: " + ", ".join(f"{f}={v:.3f}" for f,v in
    sorted(zip(FEATS, imp.feature_importances_), key=lambda t:-t[1])[:9]))

# one website-tier winner per firm; the rest of the domain family drops to a weaker floor
web = (P['dom_eq'] | STEM_OK)
winner = P[web].sort_values('lrev',ascending=False).groupby('bvd').head(1)[['bvd','cid']].assign(web_win=1)
P = P.merge(winner, on=['bvd','cid'], how='left'); P['web_win'] = P['web_win'].fillna(0)
P['tier'] = np.select(
    [(P['web_win']==1) & web, web, P['tick_eq'] & (P['ts']>=.55), P['nkey_eq'] & (P['ctry']>=.5)],
    ['website','website_shared','ticker','exact_name'], default='model')
P['p_pair'] = np.where(P['tier']=='website',        np.maximum(P['p_model'],0.97),
              np.where(P['tier']=='ticker',         np.maximum(P['p_model'],0.93),
              np.where(P['tier']=='website_shared', np.maximum(P['p_model'],0.35), P['p_model'])))
od = P['p_pair'].clip(1e-6,1-1e-6); P['odds'] = od/(1-od)
den = P.groupby('bvd')['odds'].transform('sum') + 1.0
P['p_best'] = P['odds']/den; P['p_none'] = 1.0/den
P = P.sort_values(['bvd','p_best'], ascending=[True,False])
P['rank'] = P.groupby('bvd').cumcount()+1
P.to_parquet(os.environ.get("SCORED", f"{S}/pairs_scored.parquet"), index=False)
log(f"saved; firms with candidates {P['bvd'].nunique()}")
