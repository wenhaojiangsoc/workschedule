"""Stage 2: pair features."""
import sys, os, time, pickle, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
sys.path.insert(0,S)
from rapidfuzz import fuzz, process as rfproc
from rapidfuzz.distance import JaroWinkler
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

O  = pd.read_parquet(f"{S}/orbis_master.parquet").set_index('bvd', drop=False)
NV = pd.read_parquet(f"{S}/orbis_names.parquet")
G  = pd.read_parquet(f"{S}/gd_pool.parquet").reset_index(drop=True)
IDF = pickle.load(open(f"{S}/idf.pkl","rb"))
cands = pickle.load(open(os.environ.get("CANDS", f"{S}/cands.pkl"),"rb"))
log(f"{sum(len(v) for v in cands.values())} pairs over {len(cands)} firms")

ISO2NAME = {'US':'united states','GB':'united kingdom','AU':'australia','JP':'japan','DE':'germany',
 'CA':'canada','FR':'france','CH':'switzerland','IE':'ireland','NL':'netherlands','SE':'sweden',
 'ES':'spain','IT':'italy','CN':'china','IN':'india','BR':'brazil','KR':'south korea','TW':'taiwan',
 'HK':'hong kong','SG':'singapore','ZA':'south africa','MX':'mexico','NO':'norway','DK':'denmark',
 'FI':'finland','BE':'belgium','AT':'austria','PL':'poland','TH':'thailand','MY':'malaysia',
 'ID':'indonesia','PH':'philippines','NZ':'new zealand','IL':'israel','TR':'turkey','RU':'russia',
 'PT':'portugal','GR':'greece','LU':'luxembourg','BM':'bermuda','KY':'cayman islands','JE':'jersey',
 'CL':'chile','AR':'argentina','CO':'colombia','PE':'peru','AE':'united arab emirates',
 'SA':'saudi arabia','QA':'qatar','EG':'egypt','NG':'nigeria','VN':'vietnam','CZ':'czechia',
 'HU':'hungary','RO':'romania','GG':'guernsey','IM':'isle of man','MT':'malta','CY':'cyprus'}
SIZE_ORD = {'1 to 50 Employees':1,'51 to 200 Employees':2,'201 to 500 Employees':3,
 '501 to 1000 Employees':4,'1001 to 5000 Employees':5,'5001 to 10000 Employees':6,'10000+ Employees':7}
STOP = {'the','and','of','for','group','holdings','holding','company','corporation','inc','ltd',
        'llc','international','services','service','solutions','systems','technologies','global',
        'usa','us','national','american','co','plc','sa','ag','limited'}
DFLT = np.log(len(G)/1.0)
def wtoks(s):
    return {t: IDF.get(t, DFLT) for t in set(str(s).split()) if len(t)>2 and t not in STOP}

gd = {c: G[c].to_numpy() for c in ['cid','nname','nkey','ckey','dom','dom_stem','tick','hq_city',
      'hq_state','hq_country_txt','gtype','gsize','reviews','salaries','n_index']}
gd['is_us'] = G['is_us'].fillna(False).to_numpy(bool)
gd['size_ord'] = np.array([SIZE_ORD.get(x,0) for x in G['gsize'].fillna('')])
gd_wt = [wtoks(x) for x in gd['nname']]
log("gd token weights ready")

nv_by_firm = defaultdict(list)
for r in NV.itertuples(index=False):
    nv_by_firm[r.bvd].append((r.nname, r.nkey, r.ckey, r.src))
BRAND = {'brand','aka'}

rows=[]
for n,(b, cd) in enumerate(cands.items()):
    if b not in O.index: continue
    o = O.loc[b]
    if isinstance(o, pd.DataFrame): o = o.iloc[0]
    idx = np.fromiter(cd.keys(), dtype=np.int64)
    variants = nv_by_firm.get(b, [])
    if not len(idx) or not variants: continue
    qs = [v[0] for v in variants]; qk = [v[1] for v in variants]
    isbr = np.array([v[3] in BRAND for v in variants])
    cn = [gd['nname'][i] for i in idx]; ck = [gd['nkey'][i] for i in idx]
    TS  = rfproc.cdist(qs, cn, scorer=fuzz.token_sort_ratio, workers=-1).astype(np.float32)/100
    TSE = rfproc.cdist(qs, cn, scorer=fuzz.token_set_ratio,  workers=-1).astype(np.float32)/100
    WR  = rfproc.cdist(qs, cn, scorer=fuzz.WRatio,           workers=-1).astype(np.float32)/100
    JW  = rfproc.cdist(qk, ck, scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
    bestv = TS.argmax(0)
    ts, tse, wr, jw = TS.max(0), TSE.max(0), WR.max(0), JW.max(0)
    ts_brand = TS[isbr].max(0) if isbr.any() else np.zeros(len(idx), np.float32)
    best_src = np.array([variants[j][3] for j in bestv])

    qw = [wtoks(q) for q in qs]
    jac = np.zeros(len(idx), np.float32); cov = np.zeros(len(idx), np.float32)
    for k,i in enumerate(idx):
        gw = gd_wt[i]; bj=bc=0.0
        for w in qw:
            if not w and not gw: continue
            inter = sum(v for t,v in w.items() if t in gw)
            uni   = sum(w.values()) + sum(v for t,v in gw.items() if t not in w)
            bj = max(bj, inter/uni if uni else 0.0)
            bc = max(bc, inter/sum(w.values()) if w else 0.0)
        jac[k], cov[k] = bj, bc

    nkeys={v[1] for v in variants}; ckeys={v[2] for v in variants}
    nk_eq = np.array([gd['nkey'][i] in nkeys for i in idx])
    ck_eq = np.array([gd['ckey'][i] in ckeys for i in idx])
    odom, ostem, otick = o.get('dom',''), o.get('dom_stem',''), o.get('tick','')
    dom_eq  = np.array([bool(odom)  and gd['dom'][i]==odom       for i in idx])
    stem_eq = np.array([bool(ostem) and gd['dom_stem'][i]==ostem for i in idx])
    tick_eq = np.array([bool(otick) and gd['tick'][i]==otick     for i in idx])

    oc = o.get('country') or ''; ocn = ISO2NAME.get(oc,'')
    if oc=='US': ce = gd['is_us'][idx].astype(float)
    elif ocn:    ce = np.array([1.0 if ocn in (gd['hq_country_txt'][i] or '') else 0.0 for i in idx])
    else:        ce = np.full(len(idx), 0.5)
    known = np.array([bool(gd['hq_country_txt'][i]) or gd['is_us'][i] for i in idx])
    ctry = np.where(known, ce, 0.5)
    ost = o.get('us_state') or o.get('inc_state') or ''
    st  = np.array([1.0 if (ost and gd['hq_state'][i]==ost) else (0.5 if not ost or not gd['hq_state'][i] else 0.0) for i in idx])
    oci = o.get('ncity') or ''
    ci  = np.array([1.0 if (oci and gd['hq_city'][i]==oci) else (0.5 if not oci or not gd['hq_city'][i] else 0.0) for i in idx])
    listed = str(o.get('listed') or '').lower().startswith('listed')
    pub = np.array([1.0 if gd['gtype'][i]=='Company - Public' else 0.0 for i in idx])
    pub_agree = np.where(listed, pub, 1.0-pub)
    srcs=[cd[i] for i in idx]
    for k,i in enumerate(idx):
        rows.append((b, gd['cid'][i], int(i), float(ts[k]), float(tse[k]), float(wr[k]), float(jw[k]),
            float(ts_brand[k]), float(jac[k]), float(cov[k]), bool(nk_eq[k]), bool(ck_eq[k]),
            bool(dom_eq[k]), bool(stem_eq[k]), bool(tick_eq[k]), float(ctry[k]), float(st[k]),
            float(ci[k]), float(pub_agree[k]), float(np.log1p(gd['reviews'][i])),
            float(np.log1p(gd['salaries'][i])), float(np.log1p(gd['n_index'][i])),
            int(gd['size_ord'][i]), str(best_src[k]), '|'.join(sorted(srcs[k]))))
    if n and n % 2000 == 0: log(f"  {n}/{len(cands)}")

P = pd.DataFrame(rows, columns=['bvd','cid','grow','ts','tset','wr','jw','ts_brand','idf_jac','idf_cov',
    'nkey_eq','ckey_eq','dom_eq','stem_eq','tick_eq','ctry','state','city','pub_agree','lrev','lsal',
    'lidx','size_ord','best_src','block_src'])
log(f"features {P.shape}")
P.to_parquet(os.environ.get("PAIRS", f"{S}/pairs.parquet"), index=False)
log("saved")
