"""Stage 1: blocking. Orbis firms -> Glassdoor candidate employers."""
import sys, os, time, pickle, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
sys.path.insert(0,S)
from rapidfuzz import fuzz, process as rfproc
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

MINREV = int(os.environ.get("MINREV","5"))
O  = pd.read_parquet(f"{S}/orbis_master.parquet")
NV = pd.read_parquet(f"{S}/orbis_names.parquet")
G  = pd.read_parquet(f"{S}/gd_canon2.parquet")
G  = G[G['reviews']>=MINREV].reset_index(drop=True)
SUB = int(os.environ.get("SUBSET","0"))
if SUB:
    keep = set(O.sample(SUB, random_state=7)['bvd'])
    O = O[O['bvd'].isin(keep)]; NV = NV[NV['bvd'].isin(keep)]
log(f"orbis {len(O)} firms, {len(NV)} variants; glassdoor {len(G)} employers (reviews>={MINREV})")
G.to_parquet(f"{S}/gd_pool.parquet", index=False)

gd_nname = G['nname'].to_numpy(); gd_rev = G['reviews'].to_numpy(float)
def build(col):
    d = defaultdict(list)
    for i, v in enumerate(G[col].to_numpy()):
        if isinstance(v,str) and v: d[v].append(i)
    return d
IDX = {c: build(c) for c in ['dom','dom_stem','tick','nkey','ckey']}
log("hash indexes: " + ", ".join(f"{k}={len(v)}" for k,v in IDX.items()))

STOP = {'the','and','of','for','group','holdings','holding','company','corporation','inc','ltd',
        'llc','international','services','service','solutions','systems','technologies','global',
        'usa','us','national','american','co','plc','sa','ag','limited'}
tok_idx = defaultdict(list)
for i, nm in enumerate(gd_nname):
    if not isinstance(nm,str): continue
    for t in set(nm.split()):
        if len(t)>2 and t not in STOP: tok_idx[t].append(i)
DF = {t: len(v) for t,v in tok_idx.items()}
N = len(G)
IDF = {t: np.log(N/(1+df)) for t,df in DF.items()}
pickle.dump(IDF, open(f"{S}/idf.pkl","wb"))
log(f"token index {len(tok_idx)} tokens")

MAXPOST, KEEP = 80000, 150
nv_by_firm = defaultdict(list)
for r in NV.itertuples(index=False):
    nv_by_firm[r.bvd].append((r.nname, r.nkey, r.ckey, r.src))

cands = defaultdict(dict)
def add(b, rows, tag, cap=None):
    if cap and len(rows) > cap:
        rows = sorted(rows, key=lambda i: -gd_rev[i])[:cap]
    d = cands[b]
    for i in rows: d.setdefault(i, set()).add(tag)

for r in O.itertuples(index=False):
    b = r.bvd
    if isinstance(r.dom,str) and r.dom:                   add(b, IDX['dom'].get(r.dom,[]), 'dom')
    if isinstance(r.dom_stem,str) and len(r.dom_stem)>2:  add(b, IDX['dom_stem'].get(r.dom_stem,[]), 'domstem', 40)
    if isinstance(r.tick,str) and r.tick:                 add(b, IDX['tick'].get(r.tick,[]), 'tick', 40)
for b, variants in nv_by_firm.items():
    for nname, nkey, ckey, src in variants:
        add(b, IDX['nkey'].get(nkey,[]), 'nkey', 60)
        add(b, IDX['ckey'].get(ckey,[]), 'ckey', 60)
log(f"key blocking: {sum(len(v) for v in cands.values())} pairs over {len(cands)} firms")

for n,(b, variants) in enumerate(nv_by_firm.items()):
    pool = set()
    for nname, nkey, ckey, src in variants:
        toks = sorted([t for t in set(nname.split()) if len(t)>2 and t in DF], key=lambda t: DF[t])
        got = []
        for t in toks[:3]:
            if DF[t] <= MAXPOST: got.extend(tok_idx[t])
            if len(got) > 50000: break
        if not got and toks:
            got = sorted(tok_idx[toks[0]], key=lambda i: -gd_rev[i])[:20000]
        pool.update(got)
        if len(pool) > 80000: break
    if not pool: continue
    pool = list(pool); pn = [gd_nname[i] for i in pool]
    best = {}
    for q in {v[0] for v in variants}:
        sc = rfproc.cdist([q], pn, scorer=fuzz.token_sort_ratio, score_cutoff=68, workers=-1)[0]
        nz = np.nonzero(sc)[0]
        if len(nz) > KEEP: nz = nz[np.argsort(-sc[nz])[:KEEP]]
        for j in nz:
            i = pool[j]
            if sc[j] > best.get(i,0): best[i] = sc[j]
    top = sorted(best.items(), key=lambda kv: -kv[1])[:KEEP]
    add(b, [i for i,_ in top], 'fuzzy')
    if n and n % 2000 == 0: log(f"  fuzzy {n}/{len(nv_by_firm)}")
log(f"after fuzzy: {sum(len(v) for v in cands.values())} pairs")
pickle.dump(dict(cands), open(os.environ.get("CANDS", f"{S}/cands.pkl"),"wb"))
log("saved")
