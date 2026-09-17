"""Canonical Glassdoor employer universe: collapse duplicate indices, merge Apollo + formatted CSV."""
import sys, re, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
S="/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
sys.path.insert(0,S)
from norm import norm_name, core_name, domain_of, domain_stem, norm_ticker, norm_city

ap = pd.read_csv(f"{S}/gd_apollo.csv", dtype=str)
ap = ap[ap['canon_id'].notna()]
print("apollo rows", len(ap), "| queried", ap['queried_id'].nunique(), "| canonical", ap['canon_id'].nunique(), flush=True)
ap['reviewCount'] = pd.to_numeric(ap['reviewCount'], errors='coerce').fillna(0)
ap['salaryCount'] = pd.to_numeric(ap['salaryCount'], errors='coerce').fillna(0)

# index -> canonical map
xmap = ap[['queried_id','canon_id']].dropna().drop_duplicates('queried_id')

# collapse apollo to canonical employers (richest row wins)
ap = ap.sort_values(['canon_id','reviewCount'], ascending=[True,False])
A = ap.groupby('canon_id', as_index=False).first()
print("canonical apollo employers", len(A), flush=True)

# formatted CSVs, keyed on queried index
ov = pd.read_csv(f"{S}/gd_overview_all.csv", dtype=str)
ov = ov.rename(columns={'companyid':'queried_id'})
ov['review_count'] = pd.to_numeric(ov['review_count'], errors='coerce').fillna(0)
ov['salary_count'] = pd.to_numeric(ov['salary_count'], errors='coerce').fillna(0)
ov = ov.merge(xmap, on='queried_id', how='left')
ov['cid'] = ov['canon_id'].fillna(ov['queried_id'])       # unmapped index = its own entity
print("csv rows", len(ov), "| mapped to apollo canon", ov['canon_id'].notna().mean().round(3), flush=True)

ov = ov.sort_values(['cid','review_count'], ascending=[True,False])
O = ov.groupby('cid', as_index=False).agg(
    companyname=('companyname','first'), type_c=('type','first'), revenue_c=('revenue','first'),
    headquarters_c=('headquarters','first'), size_c=('size','first'), description_c=('description','first'),
    mission_c=('mission','first'), industry_c=('industry_name','first'), state_c=('state','first'),
    review_count_c=('review_count','max'), salary_count_c=('salary_count','max'), n_index=('queried_id','size'))
print("csv canonical entities", len(O), flush=True)

g = A.merge(O, left_on='canon_id', right_on='cid', how='outer')
g['cid'] = g['cid'].fillna(g['canon_id'])
g['name'] = g['shortName'].fillna(g['companyname'])
g = g[g['name'].notna() & g['cid'].notna()]

pick = lambda a,b: g[a].fillna(g[b]) if a in g and b in g else (g[a] if a in g else g[b])
g['hq']       = pick('headquarters','headquarters_c')
g['gtype']    = pick('type','type_c')
g['gsize']    = pick('size','size_c')
g['grevenue'] = pick('revenue','revenue_c')
g['gdesc']    = pick('description','description_c')
g['gind']     = pick('industryName','industry_c')
g['reviews']  = np.fmax(pd.to_numeric(g['reviewCount'],errors='coerce').fillna(0),
                        pd.to_numeric(g['review_count_c'],errors='coerce').fillna(0))
g['salaries'] = np.fmax(pd.to_numeric(g['salaryCount'],errors='coerce').fillna(0),
                        pd.to_numeric(g['salary_count_c'],errors='coerce').fillna(0))
g['n_index']  = pd.to_numeric(g['n_index'],errors='coerce').fillna(1)

# geography
hq = g['hq'].fillna('')
g['hq_city'] = hq.str.split(',').str[0].map(norm_city)
tail = hq.str.split(',').str[-1].str.strip()
g['hq_state'] = np.where(tail.str.fullmatch(r'[A-Z]{2}', na=False), tail,
                         g['office_state'].fillna(g['state_c']).fillna(''))
g['hq_tailtxt'] = tail.fillna('').map(norm_city)
g['hq_country_txt'] = np.where(tail.str.fullmatch(r'[A-Z]{2}', na=False), 'united states',
                               g['hq_tailtxt'])
g['is_us'] = tail.str.fullmatch(r'[A-Z]{2}', na=False) | g['office_country'].fillna('').str.contains('United States')

# keys
g['dom']      = g['website'].map(domain_of)
g['dom_stem'] = g['dom'].map(domain_stem)
g['tick']     = g['stock'].map(norm_ticker)
g['nname']    = g['name'].map(norm_name)
g['cname']    = g['name'].map(core_name)
g['nkey']     = g['nname'].str.replace(' ','',regex=False)
g['ckey']     = g['cname'].str.replace(' ','',regex=False)
JUNK = {'na','no','none','null','private','selfemployed','self employed','n a','unknown','test',
        'confidential','abc','xyz','company','business','anonymous','home','retired','student'}
g = g[(g['nkey'].str.len()>=2) & (~g['nname'].isin(JUNK))]

keep=['cid','name','nname','cname','nkey','ckey','dom','dom_stem','tick','gtype','gsize','grevenue',
      'hq','hq_city','hq_state','hq_country_txt','is_us','gind','gdesc','reviews','salaries',
      'yearFounded','parentId','parentName','competitors','n_index','activeStatus']
g = g[[c for c in keep if c in g.columns]].drop_duplicates('cid')
g.to_parquet(f"{S}/gd_canon.parquet", index=False)
print("\nFINAL canonical employers", len(g), flush=True)
print("  with domain %.1f%%  ticker %.1f%%  hq %.1f%%  desc %.1f%%" % (
    100*(g['dom']!='').mean(), 100*(g['tick']!='').mean(),
    100*g['hq'].notna().mean(), 100*g['gdesc'].notna().mean()))
print("\ncoverage by review stratum:")
for lo,hi in [(0,0),(1,4),(5,24),(25,99),(100,999),(1000,10**9)]:
    s=g[(g['reviews']>=lo)&(g['reviews']<=hi)]
    if len(s): print(f"  {lo:>5}-{hi:<11} n={len(s):>9,}  dom={100*(s['dom']!='').mean():5.1f}%  tick={100*(s['tick']!='').mean():4.1f}%  hq={100*s['hq'].notna().mean():5.1f}%")
print("\ntop employers:"); print(g.nlargest(8,'reviews')[['cid','name','reviews','dom','tick','hq']].to_string(index=False))
