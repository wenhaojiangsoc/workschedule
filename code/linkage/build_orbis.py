"""Orbis firm master + name-variant table for the 10,210 Big-3 sample firms."""
import sys, re, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
S = "/private/tmp/claude-501/-Users-wj93-Library-CloudStorage-Box-Box-Finance-and-Work-Flexibility/7bbc71e4-1316-4d6d-9e0c-6a82c046313d/scratchpad"
sys.path.insert(0, S)
from norm import norm_name, core_name, domain_of, domain_stem, norm_ticker, norm_city
BOX = "/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility"

def rd(name):
    try:
        d = pd.read_csv(f"{S}/orbis/{name}.tsv", sep='\t', dtype=str,
                        encoding='utf-8-sig', on_bad_lines='skip', quoting=3)
        d.columns = [c.strip().lstrip('﻿') for c in d.columns]
        d = d.rename(columns={d.columns[0]: 'bvd'})
        return d[d['bvd'].notna()]
    except Exception as e:
        print(f"  !! {name}: {e}"); return pd.DataFrame(columns=['bvd'])

# ---------- base panel ----------
big3 = pd.read_csv(f"{BOX}/processed data/Orbis/big3_shares_allyears_merged.csv")
base = (big3.groupby('company_id')
        .agg(name_legal=('Name','first'), country=('Country ISO code','first'),
             etype=('Entity type','first'), yr_min=('year','min'), yr_max=('year','max'),
             n_years=('year','size'), bl=('blackrock_pct','mean'),
             vg=('vanguard_pct','mean'), ss=('statestreet_pct','mean'))
        .reset_index().rename(columns={'company_id':'bvd'}))
print("base", base.shape, flush=True)

# ---------- contact info ----------
ci = rd('contact_info')
if len(ci):
    ci = ci.drop_duplicates('bvd')
    g = lambda c: ci[c] if c in ci.columns else ''
    base = base.merge(pd.DataFrame({
        'bvd': ci['bvd'], 'name_internat': g('NAME_INTERNAT'), 'name_native': g('NAME_NATIVE'),
        'website': g('Website address'), 'city': g('City'), 'postcode': g('Postcode'),
        'ci_country': g('Country ISO code'), 'us_state': g('State or province (in US or Canada)'),
        'metro': g('Metropolitan area (in US)'), 'phone': g('Telephone number'),
    }), on='bvd', how='left')
print("after contact", base.shape, flush=True)

# ---------- identifiers: ticker / ISIN / LEI ----------
idf = rd('identifiers')
if len(idf):
    tk = (idf[idf.get('Ticker symbol','').astype(str).str.len()>0][['bvd','Ticker symbol']]
          .drop_duplicates('bvd').rename(columns={'Ticker symbol':'ticker'}))
    isin = (idf[idf.get('ISIN number','').astype(str).str.len()>0][['bvd','ISIN number']]
            .drop_duplicates('bvd').rename(columns={'ISIN number':'isin'}))
    lei = (idf[idf.get('LEI (Legal Entity Identifier)','').astype(str).str.len()>0]
           [['bvd','LEI (Legal Entity Identifier)']].drop_duplicates('bvd')
           .rename(columns={'LEI (Legal Entity Identifier)':'lei'}))
    for t in (tk, isin, lei): base = base.merge(t, on='bvd', how='left')
print("after identifiers", base.shape, flush=True)

# ---------- legal info ----------
li = rd('legal_info')
if len(li):
    l1 = li.drop_duplicates('bvd')
    g = lambda c: l1[c] if c in l1.columns else ''
    base = base.merge(pd.DataFrame({
        'bvd': l1['bvd'], 'listed': g('Listed/Delisted/Unlisted'), 'exchange': g('Main exchange'),
        'status': g('Status'), 'inc_state': g('State of incorporation (in US)'),
        'legal_form': g('Standardised legal form'), 'ipo': g('IPO date')}), on='bvd', how='left')
print("after legal", base.shape, flush=True)

# ---------- overviews: description + brands ----------
ov = rd('overviews')
if len(ov):
    o1 = ov.drop_duplicates('bvd')
    g = lambda c: o1[c] if c in o1.columns else ''
    base = base.merge(pd.DataFrame({
        'bvd': o1['bvd'], 'overview': g('Full overview'), 'brands_raw': g('Main brand names'),
        'products': g('Main products and services'), 'size_est': g('Size estimate'),
        'biz_line': g('Primary business line')}), on='bvd', how='left')
print("after overviews", base.shape, flush=True)

# ---------- LinkedIn slug ----------
ai = rd('addl_info')
if len(ai) and 'Link to social networks accounts' in ai.columns:
    ln = ai[ai['Link to social networks accounts'].astype(str).str.contains('linkedin', case=False, na=False)]
    ln = ln.assign(li_slug=ln['Link to social networks accounts'].str.extract(
        r'linkedin\.com/(?:company|school)/([^/?&#\s]+)', flags=re.I)[0].str.lower())
    ln = ln[ln['li_slug'].notna()].drop_duplicates('bvd')[['bvd','li_slug']]
    base = base.merge(ln, on='bvd', how='left')
print("after linkedin", base.shape, flush=True)

for c in ['name_internat','name_native','website','city','us_state','ticker','isin','lei','listed',
          'exchange','overview','brands_raw','products','li_slug','ci_country','metro','inc_state']:
    if c not in base.columns: base[c] = np.nan

# ---------- derived keys ----------
base['dom']      = base['website'].map(domain_of)
base['dom_stem'] = base['dom'].map(domain_stem)
base['tick']     = base['ticker'].map(norm_ticker)
base['ncity']    = base['city'].map(norm_city)
base['country']  = base['country'].fillna(base['ci_country'])
base.to_parquet(f"{S}/orbis_master.parquet", index=False)

# ---------- name variants (long) ----------
rows = []
def add(bvd, nm, src):
    if not isinstance(nm, str): return
    nm = nm.strip()
    if len(nm) < 2 or nm.lower() in {'n.a.','na','nan','-'}: return
    rows.append((bvd, nm, src))

for r in base.itertuples(index=False):
    add(r.bvd, r.name_legal, 'legal')
    add(r.bvd, getattr(r,'name_internat',None), 'internat')
    add(r.bvd, getattr(r,'name_native',None), 'native')
    b = getattr(r,'brands_raw',None)
    if isinstance(b, str):
        for piece in re.split(r'[;,]', b)[:12]: add(r.bvd, piece, 'brand')

for src_file, cols, tag in [('addl_info',
        ['Previous company name (original language)','Previous company name (international language)',
         'Also known as name (original language)','Also known as name (international language)'], None),
      ('legal_info', ['Previous company name','Also known as name'], None)]:
    d = rd(src_file)
    for c in cols:
        c2 = c if c in d.columns else next((x for x in d.columns if x.strip()==c), None)
        if c2 is None: continue
        tg = 'prev' if 'Previous' in c else 'aka'
        for bvd, nm in d[['bvd', c2]].dropna().itertuples(index=False): add(bvd, nm, tg)

nv = pd.DataFrame(rows, columns=['bvd','name','src']).drop_duplicates()
nv = nv[nv['bvd'].isin(set(base['bvd']))]
nv['nname'] = nv['name'].map(norm_name)
nv['cname'] = nv['name'].map(core_name)
nv = nv[nv['nname'].str.len()>0]
nv['nkey'] = nv['nname'].str.replace(' ','',regex=False)
nv['ckey'] = nv['cname'].str.replace(' ','',regex=False)
nv = nv.drop_duplicates(['bvd','nkey','src'])
nv.to_parquet(f"{S}/orbis_names.parquet", index=False)

print("\n=== coverage over", len(base), "Orbis firms")
for c,lab in [('dom','website/domain'),('tick','ticker'),('lei','LEI'),('li_slug','LinkedIn'),
              ('name_internat','intl name'),('brands_raw','brand names'),('overview','overview text'),
              ('ncity','city'),('us_state','US state'),('listed','listed flag')]:
    v = base[c].fillna('') if c in base else pd.Series([''])
    print(f"  {lab:16s} {(v.astype(str).str.len()>0).mean():6.1%}")
print("\nname variants", nv.shape, "| per firm", round(len(nv)/base['bvd'].nunique(),2))
print(nv['src'].value_counts().to_string())
