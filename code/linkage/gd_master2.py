"""Pass 2 over raw Apollo pickles: keep queried index AND canonical employer id."""
import pickle, glob, csv, sys, os, time
out = sys.argv[1]
files = sorted(glob.glob('/Users/wj93/Library/CloudStorage/Dropbox/Glassdoor/data/overview/overview_array_*'),
               key=lambda f: int(f.rsplit('_',1)[1]))
F = ['queried_id','canon_id','shortName','website','stock','headquarters','type','size','revenue',
     'yearFounded','industryName','reviewCount','salaryCount','activeStatus','parentId','parentName',
     'nSub','nDiv','description','mission','office_city','office_state','office_country','competitors']
def canon_ref(rec):
    for k,v in rec.get('ROOT_QUERY',{}).items():
        if k.startswith('employer(') and isinstance(v,dict) and '__ref' in v: return v['__ref']
    for k in rec:
        if k.startswith('Employer:'): return k
    return None
def deref(rec,v):
    return rec.get(v['__ref'],{}) if isinstance(v,dict) and '__ref' in v else (v if isinstance(v,dict) else {})
w = csv.DictWriter(open(out,'w',newline=''), fieldnames=F, extrasaction='ignore'); w.writeheader()
t0=time.time(); n=0
for f in files:
    try: obj = pickle.load(open(f,'rb'))
    except Exception as e:
        print(f'SKIP {os.path.basename(f)}: {e}', file=sys.stderr, flush=True); continue
    for rec in obj:
        if not isinstance(rec,dict): continue
        ref = canon_ref(rec)
        if not ref: continue
        e = rec.get(ref) or {}
        if not isinstance(e,dict): continue
        c  = e.get('counts') or {}
        pi = e.get('primaryIndustry') or {}
        ov = e.get('overview') or {}
        par= e.get('parent') or {}
        pe = deref(rec, par.get('employer')) if isinstance(par,dict) else {}
        oa = e.get('officeAddresses') or []
        a0 = deref(rec, oa[0]) if oa else {}
        comp = e.get('competitors') or []
        w.writerow(dict(
            queried_id=rec.get('companyid'), canon_id=e.get('id'), shortName=e.get('shortName'),
            website=e.get('website'), stock=e.get('stock'), headquarters=e.get('headquarters'),
            type=e.get('type'), size=e.get('size'), revenue=e.get('revenue'),
            yearFounded=e.get('yearFounded'),
            industryName=pi.get('industryName') if isinstance(pi,dict) else None,
            reviewCount=c.get('reviewCount'), salaryCount=c.get('salaryCount'),
            activeStatus=e.get('activeStatus'), parentId=par.get('employerId') if isinstance(par,dict) else None,
            parentName=pe.get('shortName'), nSub=len(e.get('subsidiaries') or []),
            nDiv=len(e.get('divisions') or []),
            description=(ov.get('description') or '')[:1500] if isinstance(ov,dict) else '',
            mission=(ov.get('mission') or '')[:400] if isinstance(ov,dict) else '',
            office_city=a0.get('city'), office_state=a0.get('state'), office_country=a0.get('country'),
            competitors='; '.join([x.get('shortName','') for x in comp if isinstance(x,dict)][:6]),
        )); n+=1
    print(f'{os.path.basename(f)} total={n} {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
