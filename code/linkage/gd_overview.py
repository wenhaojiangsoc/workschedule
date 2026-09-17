import pandas as pd, glob, warnings, sys
warnings.filterwarnings('ignore')
fs = sorted(glob.glob('/Users/wj93/Library/CloudStorage/Box-Box/Finance and Work Flexibility/raw data/company overview/overview_df_*.csv'),
            key=lambda f:int(f.rsplit('_',1)[1].split('.')[0]))
parts=[]
for i,f in enumerate(fs):
    d = pd.read_csv(f, dtype=str)
    parts.append(d)
    if i%10==0: print(i, d.shape, flush=True)
d = pd.concat(parts, ignore_index=True)
print('total', d.shape)
print('unique ids', d['companyid'].nunique())
d = d.drop_duplicates('companyid')
print('after dedupe', d.shape)
print(d.notna().mean().round(3))
d.to_csv(f'{sys.argv[1]}', index=False)
