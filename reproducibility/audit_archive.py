from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1]
files={'main':R/'data/raw/all_replications_combined.csv','location':R/'data/raw/hmm7_location_replications.csv','inference':R/'data/raw/hmm9_wald_inference_replications.csv'}
for name,p in files.items():
    d=pd.read_csv(p)
    keys=['scenario','n','rep']
    assert set(keys).issubset(d.columns),(name,'missing keys')
    assert not d.duplicated(keys).any(),(name,'duplicate keys')
    assert len(d)==(1000 if name=='inference' else 2700),(name,len(d))
    if name!='inference':
        assert len(d.groupby(['scenario','n']))==9
        assert (d.groupby(['scenario','n']).size()==300).all()
    print(name,len(d),'rows,',len(d.columns),'columns, unique keys')
a=pd.read_csv(files['main']);b=pd.read_csv(files['location'])
assert len(a[['scenario','n','rep']].merge(b[['scenario','n','rep']],on=['scenario','n','rep'],validate='one_to_one'))==2700
print('Cross-archive matching: 2700 paired keys. Audit PASS.')
