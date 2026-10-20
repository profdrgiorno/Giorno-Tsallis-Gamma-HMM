from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_manuscript():
    assert (ROOT/'paper/Giorno_Tsallis_Paper_1_2_Latest_2026-10-15.pdf').stat().st_size>100000
def test_main_keys():
    d=pd.read_csv(ROOT/'data/raw/all_replications_combined.csv')
    assert len(d)==2700 and not d.duplicated(['scenario','n','rep']).any()
    assert (d.groupby(['scenario','n']).size()==300).all()
def test_location_keys():
    d=pd.read_csv(ROOT/'data/raw/hmm7_location_replications.csv')
    assert len(d)==2700 and not d.duplicated(['scenario','n','rep']).any()
def test_inference_keys():
    d=pd.read_csv(ROOT/'data/raw/hmm9_wald_inference_replications.csv')
    assert len(d)==1000 and not d.duplicated(['scenario','n','rep']).any()
