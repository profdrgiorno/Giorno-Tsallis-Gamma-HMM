#!/usr/bin/env python
# Giorno--Tsallis HMM-7 location-adjusted Student-t benchmark.
# Uses the EXACT simulation seeds and HMM-6/HMM-9 source code in
# giorno_tsallis_complete_grok.py. Provenance: Giorno-Tsallis MOON archive.
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.signal import fftconvolve
import giorno_tsallis_complete_grok as gt

SCENARIOS=['correct','gaussian_contamination','negative_gamma_jumps']

def emit7(y,p):
    # HMM7 = common unknown location plus six original Student-t HMM parameters
    return np.column_stack((gt.t_logpdf(y-p[6],p[2],p[4]), gt.t_logpdf(y-p[6],p[3],p[5])))

def objective7(x,y):
    a=gt.decode(x[:6]); a=np.append(a,float(x[6]))
    le=emit7(y,a)
    return -gt.forward_filter(le,float(a[0]),float(a[1]))[0]

def fit7(y,quick=False):
    sd=float(np.std(y)); avg=float(np.mean(y))
    # Both centered and uncentered starting values, no hidden-state labels.
    # Use exactly HMM6 optimizer from Moon for fair initialization.
    a6,_,_ = gt.fit6(y)
    if quick:
        starters=[np.append(gt.encode(a6),avg)]
    else:
        b6,_,_ = gt.fit6(y-avg)
        starters=[np.append(gt.encode(a6),0.),np.append(gt.encode(a6),avg),np.append(gt.encode(b6),avg)]
    # Predeclared, symmetric bound large relative to true mean -0.18.
    loc_cap=max(1.0, min(4., 2.0*sd))
    bounds=gt.bounds6(y)+[(-loc_cap,loc_cap)]
    fits=[]
    for s in starters:
        s[6]=np.clip(s[6],-loc_cap,loc_cap)
        r=minimize(objective7,s,args=(y,),method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':160,'ftol':1e-9,'gtol':1e-5})
        fits.append(r)
    b=min(fits,key=lambda x: x.fun)
    a6=gt.reorder(gt.decode(b.x[:6]))
    a7=np.append(a6,float(b.x[6]))
    ll=-objective7(np.append(gt.encode(a6),a7[6]),y)
    return a7,ll,bool(b.success),int(sum(z.success for z in fits))

def oracle_emissions(y,scenario):
    a=np.array([.985,.960,*gt.NUTRUE,*gt.STRUE],float)
    if scenario=='correct':
        return gt.le6(y,a)
    if scenario=='negative_gamma_jumps':
        a=np.append(a,[.03,2.,3.]);return gt.le9(y,a)
    if scenario=='gaussian_contamination':
        # FFT convolution with a symmetric Gaussian kernel, checked against
        # scipy.integrate.quad at representative grid points.
        normal_kernel=np.exp(-0.5*(gt.GRID/5.0)**2)/(5*np.sqrt(2*np.pi))*gt.GRID_STEP
        normal_kernel/=normal_kernel.sum()
        result=[]
        for j in range(2):
            f=np.exp(gt.t_logpdf(gt.GRID,a[2+j],a[4+j]))
            conv=fftconvolve(f,normal_kernel,mode='same')
            conv=np.maximum(np.interp(y,gt.GRID,conv),1e-300)
            fdir=gt.t_logpdf(y,a[2+j],a[4+j])
            result.append(np.logaddexp(np.log(.97)+fdir,np.log(.03)+np.log(conv)))
        return np.column_stack(result)
    raise ValueError(scenario)

def compute(row,quick=False):
    sc=row['scenario'];n=int(row['n']);rep=int(row['rep'])
    y,z,jumps=gt.simulate(sc,n,rep)
    ytrain=y[:n]
    start=time.monotonic()
    p7,ll,ok,numok=fit7(ytrain,quick=quick)
    lep=emit7(y,p7)
    _,alpha,scores=gt.forward_filter(lep,float(p7[0]),float(p7[1]))
    q7=(p7[2:4]+3)/(p7[2:4]+1)
    qtrack=(1-alpha[n:])*q7[0]+alpha[n:]*q7[1]
    qtruth=gt.QTRUE[z[n:]]
    letrue=oracle_emissions(y,sc)
    _,_,oracle_sc=gt.forward_filter(letrue,float(gt.PTRUE[0,0]),float(gt.PTRUE[1,1]))
    out={'scenario':sc,'n':n,'rep':rep,'q0_loc':q7[0], 'q1_loc':q7[1],
         'q0_sqerr_loc':(q7[0]-gt.QTRUE[0])**2,'q1_sqerr_loc':(q7[1]-gt.QTRUE[1])**2,
         'mu_loc':p7[6],'mu_true':(-.18 if sc=='negative_gamma_jumps' else 0.),
         'logscore_loc':float(scores[n:].mean()),
         'filter_rmse_loc':float(np.sqrt(np.mean((qtrack-qtruth)**2))),
         'oracle_logscore':float(oracle_sc[n:].mean()),
         'loc_success':bool(ok),'loc_successful_starts':numok,
         'seconds':round(time.monotonic()-start,3),
         'q1_direct':float(row['q1_direct']),'q1_jump':float(row['q1_jump']),
         'q1_sqerr_direct':float(row['q1_sqerr_direct']),
         'q1_sqerr_jump':float(row['q1_sqerr_jump']),
         'logscore_direct':float(row['logscore_direct']),
         'logscore_jump':float(row['logscore_jump']),
         'filter_rmse_direct':float(row['filter_rmse_direct']),
         'filter_rmse_jump':float(row['filter_rmse_jump']),
         'success_direct':bool(row['success_direct']),
         'success_jump':bool(row['success_jump'])}
    return out

def summarize(df):
    rows=[]
    for (sc,n),d in df.groupby(['scenario','n']):
        nrep=len(d)
        # Relative difference, positive favors HMM9 vs HMM7-loc
        diff=d.logscore_jump-d.logscore_loc
        se=diff.std(ddof=1)/np.sqrt(nrep) if nrep>1 else np.nan
        rows.append({'scenario':sc,'n':n,'reps':nrep,
                     'q1_rmse_hmm6':np.sqrt(d.q1_sqerr_direct.mean()),
                     'q1_rmse_hmm7_loc':np.sqrt(d.q1_sqerr_loc.mean()),
                     'q1_rmse_hmm9':np.sqrt(d.q1_sqerr_jump.mean()),
                     'filtered_q_rmse_hmm6':np.sqrt(np.mean(d.filter_rmse_direct**2)),
                     'filtered_q_rmse_hmm7_loc':np.sqrt(np.mean(d.filter_rmse_loc**2)),
                     'filtered_q_rmse_hmm9':np.sqrt(np.mean(d.filter_rmse_jump**2)),
                     'logscore_hmm6':d.logscore_direct.mean(),
                     'logscore_hmm7_loc':d.logscore_loc.mean(),
                     'logscore_hmm9':d.logscore_jump.mean(),
                     'logscore_oracle':d.oracle_logscore.mean(),
                     'mean_gain_hmm9_vs_hmm7loc':diff.mean(),
                     'mc95_low':diff.mean()-1.96*se,
                     'mc95_high':diff.mean()+1.96*se,
                     'wins_hmm9_vs_hmm7loc':int((diff>0).sum()),
                     'mean_mu_loc':d.mu_loc.mean(),
                     'loc_optimizer_non_success':int((~d.loc_success).sum())})
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--moon-csv',type=Path,default=Path('all_replications_combined(2).csv'))
    parser.add_argument('--out-dir',type=Path,default=Path('gt_focused_benchmark_results'))
    parser.add_argument('--scenario',choices=['all']+SCENARIOS,default='all')
    parser.add_argument('--n',choices=['all','1000','2000','3000'],default='all')
    parser.add_argument('--reps',type=int,default=300)
    parser.add_argument('--start-rep',type=int,default=0)
    parser.add_argument('--quick-pilot',action='store_true',help='One initializer; for testing only')
    args=parser.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True)
    archive=pd.read_csv(args.moon_csv)
    assert not archive.duplicated(['scenario','n','rep']).any()
    scenarios=SCENARIOS if args.scenario=='all' else [args.scenario]
    ns=[1000,2000,3000] if args.n=='all' else [int(args.n)]
    if args.reps<1:raise ValueError('--reps must be >= 1')
    for sc in scenarios:
      for n in ns:
        fname=args.out_dir/f'{sc}_n{n}_location_benchmark.csv'
        if fname.exists():rows=pd.read_csv(fname).to_dict('records')
        else:rows=[]
        done={int(r['rep']) for r in rows}
        pool=archive[(archive.scenario==sc)&(archive.n==n)&(archive.rep>=args.start_rep)&(archive.rep<args.start_rep+args.reps)].sort_values('rep')
        if len(pool)!=args.reps:raise RuntimeError(f'Insufficient MOON records for {sc} n={n}: {len(pool)} vs {args.reps}')
        for _,r in pool.iterrows():
            if int(r.rep) in done:continue
            t=time.monotonic();o=compute(r,quick=args.quick_pilot);rows.append(o)
            temp=fname.with_suffix('.tmp');pd.DataFrame(rows).to_csv(temp,index=False);os.replace(temp,fname)
            print(f'{sc} n={n} rep={int(r.rep)} HMM7 logscore={o["logscore_loc"]:.6f} '
                  f'HMM9-HMM7={o["logscore_jump"]-o["logscore_loc"]:+.6f} '
                  f'elapsed={time.monotonic()-t:.1f}s',flush=True)
    ff=sorted(args.out_dir.glob('*_location_benchmark.csv'))
    df=pd.concat([pd.read_csv(f) for f in ff],ignore_index=True)
    df=df.drop_duplicates(['scenario','n','rep'])
    df.to_csv(args.out_dir/'all_location_benchmark_replications.csv',index=False)
    summary=summarize(df);summary.to_csv(args.out_dir/'location_benchmark_summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':main()
