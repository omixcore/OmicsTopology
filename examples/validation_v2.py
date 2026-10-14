"""Freeze a specification, then score disjoint simulated validation seeds.

This is an internal frozen configuration, not an externally registered study.
Any post-validation algorithm change invalidates the code digest in the plan.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, platform, importlib.metadata
from pathlib import Path
import numpy as np
from omictopo.protocol import analyse
from omictopo.axes import axis_geometry
from omictopo.stats import wilson_interval
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=['gaussian','ramp','clusters','branch','arc','drift','one_cycle','two_cycles','unequal_two','three_cycles']


def algorithm_hash():
    h=hashlib.sha256()
    for name in ['inference.py','protocol.py','axes.py','stats.py','geometry.py']:
        h.update(name.encode());h.update((ROOT/'src/omictopo'/name).read_bytes().replace(b'\r\n',b'\n'))
    return h.hexdigest()


def make_cloud(family,seed):
    rng=np.random.default_rng(seed);n=48;p=96
    if family in ('two_cycles','unequal_two'):
        n=64;t=np.repeat(np.arange(8),8)*2*np.pi/8;q=np.tile(np.arange(8),8)*2*np.pi/8
        weight=1 if family=='two_cycles' else .35
        S=np.c_[np.cos(t),np.sin(t),weight*np.cos(q),weight*np.sin(q)];expected=2
    elif family=='three_cycles':
        n=64;t=np.indices((4,4,4)).reshape(3,-1)*2*np.pi/4
        S=np.concatenate([np.c_[np.cos(a),np.sin(a)] for a in t],axis=1);expected=3
    elif family=='gaussian':
        return rng.normal(size=(n,p)),0
    elif family=='one_cycle':
        t=np.arange(n)*2*np.pi/n;S=np.c_[np.cos(t),np.sin(t)];expected=1
    elif family=='arc':
        t=np.linspace(0,1.2*np.pi,n);S=np.c_[np.cos(t),np.sin(t)];expected=0
    elif family=='drift':
        t=np.linspace(0,4*np.pi,n);S=np.c_[np.cos(t),np.sin(t),t/np.pi];expected=0
    elif family=='ramp':
        t=np.linspace(-1,1,n);S=np.c_[t,t**2];expected=0
    elif family=='clusters':
        S=np.repeat(np.array([[0,0],[2,0],[0,2]]),n//3,axis=0);expected=0
    elif family=='branch':
        arm=np.repeat(np.arange(3),n//3);dist=np.tile(np.linspace(0,1,n//3),3)
        S=np.c_[dist*np.cos(arm*2*np.pi/3),dist*np.sin(arm*2*np.pi/3)];expected=0
    else:raise ValueError(family)
    return S@rng.normal(size=(S.shape[1],p))+.12*rng.normal(size=(n,p)),expected


def axis_cloud(kind,seed):
    rng=np.random.default_rng(seed);t=np.tile(np.arange(12)*2*np.pi/12,4);b=np.repeat(np.arange(4),12)
    p=96;ph=rng.uniform(-np.pi,np.pi,p);off=np.array([0,.25,.5,.75])[b,None]
    theta=t[:,None]+ph
    if kind=='phase':theta+=off
    if kind=='shape':theta+=off*np.where(np.arange(p)%2,2,-2)
    X=2*np.cos(theta)+.1*rng.normal(size=(48,p))
    if kind=='independent':
        # Independent loadings, rather than the same gene phase reused twice.
        X+=2*(np.c_[np.cos(b*np.pi/2),np.sin(b*np.pi/2)]@rng.normal(size=(2,p)))
    if kind=='shift':X+=b[:,None]*rng.normal(size=p)*8
    return X,t,b


def environment():
    return {'python':sys.version,'platform':platform.platform(),
            'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','ripser','omictopo']}}


def freeze(path,replicates,quick):
    settings={'metric':'correlation','n_null':39 if quick else 99,'alpha':.05,
              'n_resamples':4 if quick else 8,'resample_n_null':39,
              'resample_fraction':.9,'stability_threshold':.75,'backend':'ripser'}
    plan={'schema_version':2,'registration':'internal frozen plan; not external preregistration',
          'algorithm_sha256':algorithm_hash(),'development_seeds':[701,79,101,113],
          'validation_seed_start':1010001,'axis_validation_seed_start':1030001,
          'replicates_per_family':replicates,'families':FAMILIES,
          'axis_families':['inert','phase','shape','independent','shift'],
          'held_out_real_accession':'GSE3431','development_real_accession':'GSE158444',
          'real_preprocessing':{'top_var':1000,'holdout_transform':'released_values',
                                 'arabidopsis_transform':'log2 CPM + 1'},
          'settings':settings,'quick_smoke':quick,
          'reference':'latent generator H1 ranks; finite samples may not resolve them'}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def score(plan):
    if plan['algorithm_sha256']!=algorithm_hash():
        raise SystemExit('Algorithm changed after freezing; freeze a NEW plan and use NEW held-out seeds.')
    rows=[]
    for j,fam in enumerate(plan['families']):
        for i in range(plan['replicates_per_family']):
            seed=plan['validation_seed_start']+j*1000+i
            X,truth=make_cloud(fam,seed)
            rec=analyse(X,seed=seed,**plan['settings'])
            rows.append({'family':fam,'seed':seed,'reference_count':truth,
                         'matrix_sha256':hashlib.sha256(np.asarray(X,dtype='<f8').tobytes()).hexdigest(),
                         'result':rec})
        print(f"scored {fam}",flush=True)
    summaries={}
    for fam in plan['families']:
        rr=[r for r in rows if r['family']==fam];answered=[r for r in rr if r['result']['n_cycles'] is not None]
        correct=sum(r['result']['n_cycles']==r['reference_count'] for r in answered)
        summaries[fam]={'n':len(rr),'answered':len(answered),'correct_among_answered':correct,
                        'coverage':len(answered)/len(rr),
                        'conditional_error':1-correct/len(answered) if answered else None,
                        'conditional_error_ci95':list(wilson_interval(len(answered)-correct,len(answered))) if answered else None,
                        'candidate_counts':[r['result']['candidate_count'] for r in rr],
                        'accepted_counts':[r['result']['n_cycles'] for r in rr]}
    negative=[r for r in rows if r['reference_count']==0];positive=[r for r in rows if r['reference_count']>0]
    fp=sum(r['result']['n_cycles'] is not None and r['result']['n_cycles']>0 for r in negative)
    hit=sum(r['result']['n_cycles'] is not None and r['result']['n_cycles']>0 for r in positive)
    axis=[]
    for j,kind in enumerate(plan['axis_families']):
        for i in range(plan['replicates_per_family']):
            seed=plan['axis_validation_seed_start']+j*1000+i
            X,t,b=axis_cloud(kind,seed)
            axis.append({'family':kind,'seed':seed,'result':axis_geometry(X,t,b,n_boot=100,seed=seed)})
    return {'plan':plan,'environment':environment(),'summaries':summaries,'records':rows,
            'axis_records':axis,'aggregate':{'negative_n':len(negative),'positive_n':len(positive),
                'false_positive_k':fp,'false_positive_ci95':list(wilson_interval(fp,len(negative))),
                'positive_detection_k':hit,'positive_detection_ci95':list(wilson_interval(hit,len(positive)))},
            'limits':['synthetic validation only','risk is evaluated against generator rank, not biological ground truth',
                      'error intervals do not validate nominal count coverage','axis labels do not prove independence']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',default='benchmark/v2/validation_plan.json');ap.add_argument('--out',default='benchmark/v2/validation.json');ap.add_argument('--freeze',action='store_true');ap.add_argument('--quick',action='store_true');ap.add_argument('--replicates',type=int,default=6);a=ap.parse_args()
    planpath=Path(a.plan)
    if a.freeze:
        if planpath.exists():raise SystemExit('Refusing to overwrite a frozen plan; select a new path.')
        plan=freeze(planpath,a.replicates,a.quick)
    else:plan=json.loads(planpath.read_text())
    result=score(plan);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
