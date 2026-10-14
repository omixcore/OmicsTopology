"""Source matrices to version-2 results for one held-out and one development study.

GSE3431 was not used by the version-1 repository. It is held out from parameter
fitting, not a prospective biological experiment. GSE158444 is a development
case and its two RNA fractions are explicitly not independent studies.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
from omictopo.protocol import analyse
from omictopo.axes import axis_geometry
from validation_v2 import algorithm_hash,environment
from prepare_plant_dataset import read_counts,parse_design
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'GSE3431':{'url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE3nnn/GSE3431/matrix/GSE3431_series_matrix.txt.gz','filename':'GSE3431_series_matrix.txt.gz','role':'held_out_study','reference':'Tu et al. Science 2005; DOI 10.1126/science.1120499'},
 'GSE158444':{'url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE158nnn/GSE158444/suppl/GSE158444_Counts_Time.txt.gz','filename':'GSE158444_Counts_Time.txt.gz','role':'development_case','reference':'Bonnot and Nagel Plant Cell 2021; DOI 10.1093/plcell/koab113'}}


def source_files(cache,allow_download):
    manifest=json.loads((ROOT/'reproduce/source_manifest.json').read_text())
    paths={}
    cache.mkdir(parents=True,exist_ok=True)
    for accession,spec in SOURCES.items():
        p=cache/spec['filename']
        if not p.exists():
            if not allow_download:raise SystemExit(f'Missing {p}; pass --download to fetch the public source')
            tmp=p.with_suffix(p.suffix+'.part')
            with urllib.request.urlopen(spec['url'],timeout=60) as inp, tmp.open('wb') as out:
                while True:
                    chunk=inp.read(1<<20)
                    if not chunk:break
                    out.write(chunk)
            tmp.replace(p)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=manifest['sources'][accession]['sha256']:
            raise SystemExit(f'Source integrity failure for {accession}; do not repin automatically')
        paths[accession]=p
    return paths,manifest


def selected(X,top_var):
    X=np.asarray(X,float)
    valid=np.isfinite(X).all(axis=0)&(X.std(axis=0)>1e-12)
    idx=np.flatnonzero(valid)
    if top_var and len(idx)>top_var:
        rank=np.argsort(-X[:,idx].var(axis=0),kind='stable')[:top_var]
        idx=np.sort(idx[rank])
    return X[:,idx],idx


def load_holdout(p,top_var):
    rows=[];metadata={};samples=[];genes=[];inside=False
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            cols=next(csv.reader([line],delimiter='\t'))
            if not cols:continue
            if cols[0]=='!series_matrix_table_begin':inside=True;continue
            if cols[0]=='!series_matrix_table_end':break
            if cols[0].startswith('!Sample_'):metadata[cols[0]]=cols[1:]
            if not inside:continue
            if cols[0]=='ID_REF':samples=cols[1:];continue
            genes.append(cols[0]);rows.append([float(v) if v not in ['null','NA',''] else np.nan for v in cols[1:]])
    expected=[f'T{i}' for i in range(1,37)]
    if metadata.get('!Sample_title')!=expected or len(samples)!=36:
        raise ValueError('GSE3431 order does not match the registered 36-interval design')
    X,idx=selected(np.array(rows).T,top_var)
    return X,{'samples':samples,'gene_ids':[genes[i] for i in idx],
              'phase_radians':(np.arange(36)%12*2*np.pi/12).tolist(),
              'cycle_block':(np.arange(36)//12).tolist(),
              'transformation':'released processed values, no additional log transform',
              'dependence':'one continuous culture, successive cycles are not independent biological replicates'}


def load_arabidopsis(p,fraction,top_var):
    genes,samples,counts=read_counts(str(p));design=parse_design(samples)
    keep=[i for i,d in enumerate(design) if d and d['fraction']==fraction and d['hour']<24]
    keep.sort(key=lambda i:(design[i]['hour'],design[i]['treatment'],design[i]['replicate']))
    C=counts[:,keep];cpm=C/np.maximum(C.sum(axis=0),1)*1e6
    expressed=(cpm>=1).sum(axis=1)>=int(np.ceil(.2*len(keep)))
    available=np.flatnonzero(expressed)
    X,idx=selected(np.log2(cpm[expressed].T+1),top_var)
    return X,{'samples':[samples[i] for i in keep],
             'gene_ids':[genes[available[j]] for j in idx],
             'groups':[design[i]['treatment'] for i in keep],
             'hours':[design[i]['hour'] for i in keep],
             'transformation':'CPM on all selected-library counts, CPM>=1 in >=20%, log2(CPM+1), variance selection',
             'dependence':'destructive biological samples at each hour; fraction arms share one source study'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='data/source');ap.add_argument('--download',action='store_true');ap.add_argument('--plan',default='benchmark/v2/validation_plan.json');ap.add_argument('--out',default='benchmark/v2/real_validation.json');a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text())
    if plan['algorithm_sha256']!=algorithm_hash():raise SystemExit('Algorithm does not match frozen plan')
    paths,manifest=source_files(Path(a.cache),a.download)
    settings=plan['settings'];top=plan['real_preprocessing']['top_var'];results={}
    X,design=load_holdout(paths['GSE3431'],top)
    # A geometric row check; do not pretend the three cycles are biological subjects.
    r=analyse(X,design['phase_radians'],period=2*np.pi,seed=950001,**settings)
    results['GSE3431']={'role':'held_out_study','design':design,'analysis':r,
                       'matrix_sha256':hashlib.sha256(np.asarray(X,dtype='<f8').tobytes()).hexdigest(),
                       'n_source_samples':36,'n_retained_features':X.shape[1]}
    print('held-out GSE3431',r['status'],r['n_cycles'],flush=True)
    # Predeclared feature sensitivity uses the same settings and no threshold tuning.
    sensitivity=[]
    for feature_count in [500,0]:
        sub,_=load_holdout(paths['GSE3431'],feature_count)
        rec=analyse(sub,design['phase_radians'],period=2*np.pi,seed=950001,**settings)
        sensitivity.append({'top_var':feature_count,'n_features':sub.shape[1],'analysis':rec})
    results['GSE3431']['feature_sensitivity']=sensitivity
    for fraction in ['TOT','TR']:
        X,design=load_arabidopsis(paths['GSE158444'],fraction,top)
        groups=np.asarray(design['groups']);hours=np.asarray(design['hours'])
        arms={}
        for treatment in ['control','heat']:
            m=groups==treatment
            rec=analyse(X[m],hours[m],period=24,seed=951001,**settings,strata=hours[m])
            arms[treatment]=rec
            print(fraction,treatment,rec['status'],rec['n_cycles'],flush=True)
        ax=axis_geometry(X,hours,groups,period1=24,seed=951001,n_boot=500,
                         permutation='samples',permutation_strata=hours,n_perm=1999)
        results['GSE158444_'+fraction]={'role':'development_case','fraction':fraction,
                'study_unit':'GSE158444','design':design,'arms':arms,'axis':ax,
                'matrix_sha256':hashlib.sha256(np.asarray(X,dtype='<f8').tobytes()).hexdigest()}
    out={'plan':plan,'environment':environment(),'source_manifest':manifest,'results':results,
         'limits':['one held-out study does not establish general biological accuracy',
                  'GSE3431 is one continuous culture without independent biological series',
                  'zero H1 detection does not rule out rhythm or a weak second loop',
                  'no claim of a novel biological discovery or validated two-cycle real manifold']}
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
