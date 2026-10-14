"""Check source provenance and recompute v2 detection summaries from raw records.

--smoke reruns scientific regression shapes. This does not regenerate the
entire 60-cloud study or download and rerun source expression matrices.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'examples'))
from validation_v2 import algorithm_hash,make_cloud
from omictopo.inference import signal_spectrum
from omictopo.protocol import analyse


def check_record(r):
    expected=signal_spectrum(r['bar_lengths'],r['null_maxima'],r['settings']['alpha'])
    if expected['candidate_count']!=r['candidate_count']:raise AssertionError('raw spectral count mismatch')
    if not np.allclose(expected['bar_pvalues_max_null'],r['bar_pvalues_max_null']):raise AssertionError('Monte Carlo tails mismatch')
    counts=r['stability']['resampled_counts'];k=r['candidate_count']
    agreement=sum(c==k for c in counts)/len(counts) if counts else None
    stable=agreement is not None and agreement>=r['settings']['stability_threshold']
    n=k if stable else None;status=('counted' if k else 'no_cycle') if stable else 'unresolved'
    if r['stability']['agreement']!=agreement or r['n_cycles']!=n or r['status']!=status:raise AssertionError('stability/status mismatch')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    plan=json.loads((ROOT/'benchmark/v2/validation_plan.json').read_text())
    if plan['algorithm_sha256']!=algorithm_hash():raise AssertionError('current algorithm differs from frozen validation')
    data=json.loads((ROOT/'benchmark/v2/validation.json').read_text())
    if data['plan']!=plan:raise AssertionError('validation plan mismatch')
    allseeds=[r['seed'] for r in data['records']]+[r['seed'] for r in data['axis_records']]
    if len(set(allseeds))!=len(allseeds) or set(allseeds)&set(plan['development_seeds']):raise AssertionError('seed leakage')
    for r in data['records']:check_record(r['result'])
    negatives=[r for r in data['records'] if r['reference_count']==0]
    fp=sum(r['result']['n_cycles'] is not None and r['result']['n_cycles']>0 for r in negatives)
    if fp!=data['aggregate']['false_positive_k'] or len(negatives)!=data['aggregate']['negative_n']:raise AssertionError('aggregate mismatch')
    for fam,s in data['summaries'].items():
        rows=[r for r in data['records'] if r['family']==fam];answered=[r for r in rows if r['result']['n_cycles'] is not None]
        good=sum(r['reference_count']==r['result']['n_cycles'] for r in answered)
        if len(rows)!=s['n'] or len(answered)!=s['answered'] or good!=s['correct_among_answered']:raise AssertionError('risk/coverage mismatch')
    real=json.loads((ROOT/'benchmark/v2/real_validation.json').read_text())
    if real['plan']!=plan:raise AssertionError('real plan mismatch')
    for key,r in real['results'].items():
        if key=='GSE3431':
            check_record(r['analysis'])
            for sub in r['feature_sensitivity']:check_record(sub['analysis'])
        else:
            for arm in r['arms'].values():check_record(arm)
            if r['axis']['cyclic_evidence']['independent_second_cycle']!='not_established':raise AssertionError('unjustified cycle claim')
    if real['source_manifest']!=json.loads((ROOT/'reproduce/source_manifest.json').read_text()):raise AssertionError('source manifest mismatch')
    if a.smoke:
        for family,target in [('one_cycle',1),('two_cycles',2),('gaussian',0)]:
            X,_=make_cloud(family,1100001)
            r=analyse(X,n_null=39,n_resamples=4,resample_n_null=39,seed=1100001)
            if r['n_cycles']!=target:raise AssertionError(f'smoke {family}: {r["status"]}, {r["n_cycles"]}')
    print('v2 integrity, seed separation, spectral tails, risk/coverage and real-record checks passed')
if __name__=='__main__':main()
