"""Run local checks or regenerate frozen validation and real-source analyses."""
from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def run(*args):
    env=os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1',
                'PYTHONPATH':str(ROOT/'src')})
    subprocess.run([sys.executable,*args],cwd=ROOT,env=env,check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--synthetic',action='store_true');ap.add_argument('--real',action='store_true');ap.add_argument('--skip-tests',action='store_true');a=ap.parse_args()
    if a.synthetic:run('examples/validation_v2.py','--out','generated/v2_validation.json')
    if a.real:run('examples/real_validation_v2.py','--download','--out','generated/v2_real_validation.json')
    if not a.skip_tests:run('-m','pytest','-q')
    run('tools/gen_api_docs.py','--check')
    run('reproduce/verify.py')
    run('reproduce/verify_v2.py','--smoke')
    print('Completed. Generated fresh analyses are in generated/; committed evidence was not overwritten.')
if __name__=='__main__':main()
