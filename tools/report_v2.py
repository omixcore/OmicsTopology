"""Generate the version-2 evidence page and figures from scored records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'benchmark/v2/validation.json').read_text())
REAL=json.loads((ROOT/'benchmark/v2/real_validation.json').read_text())
SVG=ROOT/'docs/figures';PNG=ROOT/'generated/v2_figures';SVG.mkdir(exist_ok=True);PNG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
blue='#2B6F9B';grey='#A2B0BA';orange='#C38A32';red='#B44A48'
def save(fig,i):
 fig.savefig(SVG/f'v2_figure{i}.svg',bbox_inches='tight');fig.savefig(PNG/f'figure{i}.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.4));ax.axis('off')
for x,y,text in [(.16,.8,'Feature-space distances\nPinned H1 backend'),(.5,.8,'Reconstructed Gaussian features\nIdentical preprocessing'),(.84,.8,'Observed H1 lengths\nvs surrogate maxima'),(.3,.42,'Refit null in each subsample\nKeep independent units intact'),(.72,.42,'Stable detected count or unresolved\nNo L1/L2 gate')]:
 ax.text(x,y,text,transform=ax.transAxes,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round,pad=.55',fc='#EDF4F8',ec=blue))
for start,end in [((.32,.8),(.35,.8)),((.67,.8),(.68,.8)),((.49,.42),(.51,.42))]:ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=grey))
ax.text(.5,.1,'Design association and second-axis diagnostics are reported separately.\nDetected H1 features do not enumerate biological oscillators.',ha='center',transform=ax.transAxes,color=blue);save(fig,1)
names=DATA['plan']['families'];fig,axs=plt.subplots(1,2,figsize=(7,3.7));targets=[next(r['reference_count'] for r in DATA['records'] if r['family']==f) for f in names]
for j,f in enumerate(names):
 r=[r for r in DATA['records'] if r['family']==f]
 for i,rec in enumerate(r):
  y=rec['result']['n_cycles'];axs[0].scatter(j+(i-2.5)*.09,-.25 if y is None else y,color=grey if y is None else blue,s=15)
 axs[0].scatter(j,targets[j],marker='_',s=180,color=orange)
axs[0].set_xticks(range(len(names)),names,rotation=65,ha='right',fontsize=6.5);axs[0].set(ylabel='Stable detected count',ylim=(-.5,3.4));axs[0].text(.01,.98,'Orange: generator reference\nGrey below zero: unresolved',transform=axs[0].transAxes,va='top',fontsize=7)
thresholds=[.25,.5,.75,1.];coverage=[];error=[]
for cut in thresholds:
 answered=[r for r in DATA['records'] if r['result']['stability']['agreement'] is not None and r['result']['stability']['agreement']>=cut]
 coverage.append(len(answered)/len(DATA['records']));error.append(sum(r['result']['candidate_count']!=r['reference_count'] for r in answered)/len(answered))
axs[1].plot(thresholds,coverage,'o-',label='Coverage',color=blue);axs[1].plot(thresholds,error,'o-',label='Error among answers',color=red);axs[1].axvline(.75,ls='--',color=grey);axs[1].set(xlabel='Required resampling agreement',ylabel='Proportion',ylim=(0,1.08));axs[1].legend(fontsize=7);fig.tight_layout();save(fig,2)
fig,axs=plt.subplots(1,2,figsize=(7,3.3));families=DATA['plan']['axis_families'];v=['no_resolved_axis_effect','phase_modulation','shape_change','strong_axis_effect'];M=np.zeros((5,4))
for i,f in enumerate(families):
 for r in DATA['axis_records']:
  if r['family']==f:M[i,v.index(r['result']['verdict'])]+=1
axs[0].imshow(M,cmap='Blues',vmin=0,vmax=6);axs[0].set_yticks(range(5),families);axs[0].set_xticks(range(4),['Small effect','Common phase','Gene-specific','Strong effect'],rotation=55,ha='right',fontsize=7)
for i in range(5):
 for j in range(4):axs[0].text(j,i,str(int(M[i,j])),ha='center',va='center',color='white' if M[i,j]>3 else 'black')
for i,f in enumerate(['phase','shape']):
 values=[r['result']['phase_or_shape']['common_shift_share'] for r in DATA['axis_records'] if r['family']==f]
 axs[1].scatter(np.full(len(values),i)+np.linspace(-.1,.1,len(values)),values,color=[blue,orange][i])
axs[1].set_xticks([0,1],['Common phase shift','Gene-specific shift']);axs[1].tick_params(axis='x',labelsize=7);axs[1].set(ylabel='Common circular shift share',ylim=(-.03,1.03));fig.tight_layout();save(fig,3)
h=REAL['results']['GSE3431'];a=h['analysis'];fig,axs=plt.subplots(1,2,figsize=(7,3.1));bars=a['bar_lengths'][:6];axs[0].bar(np.arange(len(bars))+1,bars,color=blue);axs[0].axhline(a['null_threshold'],ls='--',color=red,label='Maximum-null threshold');axs[0].set(xlabel='H1 bar rank',ylabel='Normalised persistence');axs[0].legend(fontsize=7)
counts=a['stability']['resampled_counts'];axs[1].bar(np.arange(1,len(counts)+1),counts,color=grey);axs[1].set(xlabel='Reduced-cloud replicate',ylabel='Candidate detected count',ylim=(0,1.4));axs[1].set_title(f"Full-cloud candidate {a['candidate_count']}; status unresolved",fontsize=9);fig.tight_layout();save(fig,4)
fig,axs=plt.subplots(1,2,figsize=(7,3.2));fracs=['TOT','TR']
for j,frac in enumerate(fracs):
 r=REAL['results']['GSE158444_'+frac]['axis'];p=r['phase_or_shape'];axs[0].bar(j,p['common_shift_share'],color=blue);axs[0].bar(j,p['gene_specific_share'],bottom=p['common_shift_share'],color=grey)
 axs[1].scatter(j,r['dominant']['mantel_r_axis2'],color=orange,s=45)
axs[0].set_xticks([0,1],['Total RNA','Translating RNA']);axs[0].set(ylabel='Circular dispersion share',ylim=(0,1.05));axs[0].legend(['Common mean shift','Gene-specific residual'],fontsize=7,loc='center right');axs[1].set_xticks([0,1],['Total RNA','Translating RNA']);axs[1].set(ylabel='Heat-treatment distance association r',ylim=(0,1.08));fig.tight_layout();save(fig,5)
lines=['# Version 2 evidence','', 'These are current records from the frozen v2 protocol, not a reinterpretation of v1 summaries.','',
       '## Frozen simulated validation','', '| Generator | Reference H1 rank | Stable detected counts | Answers |','|---|---:|---|---:|']
for f,reference in zip(names,targets):
 s=DATA['summaries'][f];lines.append(f"| {f} | {reference} | {s['accepted_counts']} | {s['answered']}/{s['n']} |")
ag=DATA['aggregate'];lines+=['',f"Non-cyclic clouds: {ag['false_positive_k']}/{ag['negative_n']} positive detections; Wilson 95% interval {ag['false_positive_ci95']}. This small benchmark does not establish a zero population error rate.", '',
 'Single loops and equal-strength double loops recover their reference counts. Weak double loops are undercounted; sparse triple-cycle manifolds are not resolved. Stability does not guarantee completeness of the count.', '',
 'The 30 axis simulations separate common phase shifts, gene-specific changes and small effects. Both independently varying phases and discrete shifts can receive strong_axis_effect; this is deliberately not an independence label.', '',
 '## Fresh public-source analyses','',f"GSE3431: 36 samples and {h['n_retained_features']} selected features. Candidate count {a['candidate_count']}; resampling agreement {a['stability']['agreement']:.3f}; final status {a['status']}; circular Mantel r={a['design_association']['mantel_r']:.3f}. No permutation p is assigned to the dependent time series.", '',
 'The 500-feature analysis is also unresolved; the all-feature analysis detects no H1 features. This is one retrospective held-out source study, not an independent experimental replication or a confirmed topological positive.', '']
for frac in fracs:
 r=REAL['results']['GSE158444_'+frac];lines.append(f"GSE158444 {frac}: control {r['arms']['control']['status']}, heat {r['arms']['heat']['status']}; heat distance r={r['axis']['dominant']['mantel_r_axis2']:.3f}; common-shift share={r['axis']['phase_or_shape']['common_shift_share']:.3f}.")
lines+=['','Both fractions are one source study. The treatment association and gene-specific phase changes are descriptive reanalyses. No stable detected loop comparison supports a claim that heat destroys a cycle, and no novel mechanism is asserted.','',
 '## Reproduce and inspect','',
 '- `benchmark/v2/validation_plan.json`: frozen settings, algorithm digest, disjoint seed schedule.',
 '- `benchmark/v2/validation.json`: per-cloud spectra, surrogate maxima, resampling counts and environment.',
 '- `benchmark/v2/real_validation.json`: sample identities, feature IDs, source hashes and fresh analyses.',
 '- `reproduce/reproduce_v2.py`: verification, optional simulation rerun, optional download-to-results rerun.',
 '- `docs/results_v1.md`: historical records and claims only; use legacy APIs for their reproduction.', '']
(ROOT/'docs/results.md').write_text('\n'.join(lines))
print('Generated evidence and five editable SVG figures')
