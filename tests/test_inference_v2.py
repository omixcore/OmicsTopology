"""Regression cases for scientific inference errors found in the v1 audit."""
import numpy as np
import pytest
from omictopo.inference import signal_spectrum, spectral_count, stable_cycle_count
from omictopo.protocol import analyse
from omictopo.axes import axis_geometry, circular_phase_decomposition
from omictopo.stats import mantel, circular_distance
from omictopo.geometry import pairwise_distances


def cloud(k=1, seed=101):
    rng=np.random.default_rng(seed)
    if k==1:
        t=np.arange(36)*2*np.pi/36
        coords=np.c_[np.cos(t),np.sin(t)]
    else:
        t=np.repeat(np.arange(8),8)*2*np.pi/8
        q=np.tile(np.arange(8),8)*2*np.pi/8
        coords=np.c_[np.cos(t),np.sin(t),np.cos(q),np.sin(q)]
    X=coords@rng.normal(size=(coords.shape[1],64))+.03*rng.normal(size=(len(t),64))
    return X,t


def test_two_equal_signal_bars_do_not_exclude_each_other():
    r=signal_spectrum([.8,.79,.07],np.linspace(.1,.2,99))
    assert r['candidate_count']==2
    assert r['bar_pvalues_max_null'][:2]==[.01,.01]


def test_maximum_null_is_used_for_all_bars_with_conservative_ties():
    r=signal_spectrum([.2,.1],np.full(99,.2))
    assert r['candidate_count']==0
    assert r['bar_pvalues_max_null']==[1.,1.]


def test_torus_has_two_detected_features_even_with_similar_lengths():
    X,_=cloud(2)
    r=stable_cycle_count(X,n_null=39,n_resamples=4,resample_n_null=39,seed=0)
    assert r['candidate_count']==2
    assert r['n_cycles']==2
    assert r['status']=='counted'


def test_wrong_design_labels_do_not_change_the_topological_count():
    X,t=cloud()
    a=analyse(X,t,n_null=39,n_resamples=4,resample_n_null=39,seed=3)
    b=analyse(X,np.random.default_rng(6).permutation(t),n_null=39,n_resamples=4,resample_n_null=39,seed=3)
    assert a['n_cycles']==b['n_cycles']==1
    assert a['design_association']['mantel_r']>b['design_association']['mantel_r']
    assert a['design_association']['mantel_p'] is None


def test_missing_stability_is_abstention_not_a_claim():
    X,_=cloud()
    r=stable_cycle_count(X,n_null=39,n_resamples=0)
    assert r['candidate_count']==1 and r['n_cycles'] is None
    assert r['status']=='unresolved'


def test_pinned_backend_does_not_silently_fall_back(monkeypatch):
    import omictopo.inference as m
    monkeypatch.setattr(m.importlib.util,'find_spec',lambda name:None)
    with pytest.raises(ImportError):spectral_count(cloud()[0],n_null=39)


def test_short_null_and_invalid_data_are_rejected():
    with pytest.raises(ValueError):spectral_count(cloud()[0],n_null=4)
    with pytest.raises(ValueError):spectral_count(np.ones((8,4)),n_null=39)
    X=cloud()[0];X[0,0]=np.nan
    with pytest.raises(ValueError):spectral_count(X,n_null=39)


def test_mantel_permutations_preserve_matrix_rows_and_columns():
    a=pairwise_distances(np.c_[np.arange(6),np.arange(6)**2])
    b=pairwise_distances(np.c_[np.arange(6)[::-1],np.arange(6)**3])
    iu=np.triu_indices(6,1);obs=np.corrcoef(a[iu],b[iu])[0,1]
    rng=np.random.default_rng(32);hits=0
    for _ in range(39):
        order=rng.permutation(6)
        hits+=np.corrcoef(a[iu],b[np.ix_(order,order)][iu])[0,1]>=obs-1e-12
    r,p=mantel(a,b,n_perm=39,seed=32)
    assert r==pytest.approx(obs)
    assert p==pytest.approx((hits+1)/40)


def test_nonexchangeable_and_degenerate_blocks_have_no_pvalue():
    D=pairwise_distances(np.arange(8)[:,None])
    assert mantel(D,D,permutation='none')[1] is None
    assert mantel(D,D,strata=np.arange(8))[1] is None


def test_circular_distance_handles_wraps_and_arbitrary_origin():
    a=circular_distance(np.array([-.1,0,2*np.pi+.1]))
    b=circular_distance(np.array([-.1,0,2*np.pi+.1])+17*np.pi)
    assert a[0,2]==pytest.approx(.2)
    assert np.allclose(a,b)


def phases(kind='common'):
    rng=np.random.default_rng(113)
    t=np.tile(np.arange(12)*2*np.pi/12,4);blocks=np.repeat(np.arange(4),12)
    genes=rng.uniform(-np.pi,np.pi,48)
    offset=np.array([0,.4,.8,1.2])[blocks,None]
    if kind=='specific':offset=offset*np.where(np.arange(48)%2,1,-1)
    X=np.cos(t[:,None]+genes+offset)*3+20+.01*rng.normal(size=(48,48))
    return X,t,blocks


def test_circular_phase_decomposition_is_invariant_to_phase_origin():
    X,t,b=phases()
    a=circular_phase_decomposition(X,t,b)
    c=circular_phase_decomposition(X,t+3.1,b)
    assert a['common_shift_share']>.99
    assert a['gene_offset_coherence']>.99
    assert c['common_shift_share']==pytest.approx(a['common_shift_share'],abs=1e-8)


def test_gene_specific_phase_changes_are_not_called_common_shift():
    X,t,b=phases('specific');r=axis_geometry(X,t,b,n_boot=20)
    assert r['phase_or_shape']['common_shift_share']<.1
    assert r['verdict']=='shape_change'
    assert r['cyclic_evidence']['independent_second_cycle']=='not_established'


def test_category_coding_does_not_change_axis_distance_association():
    X,t,b=phases()
    a=axis_geometry(X,t,b,n_boot=20)
    c=axis_geometry(X,t,np.array(['z','a','q','x'])[b],n_boot=20)
    assert a['dominant']['mantel_r_axis2']==pytest.approx(c['dominant']['mantel_r_axis2'])
    assert a['verdict']==c['verdict']=='phase_modulation'


def test_phase_fit_with_no_block_coverage_refuses():
    X,t,b=phases()
    r=circular_phase_decomposition(X[:8],t[:8],np.arange(8))
    assert r['status']=='unresolved'


def test_stratification_that_cannot_change_coordinate_reports_no_pvalue():
    phase=np.repeat(np.arange(4),3)
    D=circular_distance(phase,period=4)
    assert mantel(D,D,strata=phase,n_perm=39)[1] is None

def test_phase_uncertainty_uses_wrapped_intervals():
    X,t,b=phases()
    r=axis_geometry(X,t,b,n_boot=30)
    u=r['phase_or_shape']['uncertainty']
    assert u['status']=='resolved'
    assert len(u['block_offset_ci95_rad'])==4
    assert all(lo<=hi for lo,hi in u['block_offset_ci95_rad'])
