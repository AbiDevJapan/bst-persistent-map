from types import SimpleNamespace
import pytest
from localization.frontend_adaptive import EarlyExitPolicy, verified_candidate


def result(conv, inliers):
    return SimpleNamespace(converged=conv,inliers=inliers)


def test_accepts_converged_geometric_candidate():
    assert verified_candidate(result(True,500),result(True,400),10000)


def test_rejects_nonconvergence_or_weak_support():
    assert not verified_candidate(result(True,500),result(False,500),10000)
    assert not verified_candidate(result(False,500),result(True,500),10000)
    assert not verified_candidate(result(True,500),result(True,250),10000)
    assert not verified_candidate(result(True,500),result(True,400),20000)
    assert not verified_candidate(result(True,500),None,10000)


def test_policy_validation():
    with pytest.raises(ValueError): EarlyExitPolicy(0,.1)
    with pytest.raises(ValueError): EarlyExitPolicy(300,float('nan'))
