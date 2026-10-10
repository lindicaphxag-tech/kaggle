"""Synthetic smoke test for untouched frontier runner.

No EEG data are accessed. The test exercises:
- grouped source path over FULL + dense covariance grid;
- source-macro / covariance / lexicographic selections;
- score-family freeze and SHA256;
- post-freeze target evaluation;
- validity-frontier and grid-sensitivity outputs.
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np

import run_untouched_frontier as m


def make_covariances(seed=7, n_people=9, trials_per_class=8, p=5):
    rng=np.random.default_rng(seed)
    covs=[];ys=[];groups=[]
    base=np.eye(p)
    for sid in range(1,n_people+1):
        shift=0.05*rng.normal(size=(p,p))
        shift=(shift+shift.T)/2
        for label in (0,1):
            for _ in range(trials_per_class):
                A=rng.normal(size=(p,p))
                C=A@A.T/p + base
                if label==1:
                    C[0,0]+=0.35
                    C[1,1]+=0.15
                C+=0.03*shift
                e=np.linalg.eigvalsh((C+C.T)/2)
                if e.min()<=0:
                    C+=np.eye(p)*(abs(e.min())+1e-3)
                covs.append((C+C.T)/2)
                ys.append(label)
                groups.append(sid)
    return np.asarray(covs),np.asarray(ys),np.asarray(groups)


def main():
    C,y,g=make_covariances()
    outer=g==9
    inner=~outer

    path=m.source_path(C[inner],y[inner],g[inner])
    assert path["lambda_source_macro"] in m.FULL_GRID
    assert path["lambda_covariance_full"] in m.FULL_GRID
    assert path["lambda_lexicographic"] in m.FULL_GRID
    assert len(path["feasible_dense_lambdas"])>=8
    assert path["dense_covariance_regret_by_lambda"]

    with tempfile.TemporaryDirectory() as td:
        score_file=Path(td)/"scores.npz"
        receipt=m.freeze_target_scores(
            C[inner],y[inner],C[outer],path,score_file
        )
        assert score_file.exists()
        assert len(receipt["score_sha256"])==64

        result=m.evaluate_frozen_scores(
            score_file,
            y[outer],
            path["dense_covariance_regret_by_lambda"],
        )
        expected={
            "LedoitWolf_LDA",
            "OAS_LDA",
            "source_macro_full_domain_LDA",
            "source_cov_nll_full_domain_LDA",
            "discriminability_first_lexicographic_LDA",
            "TS_LogisticRegression",
            "Riemannian_MDM",
        }
        assert set(result["auc"])==expected
        assert result["validity_frontier"]["target_path_used_for_selection"] is False
        assert len(result["validity_frontier"]["grid_sensitivity"])==4
        assert result["validity_frontier"]["max_auc_span"]>=0.0

    print("PASS_SYNTHETIC_FRONTIER_SMOKE")


if __name__=="__main__":
    main()
