"""Lee held-runner fail-closed and paired stats tests without actual target EEG."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import run_lee2019_gid_held_sha_locked as r

def test_frozen_target_id_partitions_no_overlap():
    assert tuple(r.lock.SRC)==tuple(range(1,13))
    assert tuple(r.lock.TARGET_A)==tuple(range(13,25))
    assert tuple(r.lock.TARGET_B)==tuple(range(25,37))
    assert set(r.lock.SRC).isdisjoint(r.lock.TARGET_A)
    assert set(r.lock.TARGET_A).isdisjoint(r.lock.TARGET_B)

def test_signflip_exact_null_and_directional_reference():
    assert r.signflip(np.zeros(12))==1.0
    assert np.isclose(r.signflip(np.ones(12)),2/4096)
    assert r.signflip(np.r_[np.ones(6),-np.ones(6)])==1.
    with pytest.raises(AssertionError):
        r.signflip(np.ones(11))

def test_held_manifest_bad_sha_denied_before_data_download(monkeypatch,tmp_path):
    def must_not_download():
        raise AssertionError("target/source IO happened before validating SHA")
    monkeypatch.setattr(r,"load_sources",must_not_download)
    p=tmp_path/"source.json";p.write_text('{"not":"real"}')
    with pytest.raises(AssertionError,match="ORIGINAL_SHA_MISMATCH_STOP"):
        r.main(p,"invalid-sha",tmp_path/"held","A")

def test_twelve_person_group_macro_not_trial_pooled():
    persons=[]
    for id in range(13,25):
        baseline=.6
        improvement=.03 if id<19 else -.01
        persons.append({"person_id":id,"trials":100+id,
                        "auc":{"gid":baseline+improvement,"original":baseline}})
    d=r.result_block("A",persons,r.lock.TARGET_A)
    assert d["n_persons"]==12
    assert d["n_trials"]==sum(100+id for id in range(13,25))
    assert np.isclose(d["gid_minus_original_mean_delta"],.01)
    assert d["gain_positive_persons"]==6
    assert d["gain_negative_persons"]==6
    assert d["gid_losses_gt_002"]==0
    with pytest.raises(AssertionError):
        r.result_block("B",persons,r.lock.TARGET_B)
