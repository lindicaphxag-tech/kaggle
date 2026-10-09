import unittest
from research.semrepair.observable_interface_factorization import (
    TapSample, nominal_probe, identify_repair_from_numeric_taps,
    blind_closed_loop_planar_assay, I, PROBE_E1, PROBE_E2, CHECK,
)

class InterfaceFactorizationTests(unittest.TestCase):
    def test_noop_identity(self):
        train = (nominal_probe(I, I, PROBE_E1),nominal_probe(I, I, PROBE_E2))
        out=identify_repair_from_numeric_taps(train,nominal_probe(I, I, CHECK))
        self.assertEqual(out["candidate_patch"],"noop")

    def test_rank_deficient_refusal(self):
        x=TapSample((.5,0),(.5,0),(.5,0))
        y=TapSample((1.,0),(1.,0),(1.,0))
        out=identify_repair_from_numeric_taps((x,y),nominal_probe(I,I,CHECK))
        self.assertEqual(out["decision"],"REFUSE_RANK_DEFICIENT")

    def test_nonlinear_holdout_fails(self):
        train=(nominal_probe(I,I,PROBE_E1),nominal_probe(I,I,PROBE_E2))
        hold=TapSample(CHECK,(CHECK[0]+.1,CHECK[1]),CHECK)
        self.assertEqual(identify_repair_from_numeric_taps(train,hold)["decision"],
                         "REFUSE_NONLINEAR_OR_INCONSISTENT")

    def test_one_probe_not_a_general_linear_certificate(self):
        train=(nominal_probe(I,I,PROBE_E1),)
        with self.assertRaises(ValueError):
            identify_repair_from_numeric_taps(train,nominal_probe(I,I,CHECK))

    def test_unknown_linear_shear_recovered(self):
        p=((1.,.25),(0.,1.))
        c=((1.,-.25),(0.,1.))
        train=(nominal_probe(p,c,PROBE_E1),nominal_probe(p,c,PROBE_E2))
        out=identify_repair_from_numeric_taps(train,nominal_probe(p,c,CHECK))
        self.assertEqual(out["candidate_patch"],"atomic")
        self.assertLess(out["heldout_controller_error"],1e-12)

    def test_blind_task_replay(self):
        r=blind_closed_loop_planar_assay()
        self.assertEqual(r["source_case_count"],6)
        self.assertEqual(len(r["rows"]),96)
        self.assertFalse(r["input_oracle_labels_visible_to_solver"])
        self.assertFalse(r["source_attestation_implemented"])
        for k,v in r["results"].items():
            self.assertEqual(v["blind_candidate_success"],16,k)

if __name__=="__main__":
    unittest.main()
