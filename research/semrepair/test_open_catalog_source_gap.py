import unittest

from research.semrepair.open_catalog_source_gap import (
    UNSEEN, study, unseen_rollout, decide_from_sensor_traces, known_source_digest,
    HELD_EXCITATIONS, WORLDS, rollout, max_error, signature,
)


class OpenCatalogPhysicalSourceGap(unittest.TestCase):
    def test_sensored_source_alias_is_not_a_known_repair(self):
        """Two different software packages share EVERY genuine sensor trace."""
        commands=HELD_EXCITATIONS[0]
        known=next(w for w in WORLDS if w.name=="gain_fault")
        unknown=next(w for w in UNSEEN if w.id=="foreign-identical-gain")
        x=rollout(known,commands)
        y=unseen_rollout(unknown,commands)
        for channel in ("external","producer_to_controller_ctrl","transmission_force"):
            self.assertEqual(signature(x[channel]),signature(y[channel]))
        self.assertNotEqual(known.authority,unknown.physical.authority)

    def test_open_catalog_bad_authorization_and_source_guard(self):
        r=study()
        self.assertEqual(r["open_catalog_condition_cells"],12)
        self.assertEqual(r["identical_physics_different_repair_wrong_without_source"],3)
        self.assertEqual(r["with_trusted_source_rejected_unknown"],12)
        self.assertGreaterEqual(r["without_source_assumption_authorized_unknown"],3)
        self.assertGreaterEqual(r["without_source_assumption_rejected_unknown"],1)

    def test_no_source_is_refused_if_source_rule_required(self):
        commands=HELD_EXCITATIONS[0]
        templates={w.name:rollout(w,commands) for w in WORLDS}
        response=decide_from_sensor_traces(
            measured=templates["gain_fault"],templates=templates,
            enforce_source_provenance=True)
        self.assertEqual(response["answer"],"REFUSE")
        self.assertEqual(response["reason"],"no_verified_source")

    def test_verified_software_but_changed_physics_is_refused(self):
        commands=HELD_EXCITATIONS[0]
        known=next(w for w in WORLDS if w.name=="gain_fault")
        altered=next(w for w in UNSEEN if w.id=="foreign-mass-reparameterized")
        templates={w.name:rollout(w,commands) for w in WORLDS}
        response=decide_from_sensor_traces(
            measured=unseen_rollout(altered,commands),templates=templates,
            source_digest=known_source_digest(known),
            enforce_source_provenance=True)
        self.assertEqual(response["answer"],"REFUSE")
        self.assertEqual(response["reason"],"out_of_model_observation")


if __name__=="__main__":
    unittest.main()
