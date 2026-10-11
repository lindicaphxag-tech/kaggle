"""Actual, independently reported public ManiSkill #1469 sign bug regression."""
import hashlib
import unittest
from research.semrepair.maniskill_upstream_rotation_sign_case import (
    run_verified_maniskill_controller_sign_regression,
    _frozen_upstream_source,CONTROLLER,CONTROLLER_BLOB_SHA1,
    correct_one_source_attribute,N_ACTIONS
)

class VendorReportedRotationControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=run_verified_maniskill_controller_sign_regression()
    def test_frozen_actual_vendor_source_identity_and_issue(self):
        r=self.report
        self.assertEqual(r["actual_controller_git_blob"],CONTROLLER_BLOB_SHA1)
        self.assertIn("1469",r["upstream_issue"])
        self.assertIn("1472",r["upstream_open_pr"])
        self.assertFalse(r["PR_merged"])
        self.assertEqual(r["real_robot_or_SAPIEN_task_episodes"],0)
        self.assertEqual(r["unmodified_vendor_Python_AST_methods_executed"],2)
    def test_original_production_scaler_has_measured_nonzero_error(self):
        r=self.report
        self.assertEqual(r["tested_controller_actions"],N_ACTIONS)
        self.assertGreater(r["original_controller_error_L2_mean"],.04)
        self.assertEqual(r["compiled_one_ast_attribute_repair_error_L2_max"],0.)
        self.assertTrue(all(x["upstream_original_wrong_rot_minus_fixed_l2"]>0
                            for x in r["raw_actions"]))
    def test_equal_information_blackbox_sign_inversion_TIES_source_patch(self):
        r=self.report
        self.assertLess(r["equal_info_blackbox_sign_inversion_error_L2_max"],1e-12)
        self.assertIn("blackbox action-level inverse",r["interpretation"])
    def test_one_actual_PR_AST_edit_only(self):
        r=self.report["source_patch_changes"]
        self.assertEqual(r["changed_attribute_ast_nodes"],1)
        self.assertEqual(r["original_attr"],"rot_lower")
        self.assertEqual(r["proposed_attr"],"rot_upper")
        self.assertFalse(r["vendor_maintainer_merge_confirmed"])
    def test_forged_source_bytes_or_missing_authority_must_refuse(self):
        text,sha=_frozen_upstream_source(CONTROLLER,CONTROLLER_BLOB_SHA1)
        for kwargs,status in [
            ({"trusted_source_sha256":"0"*64},"REFUSE_WRONG_VENDOR_CONTROLLER_SOURCE_BYTES"),
            ({"patch_permission":False},"REFUSE_UNAUTHORIZED_SOURCE_MODIFICATION"),
            ({"requested_upstream_pr_number":0},"REFUSE_NOT_THE_REVIEWED_UPSTREAM_ROTATION_SIGN_CHANGE"),
        ]:
            m,code=correct_one_source_attribute(
                frozen_controller_source=text,
                trusted_source_sha256=kwargs.get("trusted_source_sha256",sha),
                patch_permission=kwargs.get("patch_permission",True),
                requested_upstream_pr_number=kwargs.get("requested_upstream_pr_number",1472))
            self.assertEqual(m["decision"],status)
            self.assertIsNone(code)
if __name__=="__main__":
    unittest.main()
