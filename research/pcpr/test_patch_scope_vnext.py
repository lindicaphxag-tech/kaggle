import unittest
from patch_scope_vnext import score_one, FROZEN

BASE="""from torch import nn
class Net(nn.Module):
 def __init__(self):
  super().__init__()
  self.kernel = torch.ones((2, 2))
 def forward(self,x):
  return x @ self.kernel
"""
AFTER=BASE.replace("torch.ones((2, 2))","nn.Parameter(torch.ones((2, 2)))")

class FirstScoreContractTests(unittest.TestCase):
 def test_pinned_cohort_size(self):
  self.assertEqual(len(FROZEN),6)
 def test_exposed_change_creates_candidate(self):
  rows,status=score_one("toy.py",BASE,AFTER)
  self.assertEqual(status,"PARSED")
  self.assertEqual(len(rows),1)
  self.assertEqual(rows[0]["evidence_status"],"SOURCE_ANCHORED_CANDIDATE_ONLY")
 def test_patch_only_registration_not_authority(self):
  rows,_=score_one("toy.py",BASE,AFTER)
  self.assertNotEqual(rows[0]["evidence_status"],"SCOPED_WITNESS")
 def test_unchanged_source_not_patch_derived(self):
  rows,_=score_one("toy.py",BASE,AFTER)
  self.assertIsNotNone(rows[0]["old_consumer_line"])
  self.assertEqual(rows[0]["old_kind"],"assign")
  self.assertEqual(rows[0]["new_kind"],"Parameter")
 def test_no_before_consumer_abstain(self):
  b=BASE.replace("x @ self.kernel","x")
  rows,_=score_one("toy.py",b,AFTER)
  self.assertEqual(rows[0]["evidence_status"],"ABSTAIN_NO_UNCHANGED_CONSUMER")
 def test_after_pseudoconsumer_not_self_grounding(self):
  b=BASE.replace("x @ self.kernel","x")
  rows,_=score_one("toy.py",b,AFTER)
  self.assertIsNone(rows[0]["old_consumer_line"])
 def test_same_registration_kind_abstains(self):
  rows,_=score_one("toy.py",BASE,BASE.replace("torch.ones((2, 2))","torch.zeros((2, 2))"))
  self.assertEqual(rows[0]["evidence_status"],"ABSTAIN_SAME_REGISTRATION_KIND")
 def test_new_field_without_before_anchor_abstains(self):
  rows,_=score_one("toy.py",BASE,BASE.replace("self.kernel =", "self.pnew ="))
  self.assertTrue(all(x["evidence_status"].startswith("ABSTAIN") for x in rows))
 def test_non_module_not_certified(self):
  b=BASE.replace("Net(nn.Module)","Net(object)")
  a=AFTER.replace("Net(nn.Module)","Net(object)")
  rows,_=score_one("toy.py",b,a)
  self.assertFalse(any(x["evidence_status"]=="SCOPED_WITNESS" for x in rows))
 def test_malformed_source_fail_closed(self):
  rows,status=score_one("bad.py","class Net(:","class Net(): pass")
  self.assertEqual(status,"ABSTAIN_PARSE_OR_LANGUAGE")
  self.assertFalse(rows)
 def test_unchanged_no_findings(self):
  rows,_=score_one("toy.py",BASE,BASE)
  self.assertFalse(rows)
if __name__=="__main__":unittest.main()
