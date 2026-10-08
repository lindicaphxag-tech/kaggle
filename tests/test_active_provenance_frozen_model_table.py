"""A nonlinear frozen encoder can use the same exact identity query decision tree.

Checks complete scenario table, original linear backend parity, and refused
out-of-scope identity mappings. Never equate sign invariance with correctness.
"""
import unittest
import numpy as np
from research.eeg.active_provenance_queries import (
    ProvenanceQueryPlanner, DecisionTableQueryPlanner,
)

class ModelAgnosticQueryTests(unittest.TestCase):
    def test_model_agnostic_table_matches_linear_oracle(self):
        x=np.array([[1.],[-1.],[.5],[-.5]])
        w=np.array([[1.],[-1.],[.7],[-.7]])
        linear=ProvenanceQueryPlanner(x,w,0.,[(0,1),(2,3)])
        table=DecisionTableQueryPlanner(
            linear.mappings,linear.margins,[(0,1),(2,3)])
        self.assertEqual(table.propose(),linear.propose())
        for true_mapping in linear.mappings:
            self.assertEqual(table.interact(true_mapping),
                             linear.interact(true_mapping))

    def test_incomplete_table_never_authorizes(self):
        linear=ProvenanceQueryPlanner(
            np.array([[1.],[-1.],[.5],[-.5]]),
            np.array([[1.],[-1.],[.7],[-.7]]),0.,[(0,1),(2,3)])
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            DecisionTableQueryPlanner(
                linear.mappings[:-1],linear.margins[:-1],[(0,1),(2,3)])

    def test_nonfinite_decision_refused(self):
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            DecisionTableQueryPlanner(
                [(0,1),(1,0)],[1.,float("nan")],[(0,1)])

if __name__=="__main__":
    unittest.main()
