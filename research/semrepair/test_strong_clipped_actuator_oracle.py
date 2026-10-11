import itertools
import math
import unittest
from research.semrepair.strong_clipped_actuator_oracle import (
    output_reachability_polygons,optimal_clipped_output,mm,clip_scalar,
    hull,nearest_convex_polygon
)

C=((1.,-.8),(0.,1.))


class ExactTwoClipOracleTests(unittest.TestCase):
    def setUp(self):
        self.poly=output_reachability_polygons(C)

    def test_zero_and_interior_feasible(self):
        for action in ((0.,0.),(.2,.2),(-.1,.3)):
            o=optimal_clipped_output(action,self.poly)
            self.assertLess(o["minimum_euclidean_error"],1e-9)

    def test_challenging_command_physically_unreachable(self):
        o=optimal_clipped_output((.8,.8),self.poly)
        self.assertGreater(o["minimum_euclidean_error"],.01)

    def test_dense_direct_physical_source_input_cannot_beat_exact_oracle(self):
        targets=((.8,.8),(-.8,-.8),(.9,-.2),(.6,.8),(0.,0.),
                 (.4,-.5),(-1.,1.),(1.,-1.))
        for target in targets:
            exact=optimal_clipped_output(target,self.poly)
            for i,j in itertools.product(range(-20,21),repeat=2):
                z=(i/20.,j/20.)
                y=mm(C,z)
                applied=tuple(clip_scalar(val,1) for val in y)
                self.assertGreaterEqual(
                    math.dist(applied,target)+1e-8,
                    exact["minimum_euclidean_error"])

    def test_output_polygon_hull_can_be_degenerate(self):
        self.assertEqual(hull([(0,0),(0,1),(0,.5),(0,1)]),[(0,0),(0,1)])
        self.assertEqual(nearest_convex_polygon((.2,.7),[(0.,0.),(0.,1.)]),(0.,.7))

    def test_clip_only_identity_has_full_box(self):
        polygons=output_reachability_polygons(((1.,0.),(0.,1.)))
        for target in ((.9,.9),(-.7,.4),(0.,0.)):
            r=optimal_clipped_output(target,polygons)
            self.assertLess(r["minimum_euclidean_error"],1e-9)

    def test_invalid_source_model_fails(self):
        with self.assertRaises(ValueError):
            output_reachability_polygons(((0.,0.),(0.,0.)))


if __name__=="__main__":
    unittest.main()
