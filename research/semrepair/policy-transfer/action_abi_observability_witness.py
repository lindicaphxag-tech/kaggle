"""Stateful controller ABI: observational aliasing and exact-transfer witnesses.

This executable counterexample is deliberately NOT a ManiSkill experiment.
It studies 1-D target-relative actuation under explicitly stated assumptions:

  source desired goal g = x + u (achieved-relative source control)
  target next command q = t + v (previous-target-relative destination)
  bounded destination control |v| <= B

A correct exact adapter needs v = g - t, while a state-blind controller
that substitutes achieved pose x for target memory t uses v_blind = u.

Two worlds with the same public x, u and observed history but different
hidden initial targets t have different required v, proving no deterministic
public-history-only adapter can be exact in both worlds *unless the history
contains sufficient information to identify t*. This is a scoped
nonidentifiability witness, not a safety theorem.
"""
from dataclasses import dataclass
import unittest


@dataclass(frozen=True)
class World:
    x: float
    target_memory: float
    source_action: float
    bound: float = 1.0

    @property
    def source_goal(self) -> float:
        return self.x + self.source_action

    @property
    def required_target_action(self) -> float:
        return self.source_goal - self.target_memory

    def exact_feasible(self) -> bool:
        return abs(self.required_target_action) <= self.bound

    def execute(self, target_action: float) -> float:
        if abs(target_action) > self.bound:
            raise ValueError("destination normalized action out of bounds")
        return self.target_memory + target_action


def live_memory_adapter(world: World) -> float:
    """Only returns an exact command when representable."""
    v = world.required_target_action
    if not world.exact_feasible():
        raise ValueError("exact transport impossible under destination bound")
    return v


def state_blind_adapter(public_x: float, source_action: float) -> float:
    """Achieved-pose substitution; intentionally no access to target memory."""
    del public_x
    return source_action


def history_observer_known_reset(initial_target: float, prior_actions: tuple[float, ...]) -> float:
    """Under the additive-target model, reset target + full command history suffices."""
    return initial_target + sum(prior_actions)


def indistinguishable_public_worlds() -> tuple[World, World]:
    # Both have identical x, source action and empty observable history;
    # target memory differs because its initial value was not observed.
    return World(x=0.2, target_memory=0.2, source_action=0.1), World(
        x=0.2, target_memory=0.5, source_action=0.1
    )


class ActionABIWitnessTests(unittest.TestCase):
    def test_observation_aliasing_prevents_universal_exact_adapter(self):
        a, b = indistinguishable_public_worlds()
        self.assertEqual((a.x, a.source_action), (b.x, b.source_action))
        self.assertNotEqual(a.required_target_action, b.required_target_action)
        self.assertTrue(a.exact_feasible() and b.exact_feasible())
        # Same public information forces deterministic state-blind command to be equal.
        v = state_blind_adapter(a.x, a.source_action)
        self.assertEqual(v, state_blind_adapter(b.x, b.source_action))
        self.assertNotEqual(a.execute(v), b.source_goal)
        self.assertEqual(a.execute(live_memory_adapter(a)), a.source_goal)
        self.assertEqual(b.execute(live_memory_adapter(b)), b.source_goal)

    def test_full_command_history_after_known_reset_recovers_memory(self):
        initial = 0.2
        history = (0.1, -0.05, 0.25)
        t = history_observer_known_reset(initial, history)
        world = World(x=0.22, target_memory=t, source_action=0.15)
        recovered = world.source_goal - t
        self.assertAlmostEqual(world.execute(recovered), world.source_goal)
        self.assertAlmostEqual(recovered, live_memory_adapter(world))

    def test_bounded_exactness_is_not_guaranteed(self):
        world = World(x=0.0, target_memory=2.0, source_action=0.1, bound=1.0)
        self.assertFalse(world.exact_feasible())
        with self.assertRaises(ValueError):
            live_memory_adapter(world)

    def test_state_blind_fails_without_memory_even_for_legal_action(self):
        a, b = indistinguishable_public_worlds()
        v = state_blind_adapter(b.x, b.source_action)
        self.assertLessEqual(abs(v), b.bound)
        self.assertNotAlmostEqual(b.execute(v), b.source_goal)


if __name__ == "__main__":
    unittest.main()
