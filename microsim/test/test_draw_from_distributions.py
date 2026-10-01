import unittest

import numpy as np

from microsim.common.variable_type import VariableType
from microsim.population.population_factory import PopulationFactory


def _distributions(distMin, distMax, size=5):
    """Builds the smallest distributions dict draw_from_distributions accepts: one group, a standard
       normal over the NHANES continuous variables, and whatever bounds the test wants to impose.
       singular is False so that the group draws from its own distribution and the alt key is never
       read (get_alt_groups is what fills that in for the real dicts)."""
    nVariables = len(PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value])
    key = ("onlyGroup",)
    return {"mean": {key: np.zeros(nVariables)},
            "cov": {key: np.identity(nVariables)},
            "min": {key: distMin*np.ones(nVariables)},
            "max": {key: distMax*np.ones(nVariables)},
            "singular": {key: False},
            "size": {key: size},
            "names": {key: list(range(size))}}, key


class TestDrawFromDistributionsTerminates(unittest.TestCase):
    """The draw loop re-draws whatever falls outside the group bounds. Bounds the distribution
       almost never satisfies used to spin forever instead of failing."""

    def test_unsatisfiable_bounds_raise_instead_of_hanging(self):
        #every draw is below 0.9*min and above 1.1*max at once, so nothing can ever be accepted
        distributions, _ = _distributions(distMin=1000., distMax=-1., size=3)
        with self.assertRaises(RuntimeError) as cm:
            PopulationFactory.draw_from_distributions(distributions, maxDraws=30)
        self.assertIn("acceptance rate", str(cm.exception))

    def test_the_budget_is_reported_with_the_group(self):
        distributions, key = _distributions(distMin=1000., distMax=-1., size=3)
        with self.assertRaises(RuntimeError) as cm:
            PopulationFactory.draw_from_distributions(distributions, maxDraws=30)
        self.assertIn(str(key), str(cm.exception))


class TestDrawFromDistributionsStillDraws(unittest.TestCase):
    """The budget must not disturb groups whose bounds are satisfiable."""

    def test_satisfiable_bounds_return_exactly_size_draws(self):
        size = 10
        #a standard normal essentially always falls inside these, so the first pass is accepted whole
        distributions, key = _distributions(distMin=-5., distMax=5., size=size)
        drawsForGroups, namesForGroups = PopulationFactory.draw_from_distributions(distributions)
        self.assertEqual(size, drawsForGroups[key].shape[0])
        self.assertEqual(size, len(namesForGroups[key]))

    def test_bounds_are_honored_in_the_returned_draws(self):
        size = 20
        distributions, key = _distributions(distMin=-1., distMax=1., size=size)
        drawsForGroups, _ = PopulationFactory.draw_from_distributions(distributions)
        draws = drawsForGroups[key]
        #the loop widens the bounds by 10% before rejecting, so that is what the draws have to respect
        self.assertTrue(np.all(draws >= 0.9*(-1.)))
        self.assertTrue(np.all(draws <= 1.1*1.))


if __name__ == "__main__":
    unittest.main()
