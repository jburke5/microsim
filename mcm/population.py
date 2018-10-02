from mcm.person import Person
from mcm.linear_risk_factor_model import LinearRiskFactorModel
from statsmodels.regression.linear_model import OLSResults

import pandas as pd


class Population:
    """Unit of people subject to treatment program over time."""

    def __init__(self, people):
        self._people = people


def build_people_using_nhanes_for_sampling(nhanes, n, random_seed=None):
    repeated_sample = nhanes.sample(
        n, weights=nhanes.wtint2yr, random_state=random_seed, replace=True)
    people = repeated_sample.apply(lambda x: Person(
        x.age, x.gender, x.raceEthnicity, x.meanSBP, x.meanDBP, x.a1c, x.hdl, x.chol), axis=1)
    return people


class NHANESDirectSamplePopulation(Population):
    """ Simple base class to sample with replacement from 2015/2016 NHANES """

    def __init__(self, n=10000, random_seed=None):
        nhanes = pd.read_stata("mcm/nhanes2015-2016Combined.dta")
        super().__init__(build_people_using_nhanes_for_sampling(
            nhanes, n, random_seed=random_seed))
        self.n = n
        self._initializeRiskModels()

    def _initializeRiskModels(self):
        self._risk_model_repository = {}
        sbpModelResults = OLSResults.load(
            "/Users/burke/Documents/research/bpCog/nhanes/meanSBPModel.pickle")
        self._risk_model_repository['sbp'] = LinearRiskFactorModel(
            'sbp', sbpModelResults.params, sbpModelResults.bse)

    def advance(self, years):
        for year in range(1, years):
            for person in self._people:
                person.advanceRiskFactors(self._risk_model_repository)
                person.advanceOutcomes()
            self.applyRecalibrationStandards()

    def applyRecalibrationStandards(self):
        pass
