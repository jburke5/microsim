from mcm.person import Person
from mcm.linear_risk_factor_model import LinearRiskFactorModel
from mcm.log_linear_risk_factor_model import LogLinearRiskFactorModel
from statsmodels.regression.linear_model import OLSResults

import pandas as pd


class Population:
    """Unit of people subject to treatment program over time."""

    def __init__(self, people):
        self._people = people


def build_people_using_nhanes_for_sampling(nhanes, n, random_seed=None):
    repeated_sample = nhanes.sample(
        n, weights=nhanes.WTINT2YR, random_state=random_seed, replace=True)
    people = repeated_sample.apply(
        lambda x: Person(
            age=x.age,
            gender=x.gender,
            race_ethnicity=x.raceEthnicity,
            sbp=x.meanSBP,
            dbp=x.meanDBP,
            a1c=x.a1c,
            hdl=x.hdl,
            tot_chol=x.tot_chol,
            dfIndex=x.index,
            diedBy2011=x.diedBy2011), axis=1)
    return people


class NHANESDirectSamplePopulation(Population):
    """ Simple base class to sample with replacement from 2015/2016 NHANES """

    def __init__(self, n, year, random_seed=None):
        nhanes = pd.read_stata("mcm/data/fullyImputedDataset.dta")
        nhanes = nhanes.loc[nhanes.year == year]
        super().__init__(build_people_using_nhanes_for_sampling(
            nhanes, n, random_seed=random_seed))
        self.n = n
        self._initialize_risk_models()

    def _initialize_risk_models(self):
        self._risk_model_repository = {}
        sbpModelResults = OLSResults.load(
            "mcm/data/logSBPModel.pickle")
        dbpModelResults = OLSResults.load(
            "mcm/data/meanDBPModel.pickle")
        self._risk_model_repository['sbp'] = LogLinearRiskFactorModel(
            'sbp', sbpModelResults.params, sbpModelResults.bse)
        self._risk_model_repository['dbp'] = LinearRiskFactorModel(
            'dbp', dbpModelResults.params, dbpModelResults.bse)

    def advance(self, years):
        for _ in range(years):
            for person in self._people:
                person.advance_risk_factors(self._risk_model_repository)
                person.advance_outcomes()
            self.apply_recalibration_standards()

    def apply_recalibration_standards(self):
        pass
