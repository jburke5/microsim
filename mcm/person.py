import numpy as np


class Person:
    """Person is using risk factors and demographics based off NHANES"""

    # TODO: please remember to come back and make race and gender enums
    # TODO: probably should also add a view of the "most recent" version of risk factor values
    def __init__(self, age, gender, race_ethnicity, sbp, dbp, a1c, hdl, tot_chol, **kwargs):
        self._gender = gender
        self._race_ethnicity = race_ethnicity

        self._alive = [True]

        self._age = [age]
        self._sbp = [sbp]
        self._dbp = [dbp]
        self._a1c = [a1c]
        self._hdl = [hdl]
        self._tot_chol = [tot_chol]
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_next_risk_factor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get(riskFactor)
        return model.estimateNextRisk(self._age[-1], self._gender, self._race_ethnicity,
                                      self._sbp[-1], self._dbp[-1], self._a1c[-1], self._hdl[-1],
                                      self._tot_chol[-1])

    def advance_risk_factors(self, risk_model_repository):
        ''' dummy risk models for now — 90% of your prior risk + a random normal intercept
        with a mean slightly greater than 10% of the population mean '''

        next_sbp = self.get_next_risk_factor("sbp", risk_model_repository)
        self._sbp.append(next_sbp)

        next_dbp = self.get_next_risk_factor("dbp", risk_model_repository)
        self._dbp.append(next_dbp)

        next_a1c = self._a1c[-1]
        self._a1c.append(next_a1c)

        next_hdl = self._hdl[-1]
        self._hdl.append(next_hdl)

        next_tot_chol = self._tot_chol[-1]
        self._tot_chol.append(next_tot_chol)

        self._age.append(self._age[-1]+1)

    def advance_outcomes(self):
        pass

    def __repr__(self):
        return (f"Person(age={self._age[-1]}, "
                f"gender={self._gender}, "
                f"race/eth={self._race_ethnicity}, "
                f"sbp={self._sbp[-1]}, "
                f"dbp={self._dbp[-1]}, "
                f"a1c={self._a1c[-1]}, "
                f"hdl={self._hdl[-1]}, "
                f"tot_chol={self._tot_chol[-1]}"
                f")")
