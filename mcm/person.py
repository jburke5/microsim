import numpy as np


class Person:
    """Person is using risk factors and demographics based off NHANES"""

    # TODO: please remember to come back and make race and gender enums
    # TODO: probably should also add a view of the "most recent" version of risk factor values
    def __init__(self, age, gender, race_ethnicity, sbp, dbp, a1c, hdl, chol):
        self._gender = gender
        self._race_ethnicity = race_ethnicity

        self._alive = [True]

        self._age = [age]
        self._sbp = [sbp]
        self._dbp = [dbp]
        self._a1c = [a1c]
        self._hdl = [hdl]
        self._chol = [chol]

    def getNextRiskFactor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get(riskFactor)
        return model.estimateNextRisk(self._age[-1], self._gender, self._race_ethnicity,
                                      self._sbp[-1], self._dbp[-1], self._a1c[-1], self._hdl[-1],
                                      self._chol[-1])

    def advanceRiskFactors(self, risk_model_repository):
        ''' dummy risk models for now — 90% of your prior risk + a random normal intercept 
        with a mean slightly greater than 10% of the population mean '''

        next_sbp = self.getNextRiskFactor("sbp", risk_model_repository)
        self._sbp.append(next_sbp)

        next_dbp = .9 * self._dbp[-1] + np.random.normal(7.5, 3)
        self._dbp.append(next_dbp)

        next_a1c = .9 * self._a1c[-1] * np.random.normal(.6, .3)
        self._a1c.append(next_a1c)

        next_hdl = .9 * self._hdl[-1] * np.random.normal(5.6, 3)
        self._hdl.append(next_hdl)

        next_chol = .9 * self._chol[-1] * np.random.normal(20, 8)
        self._chol.append(next_chol)

        self._age.append(self._age[-1]+1)

    def advanceOutcomes(self):
        pass
