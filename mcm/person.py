import numpy as np


class Person:
    """Person is using risk factors and demographics based off NHANES"""

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

    def advanceRiskFactors(self):
        ''' dummy risk models for now — 90% of your prior risk + a random normal intercept 
        with a mean slightly greater than 10% of the population mean '''

        next_sbp = .9 * self._sbp[-1] + \
            np.random.normal(14, 5)
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
