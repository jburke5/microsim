

class Person:
    """Person is using risk factors and demographics based off NHANES"""

    # TODO: probably should also add a view of the "most recent" version of risk factor values
    def __init__(self, age, gender, race_ethnicity, sbp, dbp, a1c, hdl, tot_chol, bmi,
                 smoking_status, **kwargs):

        # building in manual bounds on extreme values
        self._lowerBounds = {"sbp": 60, "dbp": 20}
        self._upperBounds = {"sbp": 300, "dbp": 180}

        self._gender = gender
        self._race_ethnicity = race_ethnicity

        self._alive = [True]

        self._age = [age]
        self._sbp = [self.apply_bounds("sbp", sbp)]
        self._dbp = [self.apply_bounds("dbp", dbp)]
        self._a1c = [a1c]
        self._hdl = [hdl]
        self._tot_chol = [tot_chol]
        self._bmi = [bmi]
        # TODO : change smoking status into a factor that changes over time
        self._smoking_status = smoking_status
        for k, v in kwargs.items():
            setattr(self, k, v)

    def has_diabetes(self):
        return sorted(self._a1c)[-1] >= 6.5

    def get_next_risk_factor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get(riskFactor)
        return model.estimate_next_risk(self._age[-1], self._gender, self._race_ethnicity,
                                        self._sbp[-1], self._dbp[-1], self._a1c[-1], self._hdl[-1],
                                        self._tot_chol[-1], self._bmi[-1], self._smoking_status)

    def apply_bounds(self, varName, varValue):
        """
        Ensures that risk factor are within static prespecified bounds.

        Other algorithms might be needed in the future to avoid pooling in the tails,
        if there are many extreme risk factor results.
        """
        if varName in self._upperBounds:
            upperBound = self._upperBounds[varName]
            varValue = varValue if varValue < upperBound else upperBound
        if varName in self._lowerBounds:
            lowerBound = self._lowerBounds[varName]
            varValue = varValue if varValue > lowerBound else lowerBound
        return varValue

    def advance_risk_factors(self, risk_model_repository):

        self._sbp.append(self.apply_bounds(
            "sbp", self.get_next_risk_factor("sbp", risk_model_repository)))

        self._dbp.append(self.apply_bounds(
            "dbp", self.get_next_risk_factor("dbp", risk_model_repository)))
        self._a1c.append(self.get_next_risk_factor("a1c", risk_model_repository))
        self._hdl.append(self.get_next_risk_factor("hdl", risk_model_repository))
        self._tot_chol.append(self.get_next_risk_factor("tot_chol", risk_model_repository))
        self._bmi.append(self.get_next_risk_factor("bmi", risk_model_repository))
        self._age.append(self._age[-1]+1)

    def advance_outcomes(self):
        # do you have an event?
        pass

        # if so, which event did you have?

    def __repr__(self):
        return (f"Person(age={self._age[-1]}, "
                f"gender={self._gender}, "
                f"race/eth={self._race_ethnicity}, "
                f"sbp={self._sbp[-1]}, "
                f"dbp={self._dbp[-1]}, "
                f"a1c={self._a1c[-1]}, "
                f"hdl={self._hdl[-1]}, "
                f"tot_chol={self._tot_chol[-1]}"
                f"bmi={self.bmi[-1]}"
                f"smoking={self.smoking_status}"
                f")")
