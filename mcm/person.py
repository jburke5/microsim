from mcm.gender import NHANESGender
from mcm.outcome_type import OutcomeType
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus

import numpy.random as npRand


class Person:
    """Person is using risk factors and demographics based off NHANES"""

    # TODO: probably should also add a view of the "most recent" version of risk factor values
    def __init__(
        self,
        age: int,
        gender: NHANESGender,
        raceEthnicity: NHANESRaceEthnicity,
        sbp: int,
        dbp: int,
        a1c: float,
        hdl: int,
        totChol: int,
        bmi: float,
        ldl: int,
        trig: int,
        smokingStatus: SmokingStatus,
        **kwargs,
    ) -> None:

        # building in manual bounds on extreme values
        self._lowerBounds = {"sbp": 60, "dbp": 20}
        self._upperBounds = {"sbp": 300, "dbp": 180}

        self._gender = gender
        self._raceEthnicity = raceEthnicity

        self._alive = [True]

        self._age = [age]
        self._sbp = [self.apply_bounds("sbp", sbp)]
        self._dbp = [self.apply_bounds("dbp", dbp)]
        self._a1c = [a1c]
        self._hdl = [hdl]
        self._ldl = [ldl]
        self._trig = [trig]
        self._totChol = [totChol]
        self._bmi = [bmi]
        # TODO : change smoking status into a factor that changes over time
        self._smokingStatus = smokingStatus

        # TODO : initialize with prior stroke and mi
        # outcomes is a list of dictionaries. for each year a dictionary of new
        # outcomes will be added.
        self._mi = 0
        self._stroke = 0
        self._outcomes = [{}]
        for k, v in kwargs.items():
            setattr(self, k, v)

    def has_diabetes(self):
        return sorted(self._a1c)[-1] >= 6.5

    def get_next_risk_factor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get_model(riskFactor)
        return model.estimate_next_risk(self)

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

    def advance_year(self, risk_model_repository, outcome_model_repository):
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance year")

        self.advance_risk_factors(risk_model_repository)
        self.advance_outcomes(outcome_model_repository)
        if not self.is_dead():
            self._age.append(self._age[-1] + 1)

    def is_dead(self):
        return not self._alive[-1]

    def advance_risk_factors(self, risk_model_repository):
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance risk factors")

        self._sbp.append(self.apply_bounds(
            "sbp", self.get_next_risk_factor("sbp", risk_model_repository)))

        self._dbp.append(self.apply_bounds(
            "dbp", self.get_next_risk_factor("dbp", risk_model_repository)))
        self._a1c.append(self.get_next_risk_factor("a1c", risk_model_repository))
        self._hdl.append(self.get_next_risk_factor("hdl", risk_model_repository))
        self._totChol.append(self.get_next_risk_factor("totChol", risk_model_repository))
        self._bmi.append(self.get_next_risk_factor("bmi", risk_model_repository))
        self._ldl.append(self.get_next_risk_factor("ldl", risk_model_repository))
        self._trig.append(self.get_next_risk_factor("trig", risk_model_repository))

    def _has_cvd_event(self, ascvdModel):
        return npRand.uniform(size=1) < ascvdModel.estimate_next_risk(self)

    def _has_mi(self, miProbability):
        return npRand.uniform(size=1) < miProbability

    def _has_fatal_mi(self, fatalMIProb):
        return npRand.uniform(size=1) < fatalMIPRob

    def _has_fatal_stroke(self, fatalStrokeProb):
        return npRand.uniform(size=1) < fatalStrokeProb

    def advance_outcomes(self, outcome_model_repository):
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance outcomes")

        miProbability = 0.7
        fatalMIPRob = 0.20
        fatalStrokeProb = 0.20

        ascvdModel = outcome_model_repository.select_model_for_person(
            self, OutcomeType.CARDIOVASCULAR)
        outcomesForThisYear = {}
        if self._has_cvd_event(ascvdModel):
            if self._has_mi(miProbability):
                if self._has_fatal_mi(fatalMIPRob):
                    outcomesForThisYear[OutcomeType.CARDIOVASCULAR] = {'name': 'MI', 'fatal': True}
                    self._alive.append(False)
                else:
                    outcomesForThisYear[OutcomeType.CARDIOVASCULAR] = {
                        'name': 'MI', 'fatal': False}
                self._mi = 1
            else:
                if self._has_fatal_stroke(fatalStrokeProb):
                    outcomesForThisYear[OutcomeType.CARDIOVASCULAR] = {
                        'name': 'Stroke', 'fatal': True}
                    self._alive.append(False)
                else:
                    outcomesForThisYear[OutcomeType.CARDIOVASCULAR] = {
                        'name': 'Stroke', 'fatal': False}
                self._stroke = 1

        # TODO: needs to be changed to represent NON cardiovascular mortality only
        if (not self.is_dead()):
            mortalityModel = outcome_model_repository.select_model_for_person(
                self, OutcomeType.MORTALITY)
            if (npRand.uniform(size=1) < mortalityModel.estimate_next_risk(self)):
                self._alive.append(False)

    def __repr__(self):
        return (f"Person(age={self._age[-1]}, "
                f"gender={self._gender}, "
                f"race/eth={self._raceEthnicity}, "
                f"sbp={self._sbp[-1]}, "
                f"dbp={self._dbp[-1]}, "
                f"a1c={self._a1c[-1]}, "
                f"hdl={self._hdl[-1]}, "
                f"totChol={self._totChol[-1]}, "
                f"bmi={self._bmi[-1]}, "
                f"ldl={self._ldl[-1]}, "
                f"trig={self._trig[-1]}, "
                f"smoking={self._smokingStatus}"
                f")")
