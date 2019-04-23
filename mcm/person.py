from mcm.gender import NHANESGender
from mcm.outcome_model_type import OutcomeModelType
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.outcome import OutcomeType
from mcm.outcome import Outcome
from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from mcm.regression_model import RegressionModel

import numpy.random as npRand
import scipy.special as scipySpecial
import os
import json


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

        # TODO: need to initialized with whether somebody had a prior stroke or MI
        # for prior events, ages should be initialized as -1 (meaning prior to simulation)

        # outcomes is a dictionary of arrays. each element in the dictionary represents
        # a differnet outcome type each element in the array is a tuple representting
        # the age of the patient at the time of an event (element zero). and the outcome
        # (element one).multiple events can be accounted for by having multiple
        # elements in the array.
        self._outcomes = {OutcomeType.MI: [], OutcomeType.STROKE: []}
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def _mi(self):
        return len(self._outcomes[OutcomeType.MI]) > 0

    @property
    def _stroke(self):
        return len(self._outcomes[OutcomeType.STROKE]) > 0

    def has_diabetes(self):
        return sorted(self._a1c)[-1] >= 6.5

    def years_in_simulation(self):
        return len(self._age) - 1

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

    def has_stroke_prior_to_simulation(self):
        return any([ageAtEvent < 0 for ageAtEvent, _ in self._outcomes[OutcomeType.STROKE]])

    def has_stroke_during_simulation(self):
        return any([ageAtEvent >= 0 for ageAtEvent, _ in self._outcomes[OutcomeType.STROKE]])

    def has_fatal_stroke(self):
        return any([stroke.fatal for _, stroke in self._outcomes[OutcomeType.STROKE]])

    def has_fatal_mi(self):
        return any([mi.fatal for _, mi in self._outcomes[OutcomeType.MI]])

    def has_mi_prior_to_simulation(self):
        return any([ageAtEvent < 0 for ageAtEvent, _ in self._outcomes[OutcomeType.MI]])

    def has_mi_during_simulation(self):
        return any([ageAtEvent >= 0 for ageAtEvent, _ in self._outcomes[OutcomeType.MI]])

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

    def _will_have_cvd_event(self, ascvdProb):
        return npRand.uniform(size=1) < ascvdProb

    def _will_have_mi(self, manualMIProb=None):
        if manualMIProb is not None:
            return manualMIProb
        # if no manual MI probabiliyt, estimate it from oru partitioned model
        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        model_spec_path = os.path.normpath(os.path.join(abs_module_path,
                                                        "./data/StrokeMIPartitionModelSpec.json"))
        with open(model_spec_path, 'r') as model_spec_file:
            model_spec = json.load(model_spec_file)
        strokePartitionModel = StatsModelLinearRiskFactorModel(RegressionModel(**model_spec))
        strokeProbability = scipySpecial.expit(strokePartitionModel.estimate_next_risk(self))

        return npRand.uniform(size=1) < (1 - strokeProbability)

    def _will_have_fatal_mi(self, fatalMIProb):
        return npRand.uniform(size=1) < fatalMIProb

    def _will_have_fatal_stroke(self, fatalStrokeProb):
        return npRand.uniform(size=1) < fatalStrokeProb

    # fatal stroke probability estimated from our meta-analysis of BASIC, NoMAS, GCNKSS, REGARDS
    # fatal mi probability from: Wadhera, R. K., Joynt Maddox, K. E., Wang, Y., Shen, C., Bhatt,
    # D. L., & Yeh, R. W.
    # (2018). Association Between 30-Day Episode Payments and Acute Myocardial Infarction Outcomes
    # Among Medicare
    # Beneficiaries. Circ. Cardiovasc. Qual. Outcomes, 11(3), e46–9.
    # http://doi.org/10.1161/CIRCOUTCOMES.117.004397
    def advance_outcomes(
            self,
            outcome_model_repository,
            manualStrokeMIProbability=None,
            fatalMIPRob=0.13,
            fatalStrokeProb=0.15):
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance outcomes")

        if self._will_have_cvd_event(
            outcome_model_repository.get_risk_for_person(
                self,
                OutcomeModelType.CARDIOVASCULAR,
                years=1)):
            if self._will_have_mi(manualStrokeMIProbability):
                if self._will_have_fatal_mi(fatalMIPRob):
                    self._outcomes[OutcomeType.MI].append(
                        (self._age[-1], Outcome(OutcomeType.MI, True)))
                    self._alive.append(False)
                else:
                    self._outcomes[OutcomeType.MI].append(
                        (self._age[-1], Outcome(OutcomeType.MI, False)))
            else:
                if self._will_have_fatal_stroke(fatalStrokeProb):
                    self._outcomes[OutcomeType.STROKE].append(
                        (self._age[-1], Outcome(OutcomeType.STROKE, True)))
                    self._alive.append(False)
                else:
                    self._outcomes[OutcomeType.STROKE].append(
                        (self._age[-1], Outcome(OutcomeType.STROKE, False)))

        if (not self.is_dead()):
            riskForPerson = outcome_model_repository.get_risk_for_person(
                self,
                OutcomeModelType.MORTALITY)

            if (npRand.uniform(size=1) < riskForPerson):
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
