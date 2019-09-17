import math
import random

from typing import Callable

from mcm.education import Education
from mcm.gender import NHANESGender
from mcm.outcome import Outcome, OutcomeType
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus


class Person:
    """Person is using risk factors and demographics based off NHANES"""

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
        waist: int,    # Waist circumference in cm
        anyPhysicalActivity: int,
        education: Education,
        smokingStatus: SmokingStatus,
        antiHypertensiveCount: int,
        statin: int,
        otherLipidLoweringMedicationCount: int,
        initializeAfib: Callable,
        selfReportStrokeAge=None,
        selfReportMIAge=None,
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
        self._waist = [waist]
        self._anyPhysicalActivity = [anyPhysicalActivity]
        self._education = education
        # TODO : change smoking status into a factor that changes over time
        self._smokingStatus = smokingStatus
        self._antiHypertensiveCount = [antiHypertensiveCount]
        self._statin = [statin]
        self._otherLipidLoweringMedicationCount = [otherLipidLoweringMedicationCount]

        # outcomes is a dictionary of arrays. each element in the dictionary represents
        # a differnet outcome type each element in the array is a tuple representting
        # the age of the patient at the time of an event (element zero). and the outcome
        # (element one).multiple events can be accounted for by having multiple
        # elements in the array.
        self._outcomes = {OutcomeType.MI: [], OutcomeType.STROKE: []}
        self._selfReportStrokePriorToSim = 0
        self._selfReportMIPriorToSim = 0

        # convert events for events prior to simulation
        if selfReportStrokeAge is not None and selfReportStrokeAge > 1:
            self._selfReportStrokePriorToSim = 1
            self._outcomes[OutcomeType.STROKE].append((-1, Outcome(OutcomeType.STROKE, False)))
        if selfReportMIAge is not None and selfReportMIAge > 1:
            self._selfReportMIPriorToSim = 1
            self._outcomes[OutcomeType.MI].append((-1, Outcome(OutcomeType.MI, False)))
        for k, v in kwargs.items():
            setattr(self, k, v)
        if initializeAfib is not None:
            self._afib = [initializeAfib(self)]

        self._bpTreatmentStrategy = None

    def reset_to_baseline(self):
        self._alive = [True]
        self._age = [self._age[0]]
        self._sbp = [self._sbp[0]]
        self._dbp = [self._dbp[0]]
        self._a1c = [self._a1c[0]]
        self._hdl = [self._hdl[0]]
        self._ldl = [self._ldl[0]]
        self._trig = [self._trig[0]]
        self._totChol = [self._totChol[0]]
        self._bmi = [self._bmi[0]]
        self._waist = [self._waist[0]]
        self._anyPhysicalActivity = [self._anyPhysicalActivity[0]]
        self._antiHypertensiveCount = [self._antiHypertensiveCount[0]]
        self._statin = [self._statin[0]]
        self._otherLipidLoweringMedicationCount = [self._otherLipidLoweringMedicationCount[0]]
        self._bpTreatmentStrategy = None

        # iterate through outcomes and remove those that occured after the simulation started
        for type, outcomes_for_type in self._outcomes.items():
            self._outcomes[type] = list(filter(lambda outcome: outcome[0] < self._age[0], outcomes_for_type))

    @property
    def _mi(self):
        return len(self._outcomes[OutcomeType.MI]) > 0

    @property
    def _stroke(self):
        return len(self._outcomes[OutcomeType.STROKE]) > 0

    def get_median_age(self):
        medianYear = math.floor(len(self._age) / 2)
        return self._age[medianYear]

    def allhat_candidate(self, wave):
        return (self._age[wave] > 55) and \
            (self._sbp[wave > 140 and self._sbp[wave] < 180]) and \
            (self._dbp[wave] > 90 and self._dbp[wave] < 110) and \
            (self._smokingStatus == SmokingStatus.CURRENT or self._a1c[wave] > 6.5 or
             self.has_stroke_prior_to_simulation() or self.has_mi_prior_to_simulation or
             self._hdl[wave] < 35)

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
        # print(f"advance_year on person, age: {self._age[0]} sbp : {self._sbp[0]}")
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance year")

        self.advance_risk_factors(risk_model_repository)
        self.advance_treatment(risk_model_repository)
        self.advance_outcomes(outcome_model_repository)
        if not self.is_dead():
            self._age.append(self._age[-1] + 1)

    def is_dead(self):
        return not self._alive[-1]

    def currently_alive(self, currentYear):
        return self._alive[-1] or self._alive[currentYear]

    def has_outcome_prior_to_simulation(self, outcomeType):
        return any([ageAtEvent < 0 for ageAtEvent, _ in self._outcomes[outcomeType]])

    def has_outcome_during_simulation(self, outcomeType):
        return any([ageAtEvent >= 0 for ageAtEvent, _ in self._outcomes[outcomeType]])

    def has_outcome_at_any_time(self, outcomeType):
        return len(self._outcomes[outcomeType]) > 0

    def has_stroke_prior_to_simulation(self):
        return self.has_outcome_prior_to_simulation(OutcomeType.STROKE)

    def has_stroke_during_simulation(self):
        return self.has_outcome_during_simulation(OutcomeType.STROKE)

    def has_stroke_during_wave(self, wave):
        return (len(self._age) > wave and
                len(self._outcomes[OutcomeType.STROKE]) != 0 and
                self.has_outcome_at_age(OutcomeType.STROKE, self._age[wave]))

    def has_mi_during_wave(self, wave):
        return (len(self._age) > wave and
                len(self._outcomes[OutcomeType.MI]) != 0 and
                self.has_outcome_at_age(OutcomeType.MI, self._age[wave]))

    def has_outcome_at_age(self, type, age):
        for outcome_tuple in self._outcomes[type]:
            if outcome_tuple[0] == age:
                return True
        return False

    def has_fatal_stroke(self):
        return any([stroke.fatal for _, stroke in self._outcomes[OutcomeType.STROKE]])

    def has_fatal_mi(self):
        return any([mi.fatal for _, mi in self._outcomes[OutcomeType.MI]])

    def has_mi_prior_to_simulation(self):
        return self.has_outcome_prior_to_simulation(OutcomeType.MI)

    def has_mi_during_simulation(self):
        return self.has_outcome_during_simulation(OutcomeType.MI)

    # should only occur immediately after an event is created — we can't roll back the subsequent implicaitons of an event.
    def rollback_most_recent_event(self, outcomeType):
        # get rid of the outcome event...
        outcomes_for_type = list(self._outcomes[outcomeType])
        outcome_rolled_back = self._outcomes[outcomeType].pop()
        if self._age[-1]-1 != outcome_rolled_back[0]:
            print(self)
            print(self._age)
            print(outcomes_for_type)
            raise Exception(
                f'# of outcomes: {len(outcomes_for_type)} while trying to rollback event at age {outcome_rolled_back[0]}, but current age is {self._age[-1]-1 } - can not roll back if age has changed')

        # and, if it was fatal, reset the person to being alive.
        if (outcome_rolled_back)[1].fatal:
            self.alive[-1] = True

    def advance_treatment(self, risk_model_repository):
        if (risk_model_repository is not None):
            new_antihypertensive_count = self.get_next_risk_factor(
                "antiHypertensiveCount",
                risk_model_repository
            )
            self._antiHypertensiveCount.append(new_antihypertensive_count)

        if self._bpTreatmentStrategy is not None:
            treatment_modifications, risk_factor_modifications, recalibration_standards = self._bpTreatmentStrategy(
                self)
            self.apply_linear_modifications(treatment_modifications)
            self.apply_linear_modifications(risk_factor_modifications)
            # simple starting assumption...a treatment is applied once and has a persistent effect
            # so, the treastment strategy is nulled out after being applied
            self._bpTreatmentStrategy = None

    def apply_linear_modifications(self, modifications):
        for key, value in modifications.items():
            attribute_value = getattr(self, key)
            attribute_value[-1] = attribute_value[-1] + value

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
        self._waist.append(self.get_next_risk_factor("waist", risk_model_repository))
        self._anyPhysicalActivity.append(
            self.get_next_risk_factor(
                "anyPhysicalActivity",
                risk_model_repository))
        self._afib.append(self.get_next_risk_factor("afib", risk_model_repository))
        self._statin.append(self.get_next_risk_factor("statin", risk_model_repository))

    # redraw from models to pick new risk factors for person

    def slightly_randomly_modify_baseline_risk_factors(self, risk_model_repository):
        if (len(self._age) > 1):
            raise RuntimeError("Can not reset risk factors after advancing person in time")

        return Person(age=self._age[0] + random.randint(-2, 2),
                      gender=self._gender,
                      raceEthnicity=self._raceEthnicity,
                      sbp=self.get_next_risk_factor("sbp", risk_model_repository),
                      dbp=self.get_next_risk_factor("dbp", risk_model_repository),
                      a1c=self.get_next_risk_factor("a1c", risk_model_repository),
                      hdl=self.get_next_risk_factor("hdl", risk_model_repository),
                      totChol=self.get_next_risk_factor("totChol", risk_model_repository),
                      bmi=self.get_next_risk_factor("bmi", risk_model_repository),
                      ldl=self.get_next_risk_factor("ldl", risk_model_repository),
                      trig=self.get_next_risk_factor("trig", risk_model_repository),
                      waist=self.get_next_risk_factor("waist", risk_model_repository),
                      anyPhysicalActivity=self.get_next_risk_factor(
                          "anyPhysicalActivity", risk_model_repository),
                      education=self._education,
                      smokingStatus=self._smokingStatus,
                      antiHypertensiveCount=self.get_next_risk_factor(
                          "antiHypertensiveCount", risk_model_repository),
                      statin=self.get_next_risk_factor("statin", risk_model_repository),
                      otherLipidLoweringMedicationCount=self._otherLipidLoweringMedicationCount,
                      initializeAfib=(lambda _: False),
                      selfReportStrokeAge=50 if self._outcomes[OutcomeType.STROKE] is not None else None,
                      selfReportMIAge=50 if self._outcomes[OutcomeType.MI] is not None else None)

    def advance_outcomes(
            self,
            outcome_model_repository):
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance outcomes")

        # first determine if there is a cv event
        cv_event = outcome_model_repository.assign_cv_outcome(self)
        if cv_event is not None:
            self.add_outcome_event(cv_event)

        # if not dead from the CV event...assess non CV mortality
        if (not self.is_dead()):
            non_cv_death = outcome_model_repository.assign_non_cv_mortality(self)
            if (non_cv_death):
                self._alive.append(False)

    
    def add_outcome_event(self, cv_event):
        self._outcomes[cv_event.type].append((self._age[-1], cv_event))
        if cv_event.fatal:
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
