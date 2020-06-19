import math
import copy
import numpy.random as npRand

from typing import Callable

from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.outcome import Outcome, OutcomeType
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.gcp_model import GCPModel

# luciana-tag...lne thing that tripped me up was probable non clear communication regarding "waves"
# so, i'm going to spell it out here and try to make the code consistent.
# a patient starts in teh simulation prior to a wave with their baseline attribute statuses(i.e subscript [0])
# wave "1" refers to the transition from subscript[0] to subscript[1]
# wave "2" the transition from subscript[1] to subscript[2]
# thus, the pateint's status at the start of wave 1 is represented by subscript[0]
# and the patient status at the end of wave 1 is represtened by subscript[1]
# if a patient has an event during a wave, that means they will not have the status at the start of the wave
# and they will have the status at the end of the wave.
# so, if a patient has an event during wave 1, their status would be Negatve at subscript[0] and
# Positive at subscript[1]


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
        alcohol: AlcoholCategory,
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
        self._alcoholPerWeek = [alcohol]
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
        self._outcomes = {OutcomeType.MI: [], OutcomeType.STROKE: [], OutcomeType.DEMENTIA: []}
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
        else:
            self._afib = [False]

        self._gcp = [GCPModel().get_risk_for_person(self)]
        # for outcome mocels that require random effects, store in this dictionary
        self._randomEffects = dict()

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
        self._alcoholPerWeek = [self._alcoholPerWeek[0]]
        self._statin = [self._statin[0]]
        self._otherLipidLoweringMedicationCount = [self._otherLipidLoweringMedicationCount[0]]
        self._bpTreatmentStrategy = None
        self._gcp = [self._gcp[0]]

        # iterate through outcomes and remove those that occured after the simulation started
        for type, outcomes_for_type in self._outcomes.items():
            self._outcomes[type] = list(
                filter(lambda outcome: outcome[0] < self._age[0], outcomes_for_type))

    @property
    def _current_smoker(self):
        return self._smokingStatus == SmokingStatus.CURRENT

    @property
    def _current_bp_treatment(self):
        return self._antiHypertensiveCount[-1] > 0

    @property
    def _current_diabetes(self):
        return self.has_diabetes()

    @property
    def _mi(self):
        return len(self._outcomes[OutcomeType.MI]) > 0

    @property
    def _stroke(self):
        return len(self._outcomes[OutcomeType.STROKE]) > 0

    @property
    def _dementia(self):
        return len(self._outcomes[OutcomeType.DEMENTIA]) > 0

    @property
    def _black(self):
        return self._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK

    def get_median_age(self):
        medianYear = math.floor(len(self._age) / 2)
        return self._age[medianYear]

    def allhat_candidate(self, end_of_wave_num):
        return (self._age[end_of_wave_num] > 55) and \
            (self._sbp[end_of_wave_num > 140 and self._sbp[end_of_wave_num] < 180]) and \
            (self._dbp[end_of_wave_num] > 90 and self._dbp[end_of_wave_num] < 110) and \
            (self._smokingStatus == SmokingStatus.CURRENT or self._a1c[end_of_wave_num] > 6.5 or
             self.has_stroke_prior_to_simulation() or self.has_mi_prior_to_simulation or
             self._hdl[end_of_wave_num] < 35)

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

        # initialize random effects if they haven't already been initialized and this is our first year
        if self.years_in_simulation() == 0 and len(self._randomEffects) == 0:
            self._randomEffects = outcome_model_repository.get_random_effects()

        self.advance_risk_factors(risk_model_repository)
        self.advance_treatment(risk_model_repository)
        self.advance_outcomes(outcome_model_repository)
        if not self.is_dead():
            self._age.append(self._age[-1] + 1)
            self._alive.append(True)

    def is_dead(self):
        return not self._alive[-1]

    # this method is trying to enable simple logic in the popuation.
    # when the population asks, "who is alive at a given time point?" it can't merely check
    # the index on person._alive, because people who died prior to that time will not have an index
    # in alive at that time.

    def alive_at_start_of_wave(self, start_wave_num):
        if (self._alive[-1]) and (start_wave_num > (len(self._age))):
            raise Exception(
                f"Trying to find status for a wave: {start_wave_num} beyond current wave: {len(self._age)}")

        # we always know, regardless of what wave is being inquired about, that a person who was once dead
        # is still dead
        if (self.is_dead()) and (start_wave_num > len(self._alive) - 1):
            return False
        else:
            # this returns whether one was alive at the start of a given wave (i.e. the end of theprior wave)
            return self._alive[start_wave_num-1]

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
        return self.has_outcome_during_wave(wave, OutcomeType.STROKE)

    def has_mi_during_wave(self, wave):
        return self.has_outcome_during_wave(wave, OutcomeType.MI)

    def has_outcome_during_wave(self, wave, outcomeType):
        if (wave <= 0) or (self._alive[-1] and wave > len(self._age)-1):
            raise Exception(
                f"Can not have an event in a wave ({wave}) before 1 or after last wave ({len(self._age)-1}) for person")
        elif (not self._alive[-1]) and (wave > len(self._age)):
            return False
        return (len(self._outcomes[outcomeType]) != 0 and
                self.has_outcome_at_age(outcomeType, self._age[wave-1]))

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
        # if the patient died during the wave, then their age didn't advance and their event would be at their
        # age at teh start of the wave.
        rollbackAge = self._age[-1]-1 if self._alive[-1] else self._age[-1]
        if rollbackAge != outcome_rolled_back[0]:
            raise Exception(
                f'# of outcomes: {len(outcomes_for_type)} while trying to rollback event at age {outcome_rolled_back[0]}, but current age is {rollbackAge} - can not roll back if age has changed')

        # and, if it was fatal, reset the person to being alive.
        if (outcome_rolled_back)[1].fatal:
            self._alive[-1] = True
            self._age.append(self._age[-1]+1)

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
        self._alcoholPerWeek.append(AlcoholCategory.get_category_for_consumption(self.get_next_risk_factor(
            "alcoholPerWeek", risk_model_repository)))

    # redraw from models to pick new risk factors for person

    def slightly_randomly_modify_baseline_risk_factors(self, risk_model_repository):
        if (len(self._age) > 1):
            raise RuntimeError("Can not reset risk factors after advancing person in time")

        return Person(age=self._age[0] + npRand.randint(-2, 2),
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
                      alcohol=self._alcoholPerWeek[0],
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

        # then assign gcp and dementia...
        self._gcp.append(outcome_model_repository.get_gcp(self))

        # dementia is conceptualized as a progressive process rather than an event you only "get" it onceexit
        if (not self._dementia):
            dementia = outcome_model_repository.get_dementia(self)
            if (dementia is not None):
                self.add_outcome_event(dementia)

        # if not dead from the CV event...assess non CV mortality
        if (not self.is_dead()):
            non_cv_death = outcome_model_repository.assign_non_cv_mortality(self)
            if (non_cv_death):
                self._alive.append(False)

    def add_outcome_event(self, cv_event):
        self._outcomes[cv_event.type].append((self._age[-1], cv_event))
        if cv_event.fatal:
            self._alive.append(False)

    # Using this paper...glucose and a1c are highly related
    # Nathan, D. M., Kuenen, J., Borg, R., Zheng, H., Schoenfeld, D., Heine, R. J., for the A1c-Derived Average Glucose (ADAG) Study Group. (2008). Translating the A1C Assay Into Estimated Average Glucose Values. Diabetes Care, 31(8), 1473–1478.
    # so, will use their formula + a draw from residual distribution fo same moddel in NHANES (which has very simnilar coefficients)

    @staticmethod
    def convert_fasting_glucose_to_a1c(glucose):
        return (glucose + 46.7)/28.7

    @staticmethod
    def convert_a1c_to_fasting_glucose(a1c):
        return 28.7 * a1c - 46.7

    def get_fasting_glucose(self, use_residual=True):
        glucose = Person.convert_a1c_to_fasting_glucose(self._a1c[-1])
        if use_residual:
            glucose += npRand.normal(0, 21)
        return glucose

    def __repr__(self):
        return (f"Person(age={self._age[-1]}, "
                f"gender={self._gender}, "
                f"race/eth={self._raceEthnicity}, "
                f"sbp={self._sbp[-1]:.1f}, "
                f"dbp={self._dbp[-1]:.1f}, "
                f"a1c={self._a1c[-1]:.1f}, "
                f"hdl={self._hdl[-1]:.1f}, "
                f"totChol={self._totChol[-1]:.1f}, "
                f"bmi={self._bmi[-1]:.1f}, "
                f"ldl={self._ldl[-1]:.1f}, "
                f"trig={self._trig[-1]:.1f}, "
                f"smoking={SmokingStatus(self._smokingStatus)}, "
                f"waist={self._waist[-1]}, "
                f"anyPhysicalActivity={self._anyPhysicalActivity[-1]}, "
                f"alcohol={AlcoholCategory(self._alcoholPerWeek[-1])}, "
                f"education={Education(self._education)}, "
                f"antiHypertensiveCount={self._antiHypertensiveCount[-1]}, "
                f"otherLipid={self._otherLipidLoweringMedicationCount[-1]}, "
                f"statin={self._statin[-1]}"
                f")")

    def __ne__(self, obj):
        return not self == obj

    # luciana tag...the nice part about this method is that its highly transparent
    # the not so nice part is that if we add an attribute you have to add it here...
    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        if not other._age == self._age:
            return False
        if not other._gender == self._gender:
            return False
        if not other._raceEthnicity == self._raceEthnicity:
            return False
        if not other._sbp == self._sbp:
            return False
        if not other._dbp == self._dbp:
            return False
        if not other._a1c == self._a1c:
            return False
        if not other._hdl == self._hdl:
            return False
        if not other._totChol == self._totChol:
            return False
        if not other._bmi == self._bmi:
            return False
        if not other._ldl == self._ldl:
            return False
        if not other._trig == self._trig:
            return False
        if not other._waist == self._waist:
            return False
        if not other._anyPhysicalActivity == self._anyPhysicalActivity:
            return False
        if not other._education == self._education:
            return False
        if not other._smokingStatus == self._smokingStatus:
            return False
        if not other._alcoholPerWeek == self._alcoholPerWeek:
            return False
        if not other._antiHypertensiveCount == self._antiHypertensiveCount:
            return False
        if not other._statin == self._statin:
            return False
        if not other._otherLipidLoweringMedicationCount == self._otherLipidLoweringMedicationCount:
            return False
        if not other._afib == self._afib:
            return False
        if not other._alive == self._alive:
            return False
        if not other._gcp == self._gcp:
            return False
        if not other._randomEffects == self._randomEffects:
            return False
        return other._outcomes == self._outcomes

    # luciana tag...there is almost definitely a better way to do this..
    def __deepcopy__(self, memo):
        selfCopy = Person(age=0, gender=None, raceEthnicity=None, sbp=0, dbp=0, a1c=0, hdl=0, totChol=0,
                          bmi=0, ldl=0, trig=0, waist=0, anyPhysicalActivity=0, education=None,
                          smokingStatus=None, alcohol=None, antiHypertensiveCount=0, statin=0, otherLipidLoweringMedicationCount=0,
                          initializeAfib=None)
        selfCopy._lowerBounds = self._lowerBounds
        selfCopy._upperBounds = self._upperBounds
        selfCopy._gender = copy.deepcopy(self._gender)
        selfCopy._raceEthnicity = copy.deepcopy(self._raceEthnicity)
        selfCopy._alive = copy.deepcopy(self._alive)
        selfCopy._age = copy.deepcopy(self._age)
        selfCopy._sbp = copy.deepcopy(self._sbp)
        selfCopy._dbp = copy.deepcopy(self._dbp)
        selfCopy._a1c = copy.deepcopy(self._a1c)
        selfCopy._hdl = copy.deepcopy(self._hdl)
        selfCopy._ldl = copy.deepcopy(self._ldl)
        selfCopy._trig = copy.deepcopy(self._trig)
        selfCopy._totChol = copy.deepcopy(self._totChol)
        selfCopy._waist = copy.deepcopy(self._waist)
        selfCopy._bmi = copy.deepcopy(self._bmi)
        selfCopy._anyPhysicalActivity = copy.deepcopy(self._anyPhysicalActivity)
        selfCopy._education = copy.deepcopy(self._education)
        selfCopy._smokingStatus = copy.deepcopy(self._smokingStatus)
        selfCopy._alcoholPerWeek = copy.deepcopy(self._alcoholPerWeek)
        selfCopy._antiHypertensiveCount = copy.deepcopy(self._antiHypertensiveCount)
        selfCopy._statin = copy.deepcopy(self._statin)
        selfCopy._otherLipidLoweringMedicationCount = copy.deepcopy(
            self._otherLipidLoweringMedicationCount)
        selfCopy._outcomes = copy.deepcopy(self._outcomes)
        selfCopy._selfReportStrokePriorToSim = copy.deepcopy(self._selfReportStrokePriorToSim)
        selfCopy._selfReportMIPriorToSim = copy.deepcopy(self._selfReportMIPriorToSim)
        selfCopy._afib = self._afib
        selfCopy._bpTreatmentStrategy = self._bpTreatmentStrategy
        selfCopy._gcp = copy.deepcopy(self._gcp)
        selfCopy._randomEffects = copy.deepcopy(self._randomEffects)

        return selfCopy
