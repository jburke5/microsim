import math
import copy
import numpy.random as npRand
import numpy as np
import pandas as pd
import logging

from typing import Callable

from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.outcome import Outcome, OutcomeType
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.gfr_equation import GFREquation
from microsim.pvd_model import PVDPrevalenceModel
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.treatment import TreatmentStrategiesType, TreatmentStrategyStatus

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
    """Person is using risk factors and demographics based off NHANES.
       A Person-instance is essentially a data structure that holds all person-related data, the past and the present.
       The Person class includes functions for essentially two things:
          1) How to predict a Person-instance's future, when the models for making the predictions are provided.
             The predictive models are not stored in Person-instances but only provided as arguments to the Person functions.
          2) Tools for analyzing and reporting a Person-instance's state.
       In order to initialize a Person-instance, the risk factors, default treatments, treatment strategies and outcomes need to be 
       provided to the class in an organized way, in their corresponding dictionaries.
       Default treatments correspond to the usual care, whereas treatment strategies correspond to approaches we would like to
       try and discover their effect.
       _name: indicates the origin of the Person's instance data, eg in NHANES it will be the NHANES person unique identifier, 
              more than one Person instances can have the same name.
       _index: a unique identifier when the Person-instance is part of a bigger group, eg a Population instance (this is set from the Population instance).
       _waveCompleted: every time a complete advanced has been performed, increase this by 1, first complete advanced corresponds to 0.
                       A complete advanced = risk factors, treatment, treatment strategies, updated risk factors, outcomes.
       _outcomes: a dictionary of arrays with the keys being OutcomeTypes, each element in the array is a tuple (age, outcome).
                  Multiple events can be accounted for by having multiple elements in the array.
       _randomEffects: some outcome models require random effects, store them in this dictionary, the outcome models set their key:value."""

    def __init__(self, 
                 name, 
                 staticRiskFactorsDict, 
                 dynamicRiskFactorsDict, 
                 defaultTreatmentsDict, 
                 treatmentStrategiesDict, 
                 outcomesDict) -> None:

        self._name = name
        self._index = None
        self._waveCompleted = -1    
        self._randomEffects = dict()
        #for now, I assume that the OS-derived entropy will be different for each person instance even when mp is used
        self._rng = np.random.default_rng()

        #will it be better if static, dynamic RiskFactors and treatments were attributes-dictionaries like the outcomes?
        #will it double the attribute access time by having to find 2 pointers as opposed to 1? how significant will that be?
        #self._staticRiskFactors = staticRiskFactorsDict
        #self._dynamicRiskFactors = dynamicRiskFactorsDict
        #self._defaultTreatments = defaultTreatmentsDict
        #self._treatmentStrategies = treatmentStrategiesDict

        #also, there is currently an inconsistency: outcomes are provided ready to the Person instance but everything 
        #else is not, eg the lists are created here and not in the build_person method, if all were dictionaries this would be resolved
        #an attempt on this showed that there are deep dependencies on Person attributes, 
        #eg StatsModelLinearRiskFactorModel.get_model_argument_for_coeff_name expects to finds these attributes directly on Person instances
        #for now I will keep lists of the static, dynamic risk factors etc so that I know how to advance each person
        #even though it is not ideal for memory purposes all Person instances to have exactly the same lists...

        for key,value in staticRiskFactorsDict.items():
            setattr(self, "_"+key, value)
        self._staticRiskFactors = list(staticRiskFactorsDict.keys())
        for key,value in dynamicRiskFactorsDict.items():
            setattr(self, "_"+key, [value])
        self._dynamicRiskFactors = list(dynamicRiskFactorsDict.keys())

        for key,value in defaultTreatmentsDict.items():
            setattr(self, "_"+key, [value])
        self._defaultTreatments = list(defaultTreatmentsDict.keys())

        #for key, value in treatmentStrategiesDict.items():
        #    setattr(self, "_"+key+"TreatmentStatus", value)
        #self._treatmentStrategies = list(treatmentStrategiesDict.keys())
        self._treatmentStrategies = treatmentStrategiesDict

        self._outcomes = outcomesDict

    def advance(self, years, dynamicRiskFactorRepository, defaultTreatmentRepository, outcomeModelRepository, treatmentStrategies=None):
        """years: for how many years we want to make predictions for.
           dynamicRiskFactorRepository, defaultTreatmentRepository, outcomeModelRepository, treatmentStrategies: the rules/models
                  that predict the state of the Person-instance in the future."""
        
        #Q: how we do the first advance right after the initialization of an instance depends on the structure of the instance
        #   which depends in turn on the database we used to do the initialization
        #   eg, NHANES has risk factor information (incomplete but most of it at least) and default treatment information
        #   but no treatment strategy information and no outcome information for the first year
        #   so if we imagine Person instances created from different databases, we can 
        #   1) either modify each build_person method to create Person-instances with a complete advance cycle (call advance treatment strategy
        #      and advance outcomes)
        #   2) have the Person class do all of this work
        # For now, I have modified this advance function to accomodate the NHANES Person-instances but this will need to be modified
        # with either 1 or 2 above if we are to work with more than just NHANES Person instances

        for yearIndex in range(years):
            if self.is_alive:
                #choice of words: advance=append, update=modify last quantity in place
                if self._waveCompleted > -1:
                    self.advance_risk_factors(dynamicRiskFactorRepository)
                    self.advance_treatments(defaultTreatmentRepository)
                self.advance_treatment_strategies_and_update_risk_factors(treatmentStrategies)
                self.advance_outcomes(outcomeModelRepository)
                #finished one more complete advance 
                self._waveCompleted += 1

    def advance_risk_factors(self, rfdRepository):
        for rf in self._dynamicRiskFactors:
            nextRiskFactor = rfdRepository.apply_bounds(rf, self.get_next_risk_factor(rf, rfdRepository))
            setattr(self, "_"+rf, getattr(self,"_"+rf)+[nextRiskFactor]) 

    # Q: it is not clear to me why treatment strategies affect the person attributes directly
    #whereas treatments affect the person attributes indirectly through the attribute regression models
    #will it always be like that? keep in mind that the regression models were designed to be 1 year based predictions
    #the assumption is that the effect of the treatment strategies is instantaneous but
    #there is nothing preventing us from using a regression model as the effect of a treatment strategy
    #also, notice that dynamic risk factors and treatments are lists that get their next quantity in the same way
    def advance_treatments(self, defaultTreatmentRepository):
        for treatment in self._defaultTreatments:
            setattr(self, "_"+treatment, getattr(self,"_"+treatment)+[self.get_next_treatment(treatment, defaultTreatmentRepository)]) 

    def advance_treatment_strategies_and_update_risk_factors(self, treatmentStrategies=None):      
        #choice of words: get_next returns the final/next wave quantity, update modifies that quantity in place
        for tsType in TreatmentStrategiesType:
            ts = treatmentStrategies._repository[tsType.value] if treatmentStrategies is not None else None
            #treatment status must be updated even when there is no treatment strategy for the year
            self.update_treatment_strategy_status(ts, tsType)
            #make it explicit that treatment strategies update the treatments and risk factors
            if ts is not None:
                self.update_treatments(ts)
                self.update_risk_factors(ts)

    def update_treatments(self, treatmentStrategy):
        updatedTreatments = treatmentStrategy.get_updated_treatments(self)
        for treatment in self._defaultTreatments:
            if treatment in updatedTreatments.keys():
                getattr(self, "_"+treatment)[-1] = updatedTreatments[treatment]

    def update_risk_factors(self, treatmentStrategy):
        updatedRiskFactors = treatmentStrategy.get_updated_risk_factors(self)
        for rf in self._dynamicRiskFactors:
            if rf in updatedRiskFactors.keys():
                getattr(self, "_"+rf)[-1] = updatedRiskFactors[rf]

    def update_treatment_strategy_status(self, treatmentStrategy, treatmentStrategyType):
        if treatmentStrategy is not None:
            if self._treatmentStrategies[treatmentStrategyType.value]["status"] is None:
                if treatmentStrategy.status==TreatmentStrategyStatus.BEGIN:
                    self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.BEGIN
                else:
                    raise RuntimeError(f"{treatmentStrategyType}: Treatment strategy status None can only begin.") 
            elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.BEGIN:
                if treatmentStrategy.status==TreatmentStrategyStatus.MAINTAIN:
                    self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.MAINTAIN
                elif treatmentStrategy.status==TreatmentStrategyStatus.END:
                    self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.END
                else:
                    raise RuntimeError(f"{treatmentStrategyType}: Treatment strategy status begin can either maintain or end.")
            elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.MAINTAIN:
                if treatmentStrategy.status==TreatmentStrategyStatus.MAINTAIN:
                    pass
                elif treatmentStrategy.status==TreatmentStrategyStatus.END:
                    self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.END
                else:
                    raise RuntimeError(f"{treatmentStrategyType}: Treatment strategy status maintain can either maintain or end.")
            elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.END:
                if treatmentStrategy.status==TreatmentStrategyStatus.BEGIN:
                    self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.BEGIN
                else:
                    raise RuntimeError(f"{treatmentStrategyType}: Treatment strategy status end can only begin.")
        else:
            if self._treatmentStrategies[treatmentStrategyType.value]["status"] is None:
                pass
            elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.END:
                self._treatmentStrategies[treatmentStrategyType.value]["status"] = None
            else:
                raise RuntimeError(f"{treatmentStrategyType}: Treatment strategy status None or end are the only ones that can move to status None.")
            
        #if self._treatmentStrategies[treatmentStrategyType.value]["status"] is None:
        #    if treatmentStrategy is not None:
        #        self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.BEGIN
        #elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.BEGIN:
        #    if treatmentStrategy is not None:
        #        self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.MAINTAIN
        #    else:
        #        self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.END
        #elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.MAINTAIN:
        #    if treatmentStrategy is None: 
        #        self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.END 
        #elif self._treatmentStrategies[treatmentStrategyType.value]["status"] == TreatmentStrategyStatus.END:
        #    if treatmentStrategy is None:
        #        self._treatmentStrategies[treatmentStrategyType.value]["status"] = None
        #    else:
        #        self._treatmentStrategies[treatmentStrategyType.value]["status"] = TreatmentStrategyStatus.BEGIN 
        #else:
        #    raise RuntimeError("Unrecognized person treatment strategy status.") 
        #    #setattr(self,"_"+tsType.value+"TreatmentStatus", ts.status) 

    def advance_outcomes(self, outcomeModelRepository):
        for outcomeType in OutcomeType:
            outcome = outcomeModelRepository._repository[outcomeType].select_outcome_model_for_person(self).get_next_outcome(self)
            self.add_outcome(outcome)

    def add_outcome(self, outcome):
        if outcome is not None:
            self._outcomes[outcome.type].append((self._current_age, outcome))

    def has_outcome_at_current_age(self, outcome):
        ageAtLastOutcome = self.get_age_at_last_outcome(outcome)
        if (ageAtLastOutcome is None) | (self._current_age!=ageAtLastOutcome):
            return False
        else:
            return True
    
    def has_fatal_outcome_at_current_age(self, outcome):
        if self.has_outcome_at_current_age(outcome):
            return True if self._outcomes[outcome][-1][1].fatal else False
        else:
            return False
    
    @property
    def is_in_bp_treatment(self):
        return ( (self._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.BEGIN) |
                 (self._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.MAINTAIN) )

    #Q: the term bp seems inconsistent here, maybe change bp to hypertensive?
    @property 
    def _current_bp_treatment(self):
        return self._antiHypertensiveCount[-1] > 0

    @property
    def _current_age(self):
        return self._age[-1]
    
    @property
    def is_alive(self):
        return len(self._outcomes[OutcomeType.DEATH])==0
    @property
    def is_dead(self):
        return not self.is_alive
 
    @property
    def _baselineGcp(self):
        return self._outcomes[OutcomeType.COGNITION][0][1].gcp

    @property
    def _gcpSlope(self):
        if len(self._outcomes[OutcomeType.COGNITION])>=2:
            gcpSlope = ( self._outcomes[OutcomeType.COGNITION][-1][1].gcp -
                         self._outcomes[OutcomeType.COGNITION][-2][1].gcp )
        else:
            gcpSlope = 0
        return gcpSlope
   
    @property
    def _selfReportStrokePriorToSim(self):
        return False if len(self._outcomes[OutcomeType.STROKE])==0 else self._outcomes[OutcomeType.STROKE][0][1].priorToSim
    
    @property
    def _selfReportMIPriorToSim(self):
        return False if len(self._outcomes[OutcomeType.MI])==0 else self._outcomes[OutcomeType.MI][0][1].priorToSim
    
    def get_next_treatment(self, treatment, treatmentRepository):
        model = treatmentRepository.get_model(treatment)
        return model.estimate_next_risk(self)

    def get_gender_age_of_all_outcomes_in_sim(self, outcomeType):
        genderAge = []
        if len(self._outcomes[outcomeType])>0:
            for outcome in self._outcomes[outcomeType]:
                if not outcome[1].priorToSim:
                    genderAge += [(getattr(self, "_"+StaticRiskFactorsType.GENDER.value).value, outcome[0])]
        return genderAge

    def get_gender_age_of_all_years_in_sim(self):
        return [(getattr(self, "_"+StaticRiskFactorsType.GENDER.value).value, age) for age in getattr(self, "_"+DynamicRiskFactorsType.AGE.value)] 

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
        self._qalys = [self._qalys[0]]
        self._afib = [self._afib[0]]
        self._bpMedsAdded = [self._bpMedsAdded[0]]
        self._creatinine = [self._creatinine[0]]

        # iterate through outcomes and remove those that occured after the simulation started
        for type, outcomes_for_type in self._outcomes.items():
            self._outcomes[type] = list(
                filter(lambda outcome: outcome[0] < self._age[0], outcomes_for_type)
            )

    def get_wave_for_age(self, ageTarget):
        if ageTarget < self._age[0] or ageTarget > self._age[-1]:
            raise RuntimeError(f'Age:: {ageTarget} out of range {self._age[0]}-{self._age[-1]}')
        else:
            return self._age.index(ageTarget)

    def get_age_for_wave(self, wave):
        if (wave<0) | (wave>self._waveCompleted):
            raise RuntimeError(f'Wave: {wave} out of range 0-{self._waveCompleted}')
        else:
            return self._age[wave]

    # returns a version of the person that maintains all of their history up until 
    # a specified age threshold.
    def get_person_copy_at_age(self, age):
        personCopy = copy.deepcopy(self)
        waveForAge = personCopy.get_wave_for_age(age)
        personCopy._age = personCopy._age[:waveForAge]
        personCopy._alive = personCopy._alive[:waveForAge]
        personCopy._sbp = personCopy._sbp[:waveForAge]
        personCopy._dbp = personCopy._dbp[:waveForAge]
        personCopy._a1c = personCopy._a1c[:waveForAge]
        personCopy._hdl = personCopy._hdl[:waveForAge]
        personCopy._ldl = personCopy._ldl[:waveForAge]
        personCopy._trig = personCopy._trig[:waveForAge]
        personCopy._totChol = personCopy._totChol[:waveForAge]
        personCopy._bmi = personCopy._bmi[:waveForAge]
        personCopy._waist = personCopy._waist[:waveForAge]
        personCopy._anyPhysicalActivity = personCopy._anyPhysicalActivity[:waveForAge]
        personCopy._antiHypertensiveCount = personCopy._antiHypertensiveCount[:waveForAge]
        personCopy._alcoholPerWeek = personCopy._alcoholPerWeek[:waveForAge]
        personCopy._statin = personCopy._statin[:waveForAge]
        personCopy._otherLipidLoweringMedicationCount = personCopy._otherLipidLoweringMedicationCount[:waveForAge]
        personCopy._gcp = personCopy._gcp[:waveForAge]
        personCopy._qalys = personCopy._qalys[:waveForAge]
        personCopy._afib = personCopy._afib[:waveForAge]
        personCopy._bpMedsAdded = personCopy._bpMedsAdded[:waveForAge]
        personCopy._creatinine = personCopy._creatinine[:waveForAge]
        personCopy._populationIndex = self._populationIndex

        # iterate through outcomes and remove those that occured after the simulation started
        for type, outcomes_for_type in personCopy._outcomes.items():
            personCopy._outcomes[type] = list(
                filter(lambda outcome: outcome[0] < age, outcomes_for_type)
            )
        return personCopy

    def get_person_copy_starting_at_wave(self, wave):
        personCopy = copy.deepcopy(self)
        personCopy._age = personCopy._age[wave:]
        personCopy._alive = personCopy._alive[wave:]
        personCopy._sbp = personCopy._sbp[wave:]
        personCopy._dbp = personCopy._dbp[wave:]
        personCopy._a1c = personCopy._a1c[wave:]
        personCopy._hdl = personCopy._hdl[wave:]
        personCopy._ldl = personCopy._ldl[wave:]
        personCopy._trig = personCopy._trig[wave:]
        personCopy._totChol = personCopy._totChol[wave:]
        personCopy._bmi = personCopy._bmi[wave:]
        personCopy._waist = personCopy._waist[wave:]
        personCopy._anyPhysicalActivity = personCopy._anyPhysicalActivity[wave:]
        personCopy._antiHypertensiveCount = personCopy._antiHypertensiveCount[wave:]
        personCopy._alcoholPerWeek = personCopy._alcoholPerWeek[wave:]
        personCopy._statin = personCopy._statin[wave:]
        personCopy._otherLipidLoweringMedicationCount = personCopy._otherLipidLoweringMedicationCount[wave:]
        personCopy._gcp = personCopy._gcp[wave:]
        personCopy._qalys = personCopy._qalys[wave:]
        personCopy._afib = personCopy._afib[wave:]
        personCopy._bpMedsAdded = personCopy._bpMedsAdded[wave:]
        personCopy._creatinine = personCopy._creatinine[wave:]
        personCopy._populationIndex = self._populationIndex

        ### UNFINISHED TAG — have to figure out what to do here...what do we do with outcomes that happen
        ### prior to this wave..
        
        # iterate through outcomes and remove those that occured after the simulation started
        for type, outcomes_for_type in personCopy._outcomes.items():
            personCopy._outcomes[type] = list(
                filter(lambda outcome: outcome[0] < self.get_age_at_end_of_wave(wave), outcomes_for_type)
            )
        return personCopy

    def get_current_state_as_dict(self):
        """Returns the present, the last wave, state of the Person object (ie, no past information is included)."""
        attributes = dict()
        attributes["name"] = self._name
        attributes["index"] = self._index
        attributes["random_effects"] = self._randomEffects
        for attr in self._staticRiskFactors:
            attributes[attr] = getattr(self,'_'+attr)
        for attr in self._dynamicRiskFactors:
            attributes[attr] = getattr(self,'_'+attr)[-1]
        for attr in self._defaultTreatments:
            attributes[attr] = getattr(self,'_'+attr)[-1]
        attributes["outcomes"] = self._outcomes
        return attributes

    def get_tvc_state_as_dict(self, timeVaryingCovariates):
        tvcAttributes = {}
        for var in timeVaryingCovariates:
            attr = getattr(self, "_" + var)
            for wave in range(0, len(attr)):
                tvcAttributes[var + str(wave)] = attr[wave]
        return tvcAttributes

    def get_tvc_state_as_dict_long(self, timeVaryingCovariates):
        tvcAttributes = {}
        for var in timeVaryingCovariates:
            attr = getattr(self, "_" + var)
            tvcAttributes[var] = attr
        return tvcAttributes

    def get_full_state_as_dict(self): 
        """Includes the complete state, past and present, of the Person object."""
        attributes = dict()
        attributes["name"] = self._name
        attributes["index"] = self._index
        attributes["random_effects"] = self._randomEffects
        for attr in self._staticRiskFactors:
            attributes[attr] = getattr(self,'_'+attr)
        for attr in self._dynamicRiskFactors:
            attributes[attr] = getattr(self,'_'+attr)
        for attr in self._defaultTreatments:
            attributes[attr] = getattr(self,'_'+attr)
        attributes["outcomes"] = self._outcomes
        return attributes

    @property
    def _current_smoker(self):
        return self._smokingStatus == SmokingStatus.CURRENT

    @property
    def _current_diabetes(self):
        return self.has_diabetes()

    # Q: should we make GFR a dynamic risk factor or outcome or leave it as is?
    @property
    def _gfr(self):
        return GFREquation().get_gfr_for_person(self)

    @property
    def _current_ckd(self):
        return self._gfr < 60

    # generlized logistic function mapping GCP to MMSE in combined cohrot data
    def get_current_mmse(self):
        numerator = 30  # ceiling effect
        denominator = (0.9924 + np.exp(-0.0795 * self._gcp[-1])) ** (1 / 0.1786)
        return numerator / denominator

    @property
    def _mi(self):
        return len(self._outcomes[OutcomeType.MI]) > 0

    @property
    def _stroke(self):
        return len(self._outcomes[OutcomeType.STROKE]) > 0

    @property
    def _dementia(self):
        return len(self._outcomes[OutcomeType.DEMENTIA]) > 0

    def has_incident_event(self, outcomeType):
        # luciana-tag..this feels messy there is probably a better way to deal weith this.
        # age is updated after dementia events are set, so "incident demetnia" is dementia as of the last wave
        return (
            (len(self._outcomes[outcomeType]) > 0)
            and (len(self._age) >= 2)
            and (self._outcomes[outcomeType][0][0] == self._age[-2])
        )

    def has_incident_dementia(self):
        return self.has_incident_event(OutcomeType.DEMENTIA)

    @property
    def _black(self):
        return self._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK

    @property
    def _white(self):
        return self._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_WHITE

    def get_median_age(self):
        medianYear = math.floor(len(self._age) / 2)
        return self._age[medianYear]

    def allhat_candidate(self, end_of_wave_num):
        return (
            (self._age[end_of_wave_num] > 55)
            and (self._sbp[end_of_wave_num > 140 and self._sbp[end_of_wave_num] < 180])
            and (self._dbp[end_of_wave_num] > 90 and self._dbp[end_of_wave_num] < 110)
            and (
                self._smokingStatus == SmokingStatus.CURRENT
                or self._a1c[end_of_wave_num] > 6.5
                or self.has_stroke_prior_to_simulation()
                or self.has_mi_prior_to_simulation()
                or self._hdl[end_of_wave_num] < 35
            )
        )

    def has_diabetes(self):
        return sorted(self._a1c)[-1] >= 6.5

    def get_years_in_simulation(self):
        return self._waveCompleted+1

    def get_next_risk_factor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get_model(riskFactor)
        return model.estimate_next_risk(self)

    def get_total_qalys(self):
        return sum(list(map(lambda x: x[1].qaly, self._outcomes[OutcomeType.QUALITYADJUSTED_LIFE_YEARS])))

    def get_qalys_from_wave(self, wave):
        total = 0
        for i in range(wave - 1, len(self._qalys)):
            total += self._qalys[i]
        return total

    def get_death_age(self):
        if len(self._outcomes[OutcomeType.DEATH])>0:
            return self._outcomes[OutcomeType.DEATH][0][0]
        else:
            return None

    def dead_by_wave(self, wave):
        if len(self._outcomes[OutcomeType.DEATH])>0:
            ageAtWave = self.get_age_for_wave(wave)
            return ageAtWave > self.get_death_age()
        else:
            return False

    def alive_at_start_of_wave(self, wave):
        return not self.dead_by_wave(wave)

    def has_outcome_prior_to_simulation(self, outcomeType):
        if len(self._outcomes[outcomeType])>0:
           return any([outcome.priorToSim for _, outcome in self._outcomes[outcomeType]])
        else:
           return False

    def has_outcome_during_simulation(self, outcomeType):
        if len(self._outcomes[outcomeType])>0:
           return any([outcome.priorToSim == False for _, outcome in self._outcomes[outcomeType]])
        else:
           return False

    def get_outcomes_during_simulation(self, outcomeType):
        return list(filter(lambda x: not x[1].priorToSim, self._outcomes[outcomeType]))

    def has_outcome_during_simulation_prior_to_wave(self, outcomeType, wave):
        if len(self._outcomes[outcomeType])>0:
            ageAtWave = self.get_age_for_wave(wave)
            outcomesInSim = self.get_outcomes_during_simulation(outcomeType)
            return any( [ageAtOutcome<ageAtWave for ageAtOutcome, _ in outcomesInSim] )
        else:
            return False

    def has_outcome(self, outcomeType):
        return len(self._outcomes[outcomeType]) > 0

    def has_any_outcome(self, outcomeTypeList):
        return any( [self.has_outcome(outcomeType) for outcomeType in outcomeTypeList] )

    def has_all_outcomes(self, outcomeTypeList):
        return all( [self.has_outcome(outcomeType) for outcomeType in outcomeTypeList] )

    def has_cognitive_impairment(self):
        """Assesses if GCP change was less than half SD of population GCP.
        SD was obtained from 300,000 NHANES population (not advanced).""" 
        return self._outcomes[OutcomeType.COGNITION][-1][1].gcp - self._outcomes[OutcomeType.COGNITION][0][1].gcp < (-0.5*10.3099)

    def has_ci(self):
        return self.has_cognitive_impairement()

    def get_outcome_item(self, outcomeType, phenotypeItem):
        return list(map(lambda x: getattr(x[1], phenotypeItem), self._outcomes[outcomeType]))

    def get_outcome_item_last(self, outcomeType, phenotypeItem):
        return self.get_outcome_item(outcomeType, phenotypeItem)[-1]

    def get_outcome_item_sum(self, outcomeType, phenotypeItem):
        return sum(self.get_outcome_item(outcomeType, phenotypeItem))

    def get_outcome_item_mean(self, outcomeType, phenotypeItem):
        return np.mean(self.get_outcome_item(outcomeType, phenotypeItem))

    def has_stroke_prior_to_simulation(self):
        return self.has_outcome_prior_to_simulation(OutcomeType.STROKE)

    def has_stroke_during_simulation(self):
        return self.has_outcome_during_simulation(OutcomeType.STROKE)

    def has_stroke_during_wave(self, wave):
        return self.has_outcome_during_wave(wave, OutcomeType.STROKE)

    def has_mi_during_wave(self, wave):
        return self.has_outcome_during_wave(wave, OutcomeType.MI)

    def valid_outcome_wave(self, wave):
        if (wave<0) | (wave>self._waveCompleted):
            return False
        else:
            return True
 
    def has_outcome_during_wave(self, wave, outcomeType):
        if not self.valid_outcome_wave(wave):
            return False
        else:
            return len(self._outcomes[outcomeType]) != 0 and self.has_outcome_at_age(outcomeType, self._age[wave])

    def has_outcome_during_or_prior_to_wave(self, wave, outcomeType):
        if not self.valid_outcome_wave(wave):
            return False
        else:
            return len(self._outcomes[outcomeType]) != 0 and self.has_outcome_by_age(outcomeType, self._age[wave])

    def has_outcome_at_age(self, outcomeType, age):
        for outcome_tuple in self._outcomes[outcomeType]:
            if outcome_tuple[0] == age:
                return True
        return False
    
    def has_outcome_by_age(self, outcomeType, age):
        for outcome_tuple in self._outcomes[outcomeType]:
            if outcome_tuple[0] <= age:
                return True
        return False

    def get_age_at_first_outcome(self, outcomeType):
        if len(self._outcomes[outcomeType])>0:
            return self._outcomes[outcomeType][0][0]
        else:
            return None

    def get_age_at_last_outcome(self, outcomeType):
        #Q: should we move the age to the outcome class?
        #TO DO: need to include the selfReported argument to the MI phenotype as I did for the stroke outcome
        return self._outcomes[outcomeType][-1][0] if (len(self._outcomes[outcomeType]) > 0) else None

    def get_age_at_first_outcome_in_sim(self, outcomeType):
        for outcome_tuple in self._outcomes[outcomeType]:
            if not outcome_tuple[1].priorToSim:
                age = outcome_tuple[0]
                return age
        return None

    def get_age_at_last_outcome_in_sim(self, outcomeType):
        if (len(self._outcomes[outcomeType]) > 0):
            (age, priorToSim) = (self._outcomes[outcomeType][-1][0], self._outcomes[outcomeType][-1][1].priorToSim)
            return None if priorToSim else age
        else:
            return None

    def get_wave_at_last_stroke(self):
        ageAtLastStroke = self.get_age_at_last_outcome(OutcomeType.STROKE)
        if (ageAtLastStroke is None): #never had stroke outcome
            return None
        elif (ageAtLastStroke<self._age[0]): #had stroke outcome prior to sim and not in sim
            return None
        else:
            return self.get_wave_for_age(ageAtLastStroke)

    def get_attr_prior_first_stroke_in_sim(self, attr): #assuming that the attribute is a list of floats
        attrList = getattr(self, attr)
        ageAtFirstStroke = self.get_age_at_first_outcome(OutcomeType.STROKE)
        if (ageAtFirstStroke is None): #never had stroke outcome
            return None
        elif (ageAtFirstStroke<self._age[0]): #had stroke outcome prior to sim and not in sim
            return None 
        else: #had stroke outcome in sim
            waveAtFirstStroke = self.get_wave_for_age(ageAtFirstStroke)
            return (attrList[:waveAtFirstStroke])

    def get_attr_prior_last_stroke(self, attr): #assuming that the attribute is a list of floats
        attrList = getattr(self, attr)
        ageAtLastStroke = self.get_age_at_last_outcome(OutcomeType.STROKE)
        if (ageAtLastStroke is None): #never had stroke outcome
            return None
        elif (ageAtLastStroke<self._age[0]): #had stroke outcome prior to sim and not in sim
            return attrList[0] #return the baseline attr as our best estimate of the median prestroke attribute
        else: #had stroke outcome in sim
            waveAtLastStroke = self.get_wave_for_age(ageAtLastStroke)
            return (attrList[:waveAtLastStroke])

    def get_attr_since_last_stroke(self, attr): #assuming that the attribute is a list of floats
        attrList = getattr(self, attr)
        ageAtLastStroke = self.get_age_at_last_outcome(OutcomeType.STROKE)
        if (ageAtLastStroke is None): #never had stroke outcome
            return None
        elif (ageAtLastStroke<self._age[0]): #had stroke outcome prior to sim and not in sim
            return attrList #return the entire list
        else: #had stroke outcome in sim
            waveAtLastStroke = self.get_wave_for_age(ageAtLastStroke)
            return (attrList[waveAtLastStroke:])

    def get_median_attr_prior_last_stroke(self, attr): #assuming that the attribute is a list of floats
        attrPriorToLastStroke = self.get_attr_prior_last_stroke(attr)
        return None if (attrPriorToLastStroke is None) else np.median(attrPriorToLastStroke)

    def get_mean_attr_prior_last_stroke(self, attr): #assuming that the attribute is a list of floats
        attrPriorToLastStroke = self.get_attr_prior_last_stroke(attr)
        return None if (attrPriorToLastStroke is None) else np.mean(attrPriorToLastStroke)

    def get_mean_attr_since_last_stroke(self, attr): #assuming that the attribute is a list of floats
        attrSinceLastStroke = self.get_attr_since_last_stroke(attr)
        return None if (attrSinceLastStroke is None) else np.mean(attrSinceLastStroke)

    def has_fatal_stroke(self):
        return any([stroke.fatal for _, stroke in self._outcomes[OutcomeType.STROKE]])

    def has_fatal_mi(self):
        return any([mi.fatal for _, mi in self._outcomes[OutcomeType.MI]])

    def has_mi_prior_to_simulation(self):
        return self.has_outcome_prior_to_simulation(OutcomeType.MI)

    def has_mi_during_simulation(self):
        return self.has_outcome_during_simulation(OutcomeType.MI)

    def apply_linear_modifications(self, modifications):
        for key, value in modifications.items():
            attribute_value = getattr(self, key)
            attribute_value[-1] = attribute_value[-1] + value

    def apply_static_modifications(self, modifications):
        for key, value in modifications.items():
            attribute_value = getattr(self, key)
            attribute_value.append(value)

    # redraw from models to pick new risk factors for person

    def slightly_randomly_modify_baseline_risk_factors(self, risk_model_repository, rng=None):
        #rng = np.random.default_rng(rng)
        if len(self._age) > 1:
            raise RuntimeError("Can not reset risk factors after advancing person in time")

        return Person(
            age=self._age[0] + rng.integers(-2, 2), #replaces np.random.randint with endpoint=False
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
                "anyPhysicalActivity", risk_model_repository
            ),
            education=self._education,
            smokingStatus=self._smokingStatus,
            alcohol=self._alcoholPerWeek[0],
            antiHypertensiveCount=self.get_next_risk_factor(
                "antiHypertensiveCount", risk_model_repository
            ),
            statin=self.get_next_risk_factor("statin", risk_model_repository),
            otherLipidLoweringMedicationCount=self._otherLipidLoweringMedicationCount,
            creatinine=self.get_next_risk_factor("creatinine", risk_model_repository),
            initializeAfib=(lambda _: False),
            selfReportStrokeAge=50 if self._outcomes[OutcomeType.STROKE] is not None else None,
            selfReportMIAge=50 if self._outcomes[OutcomeType.MI] is not None else None,
        )

    # Using this paper...glucose and a1c are highly related
    # Nathan, D. M., Kuenen, J., Borg, R., Zheng, H., Schoenfeld, D., Heine, R. J., for the A1c-Derived Average Glucose (ADAG) Study Group. (2008). Translating the A1C Assay Into Estimated Average Glucose Values. Diabetes Care, 31(8), 1473–1478.
    # so, will use their formula + a draw from residual distribution fo same moddel in NHANES (which has very simnilar coefficients)

    @staticmethod
    def convert_fasting_glucose_to_a1c(glucose):
        return (glucose + 46.7) / 28.7

    @staticmethod
    def convert_a1c_to_fasting_glucose(a1c):
        return 28.7 * a1c - 46.7

    def get_fasting_glucose(self, use_residual=True, rng=None):
        #rng = np.random.default_rng(rng)
        glucose = Person.convert_a1c_to_fasting_glucose(self._a1c[-1])
        if use_residual:
            glucose += rng.normal(0, 21)
        return glucose

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        personRepr = f"Person(name = {self._name} index = {self._index} "
        for attr in self._staticRiskFactors:
            personRepr += f" {attr}={getattr(self,'_'+attr)}"
        for attr in self._dynamicRiskFactors:
            personRepr += f" {attr}={getattr(self,'_'+attr)[-1]:.1f}"
        for attr in self._defaultTreatments:
            personRepr += f" {attr}={getattr(self,'_'+attr)[-1]}"
        personRepr += ")"
        return personRepr

    def __ne__(self, obj):
        return not self == obj

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        if (self._name!=other._name) | (self._index!=other._index) | (self._randomEffects!=other._randomEffects):
            return False
        for attr in self._staticRiskFactors:
            if getattr(self, '_'+attr) != getattr(other, '_'+attr):
                return False
        for attr in self._dynamicRiskFactors:
            if getattr(self, '_'+attr) != getattr(other, '_'+attr):
                return False     
        for attr in self._defaultTreatments:
            if getattr(self, '_'+attr) != getattr(other, '_'+attr):
                return False   
        if self._outcomes != other._outcomes:
            return False
        else:
            return True

    def __deepcopy__(self):
        staticRiskFactorsDict = dict()
        for key in self._staticRiskFactors:
            staticRiskFactorsDict[key] = getattr(self, "_"+key)
        dynamicRiskFactorsDict = dict()
        for key in self._dynamicRiskFactors:
            dynamicRiskFactorsDict[key] = getattr(self, "_"+key)[0]
        defaultTreatmentsDict = dict()
        for key in self._defaultTreatments:
            defaultTreatmentsDict[key] = getattr(self, "_"+key)[0]
        treatmentStrategiesDict = copy.deepcopy(self._treatmentStrategies)
        outcomesDict = copy.deepcopy(self._outcomes)
        name = self._name
        selfCopy = Person(name, staticRiskFactorsDict, dynamicRiskFactorsDict,
                          defaultTreatmentsDict, treatmentStrategiesDict, outcomesDict)
        selfCopy._index = self._index
        selfCopy._waveCompleted = self._waveCompleted
        selfCopy._randomEffects = copy.deepcopy(self._randomEffects)
        return selfCopy
