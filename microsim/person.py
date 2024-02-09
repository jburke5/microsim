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

# lucianatag: for this and GCP, this approach is a bit inelegant. the idea is to have classees that can be swapped out
# at the population level to change the behavior about how people change over time.
# but, when we instantiate a person, we don't want to keep a refernce tot the population.
# is the fix just to have the population create people (such that the repository/strategy/model classes can be assigned from within
# the population)

class Person:
    """Person is using risk factors and demographics based off NHANES.
       A Person-instance is essentially a data structure that holds all person-related data, the past and the present.
       The Person class includes functions for essentially two things:
          1) How to predict a Person-instance's future, when the models for making the predictions are provided.
             The predictive models are not stored in Person-instances but only provided as arguments to the Person functions.
          2) Tools for analyzing and reporting a Person-instance's state.
       In order to initialize a Person-instance, the risk factors, treatments, treatment strategies and outcomes need to be 
       provided to the class in an organized way, in their corresponding dictionaries.
       _name: indicates the origin of the Person's instance data, eg in NHANES it will be the NHANES person unique identifier, 
              more than one Person instances can have the same name.
       _index: a unique identifier when the Person-instance is part of a bigger group, eg a Population instance (this is set from the Population instance).
       _waveCompleted: every time a complete advanced has been performed, increase this by 1, first complete advanced corresponds to 0.
                       A complete advanced = risk factors, treatment, treatment strategies, updated risk factors, outcomes.
       _outcomes: a dictionary of arrays with the keys being OutcomeTypes, each element in the array is a tuple (age, outcome).
                  Multiple events can be accounted for by having multiple elements in the array.
       _randomEffects: some outcome models require random effects, store them in this dictionary, the outcome models set their key:value."""

    # Q: I think these would be better stored with a risk factor class, not here....
    _lowerBounds = {DynamicRiskFactorsType.SBP.value: 60,
                    DynamicRiskFactorsType.DBP.value: 20,
                    DynamicRiskFactorsType.CREATININE.value: 0.1}
    _upperBounds = {DynamicRiskFactorsType.SBP.value: 300,
                    DynamicRiskFactorsType.DBP.value: 180}

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
        #also, there is currently an inconsistency: outcomes are provided ready to the Person instance but everything 
        #else is not, eg the lists are created here and not in the build_person method, if all were dictionaries this would be resolved
        #an attempt on this showed that there are deep dependencies on Person attributes, 
        #eg StatsModelLinearRiskFactorModel.get_model_argument_for_coeff_name expects to finds these attributes directly on Person instances
        #for now I will keep lists of the static, dynamic risk factors etc so that I know how to advance each person
        #even though it is not ideal for memory purposes all Person instances to have exactly the same lists...
        for key,value in staticRiskFactorsDict.items():
            setattr(self, "_"+key, value)
        self._staticRiskFactors = list(staticRiskFactorsDict.keys())
        #self._staticRiskFactors = staticRiskFactorsDict
        #for now, only dynamicRiskFactors have bounds, building in manual bounds on extreme values
        for key,value in dynamicRiskFactorsDict.items():
            setattr(self, "_"+key, [self.apply_bounds(key, value)])
        self._dynamicRiskFactors = list(dynamicRiskFactorsDict.keys())
        #self._dynamicRiskFactors = dynamicRiskFactorsDict

        for key,value in defaultTreatmentsDict.items():
            setattr(self, "_"+key, [value])
        self._defaultTreatments = list(defaultTreatmentsDict.keys())
        #self._defaultTreatments = defaultTreatmentsDict

        for key, value in treatmentStrategiesDict.items():
            setattr(self, "_"+key, [value])
        self._treatmentStrategies = list(treatmentStrategiesDict.keys())
        #self._treatmentStrategies = treatmentStrategiesDict

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

    # Q: may also need to implement the apply bounds functionality that is present in the current advance risk factors method 
    #    for Person-objects, I do not know when this was last used though...
    #    also, the population class does not apply bounds in the next risk factor estimates using the df....
    def advance_risk_factors(self, rfdRepository):
        for rf in self._dynamicRiskFactors:
            nextRiskFactor = self.apply_bounds(rf, self.get_next_risk_factor(rf, rfdRepository))
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

    def advance_treatment_strategies_and_update_risk_factors(self, treatmentStrategies):      
        #TO DO:
        #choice of words: get_next implies that it returns the final/next wave quantity, update implies that it modifies
        #that quantity in place
        #the vectorized bp treatment strategies are modifying the rows in place whereas the changes/absolute values are 
        #returned for person objects, the code is much more simple if the person is modified in place with treatment
        #strategies so do that for person objects
        #these two functions will need to be defined
        if treatmentStrategies is not None:
        #if treatmentStrategies[treatment] is not None:
            treatmentStrategies[treatment].update_next_treatment(self)
            #I want to make it explicit and more obvious that treatments update the risk factors
            treatmentStrategies[treatment].update_next_risk_factors(self)

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
        return self._outcomes[OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE][0][1].gcp

    @property
    def _gcpSlope(self):
        if len(self._outcomes[OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE])>=2:
            gcpSlope = ( self._outcomes[OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE][-1][1].gcp -
                         self._outcomes[OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE][-2][1].gcp )
        else:
            gcpSlope = 0
        return gcpSlope
   
    @property
    def _selfReportStrokePriorToSim(self):
        return False if len(self._outcomes[OutcomeType.STROKE])==0 else self._outcomes[OutcomeType.STROKE][0][0]==-1
    
    @property
    def _selfReportMIPriorToSim(self):
        return False if len(self._outcomes[OutcomeType.MI])==0 else self._outcomes[OutcomeType.MI][0][0]==-1
    
    def get_next_treatment(self, treatment, treatmentRepository):
        model = treatmentRepository.get_model(treatment)
        return model.estimate_next_risk(self)

    def get_gender_age_of_all_outcomes_in_sim(self, outcomeType):
        genderAge = []
        if len(self._outcomes[outcomeType])>0:
            for outcome in self._outcomes[outcomeType]:
                if not outcome[1].selfReported:
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
       
    def get_age_at_start_of_wave(self, wave):
        return self._age[wave]

    def get_age_at_end_of_wave(self, wave):
        return self._age[wave+1]


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
    # this method and the following method are used by poulation to get person informaiton
    def get_current_state_as_dict(self):
        return {
            "age": self._age[-1],
            "baseAge": self._age[0],
            "gender": self._gender,
            "raceEthnicity": self._raceEthnicity,
            "black": self._raceEthnicity == 4,
            "sbp": self._sbp[-1],
            "dbp": self._dbp[-1],
            "a1c": self._a1c[-1],
            "current_diabetes": self._a1c[-1] > 6.5,
            "gfr": self._gfr,
            "hdl": self._hdl[-1],
            "ldl": self._ldl[-1],
            "trig": self._trig[-1],
            "totChol": self._totChol[-1],
            "pvd": self._pvd[-1],
            "bmi": self._bmi[-1],
            "anyPhysicalActivity": self._anyPhysicalActivity[-1],
            "education": self._education.value,
            "afib": self._afib[-1],
            "alcoholPerWeek": self._alcoholPerWeek[-1],
            "creatinine": self._creatinine[-1],
            "antiHypertensiveCount": self._antiHypertensiveCount[-1],
            # this variable is used in the risk model...
            # this reflects whether patients have had medications assigned as a risk factor, but 
            # not whether there has been a separate trematent effect, which is tracked in bpMedsAdded
            "current_bp_treatment": self._antiHypertensiveCount[-1] > 0,
            "statin": self._statin[-1],
            "otherLipidLoweringMedicationCount": self._otherLipidLoweringMedicationCount[-1],
            "waist": self._waist[-1],
            "smokingStatus": self._smokingStatus,
            "current_smoker": self._smokingStatus == 2,
            "dead": self.is_dead(),
            "gcpRandomEffect": self._randomEffects["gcp"],
            "gcpStrokeRandomEffect": self._randomEffects["gcpStroke"],
            "gcpStrokeSlopeRandomEffect": self._randomEffects["gcpStrokeSlope"],
            "miPriorToSim": self._selfReportMIPriorToSim,
            "mi": self._selfReportMIPriorToSim or self.has_mi_during_simulation(),
            "stroke": self._selfReportStrokePriorToSim or self.has_stroke_during_simulation(),
            "ageAtFirstStroke": self.get_age_at_first_outcome(OutcomeType.STROKE),
            "ageAtFirstMI": self.get_age_at_first_outcome(OutcomeType.MI),
            "ageAtFirstDementia": self.get_age_at_first_outcome(OutcomeType.DEMENTIA),
            "ageAtLastStroke": self.get_age_at_last_outcome(OutcomeType.STROKE),
            "miInSim": self.has_mi_during_simulation(),
            "strokePriorToSim": self._selfReportStrokePriorToSim,
            "strokeInSim": self.has_stroke_during_simulation(),
            "dementia": self._dementia,
            "gcp": self._gcp[-1],
            "baseGcp": self._gcp[0],
            "gcpSlope": self._gcp[-1] - self._gcp[-2] if len(self._gcp) >= 2 else 0,
            "totalYearsInSim": self.years_in_simulation(),
            "totalQalys": np.array(self._qalys).sum(),
            "totalBPMedsAdded": np.array(self._bpMedsAdded).sum(),
            "bpMedsAdded": self._bpMedsAdded[-1],
            "meanBmiPriorToLastStroke": self.get_mean_attr_prior_last_stroke("_bmi"),
            "meanGcpPriorToLastStroke": self.get_mean_attr_prior_last_stroke("_gcp"),
            "meanSbpPriorToLastStroke": self.get_mean_attr_prior_last_stroke("_sbp"),
            "meanA1cPriorToLastStroke": self.get_mean_attr_prior_last_stroke("_a1c"),
            "meanLdlPriorToLastStroke": self.get_mean_attr_prior_last_stroke("_ldl"),
            "meanWaistPriorToLastStroke": self.get_mean_attr_prior_last_stroke("_waist"),
            "waveAtLastStroke": self.get_wave_at_last_stroke(),
            "meanSbpSinceLastStroke": self.get_mean_attr_since_last_stroke("_sbp"),
            "meanLdlSinceLastStroke": self.get_mean_attr_since_last_stroke("_ldl"),
            "meanA1cSinceLastStroke": self.get_mean_attr_since_last_stroke("_a1c")
        }

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

    def get_final_wave_state_as_dict(self): #this method works only for populations where all people have died
        tvc = ['sbp', 'dbp', 'a1c', 'hdl', 'ldl', 'a1c','trig', 'totChol','bmi', 
                'anyPhysicalActivity', 'afib', 'alcoholPerWeek', 'creatinine', 'antiHypertensiveCount',
                'statin',  'waist', 'alive', 'gcp',
                'bpMedsAdded']
        attributes = self.get_tvc_state_as_dict_long(tvc)
        attributes = {i : j[1:] for i, j in attributes.items()}
        # to get the final wave state, we are going to throw out the first observation for each risk factor, this is what was present at the 
        # start of a wave...
        updatedAgeList = copy.deepcopy(self._age)
        # to get the state at the "end" of the last wave, we need an age value that doesn't exist, so we'll append -1
        # we must do this, when a simulation person dies, they do have events that took place in the last wave, but their ._age array does not get another element
        updatedAgeList.append(-1)
        attributes['age'] = updatedAgeList[1:]
        waveCount = len(self._age) #true for dead simulation people (when a simulation person is alive: waveCount = len(self._age) - 1)
        
        # build a list of ages that includes what a patient's age would have been at teh end of the last wave
        ageThroughEndOfSim = copy.deepcopy(self._age)
        ageThroughEndOfSim.append(ageThroughEndOfSim[-1]+1)

        # on rare occasions a person can die from a CV and non-CV cause in the same year...
        # in that case, they get assigned death twice...which throws the DF lengths off...
        if len(self._alive) >= 2 and (self._alive[-1] == False) and (self._alive[-2] == False):
            attributes['alive'] = attributes['alive'][:-1]
        
        attributes["baseAge"] =  [self._age[0]] * waveCount 
        attributes["gender"] =  [self._gender] * waveCount
        attributes["raceEthnicity"] =  [self._raceEthnicity] * waveCount
        attributes["black"]= [self._raceEthnicity == 4] * waveCount
        attributes["current_diabetes"] = [np.greater(self._a1c[i], 6.5) for i in range(1, waveCount+1)]
        attributes["gfr"] = [GFREquation().get_gfr_for_person_attributes(self._gender, self._raceEthnicity,
            self._creatinine[i], ageThroughEndOfSim[i]) for i in range(1, waveCount+1)]
        attributes["education"] = [self._education.value] * waveCount
        attributes["current_bp_treatment"] = [self._antiHypertensiveCount[i] > 0 for i in range(1, waveCount+1)]
        attributes["smokingStatus"] =  [self._smokingStatus] * waveCount
        attributes["current_smoker"] = [self._smokingStatus == 2] * waveCount
        attributes["gcpRandomEffect"]= [self._randomEffects["gcp"]] * waveCount
        attributes["miPriorToSim"] = [self._selfReportMIPriorToSim] * waveCount
        attributes["strokePriorToSim"] = [self._selfReportStrokePriorToSim] * waveCount
        attributes["mi"] = [self._selfReportMIPriorToSim or self.has_outcome_during_or_prior_to_wave(i, OutcomeType.MI) for i in range(1, waveCount+1)]
        attributes["stroke"] = [self._selfReportStrokePriorToSim or self.has_outcome_during_or_prior_to_wave(i, OutcomeType.STROKE) for i in range(1, waveCount+1)]
        #ageAtFirstX lists: if person never had the outcome all elements are None, if person had outcome at age Y array is [-1,-1,...-1, Y,Y...Y]
        attributes["ageAtFirstStroke"] =  [None if self.get_age_at_first_outcome(OutcomeType.STROKE) is None else self.get_age_at_first_outcome(OutcomeType.STROKE) if self.get_age_at_first_outcome(OutcomeType.STROKE) < i else -1  for i in self._age]
        attributes["ageAtFirstMI"] =  [None if self.get_age_at_first_outcome(OutcomeType.MI) is None else self.get_age_at_first_outcome(OutcomeType.MI) if self.get_age_at_first_outcome(OutcomeType.MI) < i else -1  for i in self._age]
        attributes["ageAtFirstDementia"] =  [None if self.get_age_at_first_outcome(OutcomeType.DEMENTIA) is None else self.get_age_at_first_outcome(OutcomeType.DEMENTIA) if self.get_age_at_first_outcome(OutcomeType.DEMENTIA) < i else -1  for i in self._age]
        attributes["miInSim"] = [self.has_mi_during_wave(i) for i in range(1, waveCount+1)]
        attributes["strokeInSim"] = [self.has_stroke_during_wave(i) for i in range(1, waveCount+1)]
        attributes["dementia"] = [self.has_outcome_during_or_prior_to_wave(i, OutcomeType.DEMENTIA) for i in range(1, waveCount+1)]
        attributes["baseGcp"] = [self._gcp[0]] * waveCount
        attributes["gcpSlope"] = [self._gcp[i-1] - self._gcp[i-2] if i >= 2 else 0 for i in range(1, waveCount+1)]
        attributes["totalYearsInSim"] = np.arange(1, waveCount+1)
        attributes["totalBPMedsAdded"] = [np.array(self._bpMedsAdded[:i]).sum() for i in range(1, waveCount+1)]
        attributes["totalQalys"] = [QALYAssignmentStrategy().get_next_qaly(self, age) for age in self._age]
        
        #for key, val in attributes.items():
        #    print(f"key: {key}, len(val): {len(val)}")

        return attributes

        
    @property
    def _current_smoker(self):
        return self._smokingStatus == SmokingStatus.CURRENT

    @property
    def _current_bp_treatment(self):
        return self._antiHypertensiveCount[-1] > 0

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

    def dead_at_start_of_wave(self, year):
        return (year > len(self._age)) or (self._alive[year-1] == False)

    def dead_at_end_of_wave(self, year):
        return (year > len(self._age)) or (self._alive[year] == False)

    @property
    def _black(self):
        return self._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK

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

    def years_in_simulation(self):
        return len(self._age) - 1

    def get_next_risk_factor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get_model(riskFactor)
        return model.estimate_next_risk(self)

    def get_total_qalys(self):
        return sum(self._qalys)

    def get_qalys_from_wave(self, wave):
        total = 0
        for i in range(wave - 1, len(self._qalys)):
            total += self._qalys[i]
        return total

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

    def advance_year(
        self,
        risk_model_repository,
        outcome_model_repository,
        qaly_assignment_strategy=QALYAssignmentStrategy(),
        rng=None,
    ):
        logging.debug(f"advance_year on person, age: {self._age[0]} sbp : {self._sbp[0]}")
        if self.is_dead():
            raise RuntimeError("Person is dead. Can not advance year")

        self.advance_risk_factors(risk_model_repository, rng=rng)
        self.advance_treatment(risk_model_repository, rng=rng)
        self.advance_outcomes(outcome_model_repository, rng=rng)
        self.assign_qalys(qaly_assignment_strategy)
        if not self.is_dead():
            self._age.append(self._age[-1] + 1)
            self._alive.append(True)

    def dead_at_start_of_wave(self, wave):
        return (wave > len(self._age)) or (self._alive[wave-1] == False)

    def dead_at_end_of_wave(self, wave):
        return (wave > len(self._age)) or (self._alive[wave] == False)


    # this method is trying to enable simple logic in the popuation.
    # when the population asks, "who is alive at a given time point?" it can't merely check
    # the index on person._alive, because people who died prior to that time will not have an index
    # in alive at that time.

    def alive_at_start_of_wave(self, start_wave_num):
        if (self._alive[-1]) and (start_wave_num > (len(self._age))):
            raise Exception(
                f"Trying to find status for a wave: {start_wave_num}, beyond current wave: {len(self._age)}, index: {self._populationIndex}, person: {self}"
            )

        # we always know, regardless of what wave is being inquired about, that a person who was once dead
        # is still dead
        if (self.is_dead()) and (start_wave_num > len(self._alive) - 1):
            return False
        else:
            # this returns whether one was alive at the start of a given wave (i.e. the end of theprior wave)
            return self._alive[start_wave_num - 1]

    def has_outcome_prior_to_simulation(self, outcomeType):
        return any([ageAtEvent < 0 for ageAtEvent, _ in self._outcomes[outcomeType]])

    def has_outcome_during_simulation(self, outcomeType):
        if len(self._outcomes[outcomeType])>0:
           return any([outcome.selfReported == False for _, outcome in self._outcomes[outcomeType]])
        else:
           return False

    def get_outcomes_during_simulation(self, outcomeType):
        return list(filter(lambda x: x[0] > 0, self._outcomes[outcomeType]))

    def has_outcome_during_simulation_prior_to_wave(self, outcomeType, wave):
        return any(
            [ageAtEvent >= self._age[0] + wave for ageAtEvent, _ in self._outcomes[outcomeType]]
        )

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

    def valid_outcome_wave(self, wave):
        if (wave<0) | (wave>self._waveCompleted):
            return False
        else:
            return True
 
    def has_outcome_during_wave(self, wave, outcomeType):
        if not self.valid_outcome_wave(wave):
            return False
        else:
            return len(self._outcomes[outcomeType]) != 0 and self.has_outcome_at_age(outcomeType, self._age[wave - 1])

    def has_outcome_during_or_prior_to_wave(self, wave, outcomeType):
        if not self.valid_outcome_wave(wave):
            return False
        else:
            return len(self._outcomes[outcomeType]) != 0 and self.has_outcome_by_age(outcomeType, self._age[wave])

    def has_outcome_at_age(self, type, age):
        for outcome_tuple in self._outcomes[type]:
            if outcome_tuple[0] == age:
                return True
        return False
    
    def has_outcome_by_age(self, type, age):
        for outcome_tuple in self._outcomes[type]:
            if outcome_tuple[0] <= age:
                return True
        return False

    def get_age_at_first_outcome(self, outcomeType):
        if len(self._outcomes[outcomeType])>0:
            return self._outcomes[outcomeType][0][0]
        else:
            return None

    def get_age_at_last_outcome(self, type):
        #Q: should we move the age to the outcome class?
        #TO DO: need to include the selfReported argument to the MI phenotype as I did for the stroke outcome
        return self._outcomes[type][-1][0] if (len(self._outcomes[type]) > 0) else None

    def get_age_at_first_outcome_in_sim(self, type):
        for outcome_tuple in self._outcomes[type]:
            age = outcome_tuple[0]
            if age > 0:
                return age

        return None

    def get_age_at_last_outcome_in_sim(self, type):
        age = self._outcomes[type][-1][0] if (len(self._outcomes[type]) > 0) else None
        age = None if (age == -1) else age #if outcome was prior to sim, return None
        return age

    #def get_median_attr_prior_last_stroke(self, attr): #assuming that the attribute is a list of floats
    #    attrList = getattr(self, attr)
    #    ageAtLastStroke = self.get_age_at_last_outcome(OutcomeType.STROKE)
    #    if (ageAtLastStroke is None): #never had stroke outcome
    #        return None
    #    elif (ageAtLastStroke==-1): #had stroke outcome prior to sim and not in sim
    #        return attrList[0] #return the baseline attr as our best estimate of the median prestroke attribute
    #    else: #had stroke outcome in sim
    #        waveAtLastStroke = self.get_wave_for_age(ageAtLastStroke)
    #        return np.median(attrList[:waveAtLastStroke])

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

    # should only occur immediately after an event is created — we can't roll back the subsequent implicaitons of an event.
    def rollback_most_recent_event(self, outcomeType):
        # get rid of the outcome event...
        outcomes_for_type = list(self._outcomes[outcomeType])
        outcome_rolled_back = self._outcomes[outcomeType].pop()
        # if the patient died during the wave, then their age didn't advance and their event would be at their
        # age at teh start of the wave.
        rollbackAge = self._age[-1] - 1 if self._alive[-1] else self._age[-1]
        if rollbackAge != outcome_rolled_back[0]:
            raise Exception(
                f"# of outcomes: {len(outcomes_for_type)} while trying to rollback event at age {outcome_rolled_back[0]}, but current age is {rollbackAge} - can not roll back if age has changed, for person: {self}"
            )

        # and, if it was fatal, reset the person to being alive.
        if (outcome_rolled_back)[1].fatal:
            self._alive[-1] = True
            self._age.append(self._age[-1] + 1)

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

    def assign_qalys(self, qaly_assignment_strategy):
        self._qalys.append(qaly_assignment_strategy.get_next_qaly(self))

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
        return (
            f"Person(age={self._age[-1]}, "
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
            f"creatinine={self._creatinine[-1]}, "
            f"statin={self._statin[-1]}, "
            f"index={self._populationIndex if (hasattr(self, '_populationIndex') and self._populationIndex is not None) else None}, "
            f"outcomes={self._outcomes}"
            f")"
        )

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
        if not other._creatinine == self._creatinine:
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
        selfCopy = Person(
            age=0,
            gender=None,
            raceEthnicity=None,
            sbp=0,
            dbp=0,
            a1c=0,
            hdl=0,
            totChol=0,
            bmi=0,
            ldl=0,
            trig=0,
            waist=0,
            anyPhysicalActivity=0,
            education=None,
            smokingStatus=None,
            alcohol=None,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0,
            initializeAfib=None,
        )
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
        selfCopy._creatinine = copy.deepcopy(self._creatinine)
        selfCopy._otherLipidLoweringMedicationCount = copy.deepcopy(
            self._otherLipidLoweringMedicationCount
        )
        selfCopy._outcomes = copy.deepcopy(self._outcomes)
        selfCopy._selfReportStrokePriorToSim = copy.deepcopy(self._selfReportStrokePriorToSim)
        selfCopy._selfReportMIPriorToSim = copy.deepcopy(self._selfReportMIPriorToSim)
        selfCopy._selfReportMIAge = copy.deepcopy(self._selfReportMIAge) if hasattr(self, "_selfReportMIAge") else None  
        selfCopy._selfReportStrokeAge = copy.deepcopy(self._selfReportStrokeAge) if hasattr(self, "_selfReportStrokeAge") else None 
        selfCopy._afib = self._afib
        selfCopy._bpTreatmentStrategy = self._bpTreatmentStrategy
        selfCopy._afib = copy.deepcopy(self._afib)
        selfCopy._gcp = copy.deepcopy(self._gcp)
        selfCopy._randomEffects = copy.deepcopy(self._randomEffects)
        selfCopy._populationIndex =copy.deepcopy(self._populationIndex) if hasattr(self, "_populationIndex") else None 
        selfCopy.dfIndex =copy.deepcopy(self.dfIndex) if hasattr(self, "dfIndex") else None 

        return selfCopy
