import math
import copy
import numpy as np

from microsim.outcomes.outcome import OutcomeType
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.gfr_equation import GFREquation
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.risk_factor_bounds import RiskFactorBounds
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType, TreatmentStrategyStatus
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.risk_factors.a1c import convert_a1c_to_fasting_glucose
from microsim.outcomes.cognition_outcome import (
    MMSE_CEILING, MMSE_LOGISTIC_OFFSET, MMSE_LOGISTIC_GCP_SLOPE, MMSE_LOGISTIC_SHAPE,
    GCP_POPULATION_SD, CI_GCP_CHANGE_SD_FACTOR,
    GCP_MEAN_INTERCEPT, GCP_MEAN_AGE_COEFFICIENT, GCP_MEAN_YEARS_IN_SIM_COEFFICIENT, GCP_MEAN_SD, MCI_GCP_SD_FACTOR,
)
from microsim.risk_factors.modality import Modality, modalityGroupMap
from microsim.outcomes.wmh_severity import wmhSeverityGroupMap
from microsim.outcomes.wmh_outcome import scdGroupMap

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
       _name: indicates the specific origin of the Person's instance data, eg in NHANES it will be the NHANES person unique identifier,
              more than one Person instances can have the same name.
       _index: a unique identifier when the Person-instance is part of a bigger group, eg a Population instance (this is set from the Population instance).
       _waveCompleted: every time a complete advanced has been performed, increase this by 1, first complete advanced corresponds to 0.
                       A complete advanced = risk factors, treatment, treatment strategies, updated risk factors, outcomes.
       _outcomes: a dictionary of arrays with the keys being OutcomeTypes, each element in the array is a tuple (age, outcome).
                  Multiple events can be accounted for by having multiple elements in the array.
       _randomEffects: some outcome models require random effects, store them in this dictionary, the outcome models set their key:value."""

    # ==========================================================================
    # 1. Construction & identity
    # ==========================================================================

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
        self._rng = np.random.default_rng() #assume that the OS-derived entropy will be different for each person instance even when mp is used

        #will it be better if static, dynamic RiskFactors and treatments were attributes-dictionaries like the outcomes?
        #will it double the attribute access time by having to find 2 pointers as opposed to 1? how significant will that be?
        #self._staticRiskFactors = staticRiskFactorsDict
        #self._dynamicRiskFactors = dynamicRiskFactorsDict
        #self._defaultTreatments = defaultTreatmentsDict
        #self._treatmentStrategies = treatmentStrategiesDict

        #also, there is currently an inconsistency: outcomes are provided ready to the Person instance but everything
        #else is not, eg the lists are created here and not in the build_person method, if all were dictionaries this would be resolved
        #an attempt on this showed that there are deep dependencies on Person attributes,
        #eg LinearRiskFactorModel.get_model_argument_for_coeff_name expects to finds these attributes directly on Person instances
        #for now I will keep lists of the static, dynamic risk factors etc so that I know how to advance each person
        #even though it is not ideal for memory purposes all Person instances to have exactly the same lists...

        #all elements of risk factors and treatments will become attributes of the person object
        for key,value in staticRiskFactorsDict.items():
            setattr(self, "_"+key, value)
        self._staticRiskFactors = list(staticRiskFactorsDict.keys())
        for key,value in dynamicRiskFactorsDict.items():
            setattr(self, "_"+key, [value])
        self._dynamicRiskFactors = list(dynamicRiskFactorsDict.keys())
        for key,value in defaultTreatmentsDict.items():
            setattr(self, "_"+key, [value])
        self._defaultTreatments = list(defaultTreatmentsDict.keys())
        #treatment strategies and outcomes will remain a dictionary on the person object
        self._treatmentStrategies = treatmentStrategiesDict
        self._outcomes = outcomesDict

    # ==========================================================================
    # 2. Simulation engine (wave advancement)
    # ==========================================================================

    def advance(self, years, dynamicRiskFactorRepository, defaultTreatmentRepository, outcomeModelRepository, treatmentStrategies=None):
        """This function makes all predictions for a person object 1 year to the future.
           Since the person object does not have the models for making the predictions, all models, eg for dynamic risk factors,
           default treatments, outcomes, and treatment strategies, must be provided as arguments.
           years: for how many years we want to make predictions for.
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
        """Makes predictions for the risk factors 1 year to the future."""
        for rf in self._dynamicRiskFactors:
            nextRiskFactor = self.get_next_risk_factor(rf, rfdRepository) #bounds applied by the repository's BoundedRiskFactorModel
            setattr(self, "_"+rf, getattr(self,"_"+rf)+[nextRiskFactor])

    def get_next_risk_factor(self, riskFactor, risk_model_repository):
        model = risk_model_repository.get_model(riskFactor)
        return model.estimate_next_risk(self)

    # Q: it is not clear to me why treatment strategies affect the person attributes directly
    #whereas treatments affect the person attributes indirectly through the attribute regression models
    #will it always be like that? keep in mind that the regression models were designed to be 1 year based predictions
    #the assumption is that the effect of the treatment strategies is instantaneous but
    #there is nothing preventing us from using a regression model as the effect of a treatment strategy
    #also, notice that dynamic risk factors and treatments are lists that get their next quantity in the same way
    def advance_treatments(self, defaultTreatmentRepository):
        """Makes predictions for the default treatments 1 year to the future."""
        for treatment in self._defaultTreatments:
            setattr(self, "_"+treatment, getattr(self,"_"+treatment)+[self.get_next_treatment(treatment, defaultTreatmentRepository)])

    def get_next_treatment(self, treatment, treatmentRepository):
        model = treatmentRepository.get_model(treatment)
        return model.estimate_next_risk(self)

    def advance_treatment_strategies_and_update_risk_factors(self, treatmentStrategies=None):
        """Makes predictionr for the treatment strategies 1 year to the future and updates the risk factors based
           on the effect of those treatment strategies."""
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
        """Updates the person's risk factors due to the effect of applying the treatment strategy."""
        updatedRiskFactors = treatmentStrategy.get_updated_risk_factors(self)
        for rf in self._dynamicRiskFactors:
            if rf in updatedRiskFactors.keys():
                getattr(self, "_"+rf)[-1] = RiskFactorBounds.apply_to_person(rf, updatedRiskFactors[rf], self)

    def update_treatment_strategy_status(self, treatmentStrategy, treatmentStrategyType):
        """The treatment strategy status holds information about whether the strategy is just now being applied on the person
        simply continuous, or ends. The status is important because it dictates, at least for some strategies, what the effect
        on risk factors is. This function decides what the status is based on the current status and whether or not
        the strategy continues to be applied."""
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

    def advance_outcomes(self, outcomeModelRepository):
        """Predict the outcomes of the person for the next year (1 year only)."""
        for outcomeType in self.get_outcomes_in_order(): #loop needs to go over the outcomes in a specific order
            outcome = outcomeModelRepository._repository[outcomeType].select_outcome_model_for_person(self).get_next_outcome(self)
            self.add_outcome(outcome)

    def seed_prevalent_outcomes(self, outcomePrevalenceModelRepository):
        """Seed the person's priorToSim outcomes from the prevalence model repository.
           One-shot initialization at construction (not a wave advance): iterates outcome types in
           the same order as advance_outcomes and skips types with no registered prevalence model
           (value is None in the repository)."""
        for outcomeType in self.get_outcomes_in_order():
            perOutcomeRepo = outcomePrevalenceModelRepository._repository[outcomeType]
            if perOutcomeRepo is None:
                continue
            outcome = perOutcomeRepo.select_outcome_model_for_person(self).get_prevalent_outcome(self)
            self.add_outcome(outcome)

    def get_outcomes_in_order(self):
        """Returns the outcomes in a meaningful order so that the outcome predictions can be made correctly and consistently."""
        outcomesInOrder = list()
        for outcomeType in OutcomeType:
            if outcomeType in list(self._outcomes.keys()):
                outcomesInOrder += [outcomeType]
        return outcomesInOrder

    def add_outcome(self, outcome):
        """Adds the outcome to the person object."""
        if outcome is not None:
            age = None if outcome.priorToSim else self._current_age
            self._outcomes[outcome.type].append((age, outcome))

    # ==========================================================================
    # 3. Treatment-strategy state
    # ==========================================================================

    @property
    def is_in_bp_treatment(self):
        return self.is_in_treatment_strategy(TreatmentStrategiesType.BP.value)

    def is_in_treatment_strategy(self, tst=TreatmentStrategiesType.BP.value):
        '''This function checks if a person is part of a treatment strategy but does not check if the person is being assigned
        additional medications as part of the treatment strategy.
        Status END does not count as being in the strategy: the end effect is applied within the END wave,
        so the person is already off treatment by the time anything downstream asks.'''
        return ( (self._treatmentStrategies[tst]["status"]==TreatmentStrategyStatus.BEGIN) |
                 (self._treatmentStrategies[tst]["status"]==TreatmentStrategyStatus.MAINTAIN) )

    def is_in_any_treatment_strategy(self):
        for tst in TreatmentStrategiesType:
            if self.is_in_treatment_strategy(tst.value):
                return True
        return False

    def get_treatment_strategies_with_participation(self):
        '''Returns a list of treatment strategies where the person is participating in, based on the
        is_in_treatment_strategy function'''
        tstList = list()
        for tst in TreatmentStrategiesType:
            if self.is_in_treatment_strategy(tst.value):
                tstList += [tst.value]
        return tstList

    def has_meds_added(self, tst=TreatmentStrategiesType.BP.value):
        '''This function checks if a person is actively receiving additional medications as part of a treatment strategy,
        ie not just participating in a treatment strategy'''
        if self.is_in_treatment_strategy(tst):
            medsAdded = self._treatmentStrategies[tst].get(tst+"MedsAdded", 0) #key is set by the strategy, may not exist yet
            return True if medsAdded>0 else False
        else:
            return None #because that was a meaningless application of this function

    def has_any_meds_added(self):
        '''This function checks if a person is receiving additional medications as part of any treatment strategy.'''
        if self.is_in_any_treatment_strategy():
            for tst in TreatmentStrategiesType:
                if self.is_in_treatment_strategy(tst.value):
                    if self.has_meds_added(tst.value):
                        return True
            return False
        else:
            return None #that was a meaningless question to ask

    def get_treatment_strategies_with_meds_added(self):
        '''Returns a list with the treatment strategies where the person has actually medications for, based on the
        has_meds_added function'''
        tstList = list()
        for tst in TreatmentStrategiesType:
            if self.has_meds_added(tst.value):
                tstList += [tst.value]
        return tstList

    def get_meds_added(self, tst=TreatmentStrategiesType.BP.value):
        '''This function returns the number of meds added through the provided treatment strategy.'''
        return self._treatmentStrategies[tst].get(tst+"MedsAdded", 0)

    def _antiHypertensiveCountPlusBPMedsAdded(self):
        antiHypertensiveCount = getattr(self, "_"+DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value)[-1]
        if self.is_in_bp_treatment:
            #this means that bpMedsAdded have already been added to the person
            #if "bpMedsAdded" in self._treatmentStrategies[TreatmentStrategiesType.BP.value].keys():
            if self._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]!=TreatmentStrategyStatus.BEGIN:
                return antiHypertensiveCount + self._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]
            else:
                return antiHypertensiveCount
        else:
            return antiHypertensiveCount

    @property
    def _any_antiHypertensive(self):
        return self._antiHypertensiveCount[-1] > 0

    def get_last_default_treatment(self, defaultTreatmentType):
        return getattr(self, "_" + defaultTreatmentType)[-1]

    # ==========================================================================
    # 4. Time / age / wave
    # ==========================================================================

    @property
    def _current_age(self):
        return self._age[-1]

    def get_wave_for_age(self, ageTarget):
        if ageTarget < self._age[0] or ageTarget > self._age[-1]:
            raise RuntimeError(f'Age:: {ageTarget} out of range {self._age[0]}-{self._age[-1]}')
        else:
            return self._age.index(ageTarget)

    def get_age_for_wave(self, wave):
        if not self.valid_wave(wave):
            raise RuntimeError(f'Invalid wave {wave} for person with index {self._index} in get_age_for_wave function.')
        else:
            return self._age[wave]

    def valid_wave(self, wave):
        if (wave<0) | (wave>self._waveCompleted):
            return False
        else:
            return True

    def get_ages(self):
        '''Returns a list with the ages of the person'''
        return getattr(self, "_"+DynamicRiskFactorsType.AGE.value)

    def get_median_age(self):
        medianYear = math.floor(len(self._age) / 2)
        return self._age[medianYear]

    def get_years_in_simulation(self):
        return self._waveCompleted+1

    def get_gender_age_of_all_outcomes_in_sim(self, outcomeType):
        genderAge = []
        if len(self._outcomes[outcomeType])>0:
            for outcome in self._outcomes[outcomeType]:
                if not outcome[1].priorToSim:
                    genderAge += [(getattr(self, "_"+StaticRiskFactorsType.GENDER.value).value, outcome[0])]
        return genderAge

    def get_gender_age_of_all_years_in_sim(self):
        return [(getattr(self, "_"+StaticRiskFactorsType.GENDER.value).value, age) for age in getattr(self, "_"+DynamicRiskFactorsType.AGE.value)]

    # ==========================================================================
    # 5. Life and death
    # ==========================================================================

    def is_alive_at_index(self, index):
        """index=-1: True if the person is still alive; only then does their last
           subscript refer to the latest wave of the simulation.
           index>=0: True if the person survived wave index, False if they died
           during or before it. The boundary is exclusive: a person who died during
           wave index has risk-factor values at that subscript, measured before
           death, but does not count as alive at it.
           For a dead person _waveCompleted is the wave of death."""
        if index < -1:
            raise RuntimeError(f"Index {index} is invalid in is_alive_at_index, only -1 or non-negative indices are allowed.")
        if index == -1:
            return self.is_alive
        return True if self.is_alive else index < self._waveCompleted

    @property
    def is_alive(self):
        """This function needs to return True if a person had died during the current wave updates."""
        return len(self._outcomes[OutcomeType.DEATH])==0

    @property
    def is_dead(self):
        return not self.is_alive

    def get_death_age(self):
        if len(self._outcomes[OutcomeType.DEATH])>0:
            return self._outcomes[OutcomeType.DEATH][0][0]
        else:
            return None

    # ==========================================================================
    # 6. Outcomes - generic queries
    # ==========================================================================

    def has_outcome_at_current_age(self, outcome):
        """This function would probably be meaningfully used at the end of a wave, when outcomes have been predicted"""
        ageAtLastOutcome = self.get_age_at_last_outcome(outcome)
        if (ageAtLastOutcome is None) | (self._current_age!=ageAtLastOutcome):
            return False
        else:
            return True

    def has_fatal_outcome_at_current_age(self, outcome):
        """This function would probably be meaningfully used at the end of a wave, when outcomes have been predicted"""
        if self.has_outcome_at_current_age(outcome):
            return True if self._outcomes[outcome][-1][1].fatal else False
        else:
            return False

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

    def get_outcomes(self, outcomeType, inSim=True):
        return self.get_outcomes_during_simulation(outcomeType) if inSim else self._outcomes[outcomeType]

    def has_outcome_during_simulation_prior_to_wave(self, outcomeType, wave):
        if len(self._outcomes[outcomeType])>0:
            ageAtWave = self.get_age_for_wave(wave)
            outcomesInSim = self.get_outcomes_during_simulation(outcomeType)
            return any( [ageAtOutcome<ageAtWave for ageAtOutcome, _ in outcomesInSim] )
        else:
            return False

    def has_outcome(self, outcomeType, inSim=True):
        return len(self.get_outcomes_during_simulation(outcomeType))>0 if inSim else len(self._outcomes[outcomeType])>0

    def has_any_outcome(self, outcomeTypeList, inSim=True):
        return any( [self.has_outcome(outcomeType, inSim=inSim) for outcomeType in outcomeTypeList] )

    def has_all_outcomes(self, outcomeTypeList, inSim=True):
        return all( [self.has_outcome(outcomeType, inSim=inSim) for outcomeType in outcomeTypeList] )

    def has_outcome_at_age(self, outcomeType, age):
        for outcome_tuple in self._outcomes[outcomeType]:
            if outcome_tuple[0] == age:
                return True
        return False

    def has_outcome_by_age(self, outcomeType, age, inSim=True):
        for outcome_age, outcome in self._outcomes[outcomeType]:
            if outcome.priorToSim:
                if not inSim:
                    return True
            elif outcome_age <= age:
                return True
        return False

    def has_any_outcome_by_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=0):
        minWave = self.get_min_wave_of_first_outcomes(outcomesTypeList)
        if minWave is None:
            return False
        else:
            return True if minWave<=wave else False

    def has_outcome_during_wave(self, wave, outcomeType):
        if not self.valid_wave(wave):
            raise RuntimeError(f"Invalid wave {wave} in has_outcome_during_wave function for person with index {self._index}")
        else:
            return len(self._outcomes[outcomeType]) != 0 and self.has_outcome_at_age(outcomeType, self._age[wave])

    def has_outcome_during_or_prior_to_wave(self, wave, outcomeType):
        #because this function is looking at the current wave, which is, self._waveCompleted+1, I will check
        #for validity in wave-1 only
        #the current wave age exists only mid-advance, so also check the index actually used
        if ((wave!=0) & (not self.valid_wave(wave-1))) | (wave > len(self._age)-1):
            raise RuntimeError(f"Invalid wave {wave} in person.has_outcome_during_or_prior_to_wave function for person with wave completed {self._waveCompleted}.")
        else:
            return len(self._outcomes[outcomeType]) != 0 and self.has_outcome_by_age(outcomeType, self._age[wave])

    def has_incident_event(self, outcomeType):
        # luciana-tag..this feels messy there is probably a better way to deal weith this.
        # age is updated after dementia events are set, so "incident demetnia" is dementia as of the last wave
        inSimOutcomes = self.get_outcomes_during_simulation(outcomeType)
        return (
            (len(inSimOutcomes) > 0)
            and (len(self._age) >= 2)
            and (inSimOutcomes[0][0] == self._age[-2])
        )

    def get_age_at_first_outcome(self, outcomeType, inSim=True):
        if len(self.get_outcomes(outcomeType, inSim=inSim))>0:
            return self.get_outcomes(outcomeType, inSim=inSim)[0][0]
        else:
            return None

    def get_first_incidence_age(self, outcomeType):
        '''First-incidence numerator: returns the age of the first in-sim outcome only when the
           person was at risk for first incidence (no priorToSim outcome). Returns None if the
           person had a priorToSim outcome or had no in-sim event.
           Pairs with get_first_incidence_at_risk_ages as the denominator.'''
        if self.has_outcome_prior_to_simulation(outcomeType):
            return None
        return self.get_age_at_first_outcome(outcomeType, inSim=True)

    def get_min_age_of_first_outcomes(self, outcomeTypeList, inSim=True):
        firstAgeList = list(map(lambda x: self.get_age_at_first_outcome(x, inSim=inSim), outcomeTypeList))
        firstAgeList = list(filter(lambda x: x is not None, firstAgeList))
        return min(firstAgeList) if len(firstAgeList)>0 else None

    def get_min_wave_of_first_outcomes(self, outcomesTypeList=[OutcomeType.STROKE]):
        minAge = self.get_min_age_of_first_outcomes(outcomesTypeList)
        return self.get_wave_for_age(minAge) if minAge is not None else None

    def get_min_age_of_first_outcomes_or_last_age(self, outcomeTypeList, inSim=True):
        minAgeOfFirstOutcomes = self.get_min_age_of_first_outcomes(outcomeTypeList, inSim=inSim)
        return minAgeOfFirstOutcomes if minAgeOfFirstOutcomes is not None else getattr(self, "_"+DynamicRiskFactorsType.AGE.value)[-1]

    def get_min_wave_of_first_outcomes_or_last_wave(self, outcomeTypeList, inSim=True):
        return self.get_wave_for_age(self.get_min_age_of_first_outcomes_or_last_age(outcomeTypeList, inSim=inSim))

    def get_age_at_last_outcome(self, outcomeType):
        #TO DO: need to include the selfReported argument to the MI phenotype as I did for the stroke outcome
        return self._outcomes[outcomeType][-1][0] if (len(self._outcomes[outcomeType]) > 0) else None

    #def get_age_at_first_outcome_in_sim(self, outcomeType):
    #    for outcome_tuple in self._outcomes[outcomeType]:
    #        if not outcome_tuple[1].priorToSim:
    #            age = outcome_tuple[0]
    #            return age
    #    return None

    def get_age_at_last_outcome_in_sim(self, outcomeType):
        if (len(self._outcomes[outcomeType]) > 0):
            (age, priorToSim) = (self._outcomes[outcomeType][-1][0], self._outcomes[outcomeType][-1][1].priorToSim)
            return None if priorToSim else age
        else:
            return None

    # ==========================================================================
    # 7. Outcomes - phenotype item extraction
    # ==========================================================================

    def get_outcome_item(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: getattr(x[1], phenotypeItem), self.get_outcomes(outcomeType, inSim=inSim)))

    def get_outcome_item_last(self, outcomeType, phenotypeItem, inSim=True):
        return self.get_outcome_item(outcomeType, phenotypeItem, inSim=inSim)[-1]

    def get_outcome_item_first(self, outcomeType, phenotypeItem, inSim=True):
        return self.get_outcome_item(outcomeType, phenotypeItem, inSim=inSim)[0]

    def get_outcome_item_sum(self, outcomeType, phenotypeItem, inSim=True):
        return sum(self.get_outcome_item(outcomeType, phenotypeItem, inSim=inSim))

    def get_outcome_item_mean(self, outcomeType, phenotypeItem, inSim=True):
        return np.mean(self.get_outcome_item(outcomeType, phenotypeItem, inSim=inSim))

    def get_outcome_item_overall_change(self, outcomeType, phenotypeItem, inSim=True):
        return self.get_outcome_item_last(outcomeType, phenotypeItem, inSim=inSim) - self.get_outcome_item_first(outcomeType, phenotypeItem, inSim=inSim)

    # ==========================================================================
    # 8. Outcome shortcuts (stroke / MI / dementia / diabetes / CKD / epilepsy)
    # ==========================================================================

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
    def _selfReportStrokePriorToSim(self):
        return False if len(self._outcomes[OutcomeType.STROKE])==0 else self._outcomes[OutcomeType.STROKE][0][1].priorToSim

    @property
    def _selfReportMIPriorToSim(self):
        return False if len(self._outcomes[OutcomeType.MI])==0 else self._outcomes[OutcomeType.MI][0][1].priorToSim

    def has_stroke_prior_to_simulation(self):
        return self.has_outcome_prior_to_simulation(OutcomeType.STROKE)

    def has_stroke_during_simulation(self):
        return self.has_outcome_during_simulation(OutcomeType.STROKE)

    def has_stroke_during_wave(self, wave):
        return self.has_outcome_during_wave(wave, OutcomeType.STROKE)

    def has_fatal_stroke(self):
        return any([stroke.fatal for _, stroke in self._outcomes[OutcomeType.STROKE]])

    def has_mi_prior_to_simulation(self):
        return self.has_outcome_prior_to_simulation(OutcomeType.MI)

    def has_mi_during_simulation(self):
        return self.has_outcome_during_simulation(OutcomeType.MI)

    def has_mi_during_wave(self, wave):
        return self.has_outcome_during_wave(wave, OutcomeType.MI)

    def has_fatal_mi(self):
        return any([mi.fatal for _, mi in self._outcomes[OutcomeType.MI]])

    def has_incident_dementia(self):
        return self.has_incident_event(OutcomeType.DEMENTIA)

    def has_epilepsy(self):
        return self.has_outcome(OutcomeType.EPILEPSY, inSim=False)

    def has_diabetes(self):
        """Sticky diagnosis: True if A1C ever reached 6.5, does not reverse if A1C later drops."""
        return max(self._a1c) >= 6.5

    @property
    def _diabetes(self):
        """Regression-model hook: 'diabetes' coefficients resolve here via getattr."""
        return self.has_diabetes()

    # Q: should we make GFR a dynamic risk factor or outcome or leave it as is?
    @property
    def _gfr(self):
        return GFREquation().get_gfr_for_person(self)

    @property
    def _current_ckd(self):
        return self._gfr < 60

    # ==========================================================================
    # 9. Risk-factor derived shortcuts
    # ==========================================================================

    @property
    def _current_smoker(self):
        return self._smokingStatus == SmokingStatus.CURRENT

    @property
    def _black(self):
        return self._raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK

    @property
    def _white(self):
        return self._raceEthnicity == RaceEthnicity.NON_HISPANIC_WHITE

    # ==========================================================================
    # 10. Cognition
    # ==========================================================================

    @property
    def _baselineGcp(self):
        inSimCognition = [o for o in self._outcomes[OutcomeType.COGNITION] if not o[1].priorToSim]
        if len(inSimCognition) == 0:
            raise RuntimeError("No in-simulation cognition outcome available for baseline GCP.")
        return inSimCognition[0][1].gcp

    @property
    def _gcpSlope(self):
        if len(self._outcomes[OutcomeType.COGNITION])>=2:
            gcpSlope = ( self._outcomes[OutcomeType.COGNITION][-1][1].gcp -
                         self._outcomes[OutcomeType.COGNITION][-2][1].gcp )
        else:
            gcpSlope = 0
        return gcpSlope

    def get_current_mmse(self):
        numerator = MMSE_CEILING
        denominator = (MMSE_LOGISTIC_OFFSET + np.exp(-MMSE_LOGISTIC_GCP_SLOPE * self._outcomes[OutcomeType.COGNITION][-1][1].gcp)) ** (1 / MMSE_LOGISTIC_SHAPE)
        return numerator / denominator

    def has_cognitive_impairment(self):
        """Assesses if GCP change was less than half SD of population GCP.
        SD was obtained from 300,000 NHANES population (not advanced)."""
        #return self._outcomes[OutcomeType.COGNITION][-1][1].gcp - self._outcomes[OutcomeType.COGNITION][0][1].gcp < (-0.5*10.3099)
        return self.get_outcome_item_overall_change(OutcomeType.COGNITION, "gcp") < (-CI_GCP_CHANGE_SD_FACTOR * GCP_POPULATION_SD)

    def has_ci(self):
        return self.has_cognitive_impairment()

    def has_mild_cognitive_impairment(self, inSim=True):
        '''Assesses if GCP is below 1.5 standard deviations from the average GCP for that age and year in simulation
        This linear regression model for the average gcp by age and year in simulation was developed by using a simulated NHANES population
        A similar linear regression model was derived for the standard deviations by age and year in simulation but that showed that the
        standard deviations were essentially constant'''
        gcpMean = GCP_MEAN_INTERCEPT + GCP_MEAN_AGE_COEFFICIENT * self._current_age + GCP_MEAN_YEARS_IN_SIM_COEFFICIENT * self.get_years_in_simulation() #takes into account years in simulation and age
        gcpCutoff = gcpMean - MCI_GCP_SD_FACTOR * GCP_MEAN_SD
        return self.get_outcome_item_last(OutcomeType.COGNITION, "gcp", inSim=inSim) < gcpCutoff

    def has_mci(self, inSim=True):
        return self.has_mild_cognitive_impairment(inSim=inSim)

    # ==========================================================================
    # 11. WMH / SCD classification
    # ==========================================================================

    def has_brain_scan(self):
        '''SCD hazard terms in outcome models must apply only when WMH/SBI findings were observable via a scan'''
        return self._modality != Modality.NO.value

    def has_wmh(self):
        '''a person is defined as having wmh if wmh is set to True in the outcome or if sbi is set to True in the outcome
        For now in the Kaiser population there is only one WMH at most outcome so I am using the first outcome here'''
        if self.has_outcome(OutcomeType.WMH):
            return True if (self.get_outcome_item_first(OutcomeType.WMH, "wmh") | self.get_outcome_item_first(OutcomeType.WMH, "sbi")) else False
        else:
            return False

    def get_scd_group(self):
        '''This function categorizes the Person object based on their WMH outcome.
        Analyses performed on WMH depend, sometimes, on the presence of SBI, WMH so we need to have a categorical variable.
        For example, see Kent2021, Kent2022 and other papers by that group.'''
        sbi = int(self._outcomes[OutcomeType.WMH][0][1].sbi)
        wmh = int(self._outcomes[OutcomeType.WMH][0][1].wmh)
        return scdGroupMap[wmh][sbi]

    def get_modality_group(self):
        '''This function categorizes the Person object based on their modality.
        Analyses performed on WMH depend, sometimes, on the modality so we need a categorical variable for this.'''
        modality = self._modality
        if modality in modalityGroupMap.keys():
            return modalityGroupMap[modality]
        else:
            raise RuntimeError("unknown key in modality group")

    def get_scd_by_modality_group(self):
        '''This function categorizes the Person object based on their SCD group and modality.
        Analyses performed on WMH depend, sometimes, on SCD and modality so we need a categorical variable for this.'''
        scdGroup = self.get_scd_group()
        modalityGroup = self.get_modality_group()
        return modalityGroup*4+scdGroup # ct no sbi & no wmh -> 0, ..., mr no sbi & no wmh -> 4,...,no modality no sbi & no wmh - > 8...

    def get_wmh_severity_group(self):
        '''This function categorizes the Person object based on their WMH severity and whether severity is known or unknown.
        Analyses performed on WMH depend, sometimes, on severity and severity known status so we need a categorical variable for this.'''
        severityUnknown = self._outcomes[OutcomeType.WMH][0][1].wmhSeverityUnknown
        if severityUnknown:
            return wmhSeverityGroupMap['unknown']
        else:
            severity = self._outcomes[OutcomeType.WMH][0][1].wmhSeverity.value
            return wmhSeverityGroupMap[severity]

    def get_wmh_severity_by_modality_group(self):
        '''This function categorizes the Person object based on their WMH severity and modality.
        Analyses performed on WMH depend, sometimes, on severity and modality so we need a categorical variable for this.'''
        modalityGroup = self.get_modality_group()
        wmhSeverityGroup = self.get_wmh_severity_group()
        return modalityGroup*len(wmhSeverityGroupMap) + wmhSeverityGroup #ct and severity no -> 0,..., mr and severity no -> 5

    # ==========================================================================
    # 12. Outcome attribute slicing
    # ==========================================================================

    def get_wave_at_last_outcome(self, outcomeType):
        """Returns the wave of the last in-sim outcome, None if the person never had the
           outcome in sim (priorToSim outcomes carry age=None and have no wave)."""
        ageAtLastOutcome = self.get_age_at_last_outcome_in_sim(outcomeType)
        return None if (ageAtLastOutcome is None) else self.get_wave_for_age(ageAtLastOutcome)

    def get_attr_prior_first_outcome_in_sim(self, attr, outcomeType): #assuming that the attribute is a list of floats
        attrList = getattr(self, attr)
        ageAtFirstOutcome = self.get_age_at_first_outcome(outcomeType) #inSim by default, priorToSim outcomes are skipped
        if (ageAtFirstOutcome is None): #never had this outcome in sim
            return None
        else: #had outcome in sim
            waveAtFirstOutcome = self.get_wave_for_age(ageAtFirstOutcome)
            return (attrList[:waveAtFirstOutcome])

    def get_attr_prior_last_outcome(self, attr, outcomeType): #assuming that the attribute is a list of floats
        attrList = getattr(self, attr)
        if not self.has_outcome(outcomeType, inSim=False): #never had this outcome
            return None
        ageAtLastOutcome = self.get_age_at_last_outcome_in_sim(outcomeType)
        if (ageAtLastOutcome is None): #had outcome prior to sim and not in sim
            return attrList[0] #return the baseline attr as our best estimate of the median pre-event attribute
        else: #had outcome in sim
            waveAtLastOutcome = self.get_wave_for_age(ageAtLastOutcome)
            return (attrList[:waveAtLastOutcome])

    def get_attr_since_last_outcome(self, attr, outcomeType): #assuming that the attribute is a list of floats
        attrList = getattr(self, attr)
        if not self.has_outcome(outcomeType, inSim=False): #never had this outcome
            return None
        ageAtLastOutcome = self.get_age_at_last_outcome_in_sim(outcomeType)
        if (ageAtLastOutcome is None): #had outcome prior to sim and not in sim
            return attrList #return the entire list
        else: #had outcome in sim
            waveAtLastOutcome = self.get_wave_for_age(ageAtLastOutcome)
            return (attrList[waveAtLastOutcome:])

    def get_median_attr_prior_last_outcome(self, attr, outcomeType): #assuming that the attribute is a list of floats
        attrPriorToLastOutcome = self.get_attr_prior_last_outcome(attr, outcomeType)
        return None if (attrPriorToLastOutcome is None) else np.median(attrPriorToLastOutcome)

    def get_mean_attr_prior_last_outcome(self, attr, outcomeType): #assuming that the attribute is a list of floats
        attrPriorToLastOutcome = self.get_attr_prior_last_outcome(attr, outcomeType)
        return None if (attrPriorToLastOutcome is None) else np.mean(attrPriorToLastOutcome)

    def get_mean_attr_since_last_outcome(self, attr, outcomeType): #assuming that the attribute is a list of floats
        attrSinceLastOutcome = self.get_attr_since_last_outcome(attr, outcomeType)
        return None if (attrSinceLastOutcome is None) else np.mean(attrSinceLastOutcome)

    # ==========================================================================
    # 13. QALY and survival accounting
    # ==========================================================================

    def get_total_qalys(self):
        return sum(list(map(lambda x: x[1].qaly, self._outcomes[OutcomeType.QUALITYADJUSTED_LIFE_YEARS])))

    def get_outcome_survival_info(self, outcomesTypeList=[OutcomeType.STROKE], personFunctionsList=[lambda x: x.get_scd_group()]):
        '''Returns person information useful for survival analysis.
        Time to event (based on waves), or last time person was still alive, and other person covariates.
        The personFunctionsList argument should be a list of pure person functions.
        These are useful if you want to do the survival analysis separately for different groups.'''
        time = self.get_min_wave_of_first_outcomes_or_last_wave(outcomesTypeList)+1
        event = int(self.has_any_outcome(outcomesTypeList)) #convert logical to int
        survivalInfo = [time, event]
        if personFunctionsList is not None:
            survivalInfo += [func(self) for func in personFunctionsList]
        return survivalInfo

    def get_person_years_with_outcome_by_end_of_wave(self, outcomeType=OutcomeType.STROKE, wave=3):
        '''Returns the number of person years during which person has outcome.
        Note that wave starts from 0 with a population, so wave=0 would be the end of the first wave.'''
        outcomes = self.get_outcomes_during_simulation(outcomeType)
        #keep age of outcome, convert age to waveForAge, check if waveForAge is less than wave, then count how many
        personYearsWithOutcome = len(list(filter(lambda y: y<=wave, map(lambda x: self.get_wave_for_age(x[0]), outcomes))))
        if (personYearsWithOutcome<0) | (personYearsWithOutcome>wave+1): #waves 0 to wave inclusive can contribute wave+1 person-years
            raise RuntimeError(f"{personYearsWithOutcome=} cannot be <0 or >{wave+1}")
        return personYearsWithOutcome

    def get_followup_person_years_by_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Follow-up-style denominator: person-years until the first in-sim outcome or the end of wave,
           whichever comes first. priorToSim outcomes neither exclude the person nor truncate their time
           (cohort follow-up convention, eg Kent2021/Kent2022).
           Pairs with has_outcome / has_any_outcome_by_end_of_wave as the numerator,
           NOT with the first-incidence functions.'''
        minOrLastWave = self.get_min_wave_of_first_outcomes_or_last_wave(outcomesTypeList) # eg 5
        personYearsAtRisk = min(minOrLastWave, wave) + 1 #with wave=3: 3 + 1
        return personYearsAtRisk

    def get_first_incidence_at_risk_ages(self, outcomeType):
        '''First-incidence denominator: returns the person's at-risk ages for first incidence of outcomeType:
           - empty list if the person had a priorToSim outcome (never at risk for first incidence)
           - all ages if no in-sim event occurred
           - ages truncated at the first in-sim event age (inclusive) otherwise
           Used as person-year contribution to the at-risk denominator for first-incidence rates.
           Pairs with get_first_incidence_age as the numerator.'''
        if self.has_outcome_prior_to_simulation(outcomeType):
            return []
        firstAge = self.get_age_at_first_outcome(outcomeType, inSim=True)
        return self.get_ages() if firstAge is None else [a for a in self.get_ages() if a <= firstAge]

    def get_followup_event_and_person_years(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Follow-up convention as one inseparable pair: (event, personYears).
           event: True if any in-sim outcome in outcomesTypeList occurred by end of wave.
           personYears: years until the first in-sim outcome or the end of wave, whichever comes first.
           priorToSim outcomes neither exclude the person nor truncate their time.
           Use this pair for rates so the numerator and denominator cannot be mismatched.'''
        return (self.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave),
                self.get_followup_person_years_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave))

    def get_first_incidence_event_age_and_at_risk_ages(self, outcomeType):
        '''First-incidence convention as one inseparable pair: (eventAge, atRiskAges).
           A priorToSim outcome yields (None, []): never at risk of first incidence, never an event.
           Use this pair for rates so the numerator and denominator cannot be mismatched.'''
        return (self.get_first_incidence_age(outcomeType),
                self.get_first_incidence_at_risk_ages(outcomeType))

    def get_ages_with_outcome(self, outcomeType=OutcomeType.STROKE):
        '''Returns a list with the ages of the person that did have outcome'''
        return list(map(lambda x: x[0], self.get_outcomes_during_simulation(outcomeType)))

    def get_ages_without_outcome(self, outcomeType=OutcomeType.STROKE):
        '''Returns a list with the ages of the person that did not have the outcome'''
        ages = set(self.get_ages())
        agesWithOutcome = set(self.get_ages_with_outcome(outcomeType=outcomeType))
        agesWithoutOutcome = ages - agesWithOutcome
        return list(agesWithoutOutcome)

    # ==========================================================================
    # 14. Glucose conversions
    # ==========================================================================

    def get_fasting_glucose(self, use_residual=True):
        glucose = convert_a1c_to_fasting_glucose(self._a1c[-1])
        if use_residual:
            glucose += self._rng.normal(0, 21)
        return glucose

    # ==========================================================================
    # 15. Eligibility filters
    # ==========================================================================

    def allhat_candidate(self, end_of_wave_num):
        return (
            (self._age[end_of_wave_num] > 55)
            and (self._sbp[end_of_wave_num] > 140 and self._sbp[end_of_wave_num] < 180)
            and (self._dbp[end_of_wave_num] > 90 and self._dbp[end_of_wave_num] < 110)
            and (
                self._smokingStatus == SmokingStatus.CURRENT
                or self._a1c[end_of_wave_num] > 6.5
                or self.has_stroke_prior_to_simulation()
                or self.has_mi_prior_to_simulation()
                or self._hdl[end_of_wave_num] < 35
            )
        )

    # ==========================================================================
    # 16. State export
    # ==========================================================================

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

    # ==========================================================================
    # 17. Dunder methods
    # ==========================================================================

    def __hash__(self):
        return hash((self._name, self._index))

    def __repr__(self):
        personRepr = f"Person(name = {self._name} index = {self._index} "
        for attr in self._staticRiskFactors:
            personRepr += f" {attr}={getattr(self,'_'+attr)}"
        for attr in self._dynamicRiskFactors:
            value = getattr(self,'_'+attr)[-1]
            personRepr += f" {attr}=None" if value is None else f" {attr}={value:.1f}"
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
        if (self._waveCompleted!=other._waveCompleted) | (self._treatmentStrategies!=other._treatmentStrategies):
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
        return self._outcomes == other._outcomes

    def __deepcopy__(self):
        if self._waveCompleted > -1:
            raise RuntimeError("Person __deepcopy__ can only be used at baseline, before any advance.")
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
