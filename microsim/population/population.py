import copy
import logging
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from collections import Counter

from microsim.outcomes.cv_model_repository import CVModelRepository
from microsim.common.data_loader import (get_absolute_datafile_path,
                                  load_regression_model)
from microsim.risk_factors.education import Education
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.gfr_equation import GFREquation
from microsim.population.initialization_repository import InitializationRepository
from microsim.outcomes.outcome import Outcome, OutcomeType, EventOutcomeType
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.person.person import Person
from microsim.person.person_factory import PersonFactory
from microsim.common.age_scope import AgeScope
from microsim.outcomes.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.regression_models.logistic_risk_factor_model import \
    LogisticRiskFactorModel
from microsim.outcomes.stroke_outcome import StrokeOutcome
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType, CategoricalRiskFactorsType, ContinuousRiskFactorsType
from microsim.default_treatments.default_treatments import DefaultTreatmentsType, CategoricalDefaultTreatmentsType, ContinuousDefaultTreatmentsType
from microsim.treatment_strategies.treatment_strategies import ContinuousTreatmentStrategiesType, CategoricalTreatmentStrategiesType, TreatmentStrategiesType
from microsim.population.population_model_repository import PopulationRepositoryType, PopulationModelRepository
from microsim.population.standardized_population import StandardizedPopulation
from microsim.outcomes.wmh_severity import WMHSeverity

class Population:
    """A Population-instance has three main parts:
           1) A set of Person-instances. The state of the Population-instance is essentially the state of all Person-instances (past and present).
           2) The models for predicting the future of these Person-instances in a default way (see the note on "default" below).
           3) Tools for analyzing and reporting the state of the Population-instance.
       people: The set of Person-instances. They are completely independent of each other.
       popModelRepository: a PopulationModelRepository instance. Holds all rules/models for predicting the future of people.
                           The models included in this instance must create a self-consistent set of models.
                           Currently, this instance needs to have the rules for predicting dynamic risk factors, default treatment, and outcomes.
                           Static risk factors are also included for consistency and uniformity but of course static risk factors are not
                           a function of time.
       The Population-instance knows how to predict the future of its people but only in a default way, meaning with a default treatment
       (in order to create the self-consistent set of models). This is done with the advance method of the Population class.
       The advance method includes a treatmentStrategies argument which can be used by classes that utilize a set of Population-instances,
       eg a Trial class. A Trial-instance would then be able to apply different treatmentStrategies to the Population-instances
       by passing a different argument to the Population advance method.
       _waveCompleted: how many times the Population has predicted the future of its people (-1 is none, 0 is 1 year, 1 is 2 years).
       _people: Pandas Series of the Person-instances.
       _n: population size
       _rng: the random number generator for the Population-instance, used only for Population-level methods as all Person-instances
             have their own rng.
       _modelRepository: a dict holding all prediction models, keyed by the PopulationRepositoryType values
                         ("staticRiskFactors", "dynamicRiskFactors", "defaultTreatments", "outcomes").
                         The _staticRiskFactors, _dynamicRiskFactors and _defaultTreatments properties expose the
                         list of keys (factor/treatment names) registered in the corresponding repository.
    """

    # ==========================================================================
    # 1. Construction & repository accessors
    # ==========================================================================

    def __init__(self, people, popModelRepository):

        self._waveCompleted = -1
        self._people = people
        self._n = self._people.shape[0]
        self._rng = np.random.default_rng()
        self._modelRepository = popModelRepository._repository

    @property
    def _staticRiskFactors(self):
        return list(self._modelRepository[PopulationRepositoryType.STATIC_RISK_FACTORS.value]._repository.keys())

    @property
    def _dynamicRiskFactors(self):
        return list(self._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value]._repository.keys())

    @property
    def _defaultTreatments(self):
        return list(self._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value]._repository.keys())

    # ==========================================================================
    # 2. Simulation engine — wave advancement & parallelism
    # ==========================================================================

    def advance(self, years, treatmentStrategies=None, nWorkers=1):
        if nWorkers==1:
            self.advance_serial(years, treatmentStrategies=treatmentStrategies)
        elif nWorkers>1:
            self.advance_parallel(years, treatmentStrategies=treatmentStrategies, nWorkers=nWorkers)
        else:
            #raise, not print: returning normally leaves the population un-advanced and its reports look like results
            raise RuntimeError(f"Invalid nWorkers={nWorkers} argument provided.")

    def advance_serial(self, years, treatmentStrategies=None):
        list(map(lambda x: x.advance(years,
                                     self._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
                                     self._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
                                     self._modelRepository[PopulationRepositoryType.OUTCOMES.value],
                                     treatmentStrategies),
                 self._people))
        #note: need to remember that each Person-instance will have their own _waveCompleted attribute, which may be different than the
        #      Population-level _waveCompleted attribute
        self._waveCompleted += years

    #staticmethod, not a bound method: starmap pickles its function argument, and a bound method
    #pickles self with it, shipping the entire parent population to every worker on top of its sub-population
    @staticmethod
    def worker_advance(subPopulation, years, treatmentStrategies):
        subPopulation.advance_serial(years, treatmentStrategies)
        return subPopulation

    def advance_parallel(self, years, treatmentStrategies=None, nWorkers=2):
        with mp.Pool(nWorkers) as myPool:
            #we do not need to divide the pop in nWorkers parts, could be a different number but
            #the assumption is that all sub populations take about the same amount of time to advance
            subPopulations = self.get_sub_populations(nWorkers)
            subPopulations = myPool.starmap(self.worker_advance, [(sp, years, treatmentStrategies) for sp in subPopulations])
        self._people = pd.concat([sp._people for sp in subPopulations])
        self._waveCompleted += years

    def get_sub_populations(self, nPieces):
        """Divides the _people attribute of a single Population instance in nPieces and creates smaller Population instances
        with the same population model repository. This is a strategy in order to avoid passing the entire _people
        to each worker when advance_parallel is used. Keep in mind that this method may be used by a Population subclass,
        eg NHANESDirectSamplePopulation. The fact that we are not dividing the NHANESDirectSamplePopulation in smaller
        NHANESDirectSamplePopulation instances for now does not create a problem since we continue to use NHANES Person objects
        and the same population model repositories. Returns a list of Population instances. """
        #I am assuming that the split will happen at the beginning of the simulation when all person objects are still needed (not dead)
        #peopleParts = np.array_split(self._people, nPieces) #do not do this, as numpy will no longer allow this
        peopleParts = [self._people.iloc[indices] for indices in np.array_split(np.arange(self._n), nPieces)]
        modelRepositoryParts = [self.get_pop_model_repository_copy() for x in range(nPieces)]
        return [Population(people, modelRepository) for people, modelRepository in zip(peopleParts, modelRepositoryParts)]

    def copy(self):
        #people = self.get_people_copy()
        people = Population.get_people_copy(self._people)
        popModelRepository = self.get_pop_model_repository_copy()
        selfCopy = Population(people, popModelRepository)
        return selfCopy

    def get_pop_model_repository_copy(self):
        return PopulationModelRepository(
                                    self._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value],
                                    self._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value],
                                    self._modelRepository[PopulationRepositoryType.OUTCOMES.value],
                                    self._modelRepository[PopulationRepositoryType.STATIC_RISK_FACTORS.value])

    @staticmethod
    def get_people_copy(people):
        """The Person __deepcopy__ function assumes that the Person object has not been advanced to the future at all."""
        return pd.Series( list(map( lambda x: x.__deepcopy__(), people)) )

    # ==========================================================================
    # 3. People-level accessors & blocking
    # ==========================================================================

    @staticmethod
    def get_people_blocks(people, blockFactor, nBlocks=10):
        if blockFactor in [x.value for x in CategoricalRiskFactorsType]:
            return Population.get_people_blocks_categorical(people, blockFactor)
        elif blockFactor in [x.value for x in ContinuousRiskFactorsType]:
            return Population.get_people_blocks_continuous(people, blockFactor, nBlocks=nBlocks)
        else:
            raise RuntimeError("Unrecognized block factor type in Population get_people_blocks function.")

    @staticmethod
    def get_people_blocks_categorical(people, blockFactor):
        categories = set(list(map(lambda x: getattr(x, "_"+blockFactor), people)))
        blocks = dict()
        for cat in categories:
            blocks[cat] = pd.Series(list(filter(lambda x: getattr(x, "_"+blockFactor)==cat, people)))
        return blocks

    @staticmethod
    def get_people_blocks_continuous(people, blockFactor, nBlocks=10):
        categories = list(range(nBlocks))
        blockFactorMin, blockFactorMax = list(map(lambda x: (min(x), max(x)),
                                                           [list(map(lambda x: getattr(x, "_"+blockFactor)[-1], people))]))[0]
        blockBounds = np.linspace(blockFactorMin, blockFactorMax, nBlocks+1)
        blocks = dict()
        for cat in categories:
            #block 0's lower bound is inclusive, otherwise the person(s) at the minimum fall into no block
            lowerBound = -np.inf if cat==0 else blockBounds[cat]
            blocks[cat] = pd.Series(list(filter(lambda x: (getattr(x,"_"+blockFactor)[-1]>lowerBound) &
                                                              (getattr(x,"_"+blockFactor)[-1]<=blockBounds[cat+1]), people)))
        return blocks

    @staticmethod
    def get_alive_people_count(people):
        return len(list(map(lambda x: x._name, filter(lambda y: y.is_alive, people))))

    @staticmethod
    def get_unique_people_count(people):
        return len(set(list(map(lambda x: x._name, people))))

    @staticmethod
    def get_unique_alive_people_count(people):
        return len(set(list(map(lambda x: x._name, filter(lambda y: y.is_alive, people)))))

    def get_attr(self, attr):
        return list(map(lambda x: getattr(x, "_"+attr), self._people))

    def get_attr_baseline(self, rf):
        return self.get_attr_at_index(rf, 0)

    def get_attr_last(self, rf):
        return self.get_attr_at_index(rf, -1)

    def get_attr_at_index(self, rf, index):
        '''Returns a list of the rf attribute at the given index, for people alive at that index.'''
        return Population.get_people_attr_at_index(self._people, rf, index)

    @staticmethod
    def get_people_attr_at_index(people, rf, index):
        '''Returns a list of the alive people attributes at exactly the index specified.
        People must be alive at the index specified.'''
        rfList = list(map( lambda x: getattr(x, "_"+rf)[index] if x.is_alive_at_index(index) else None, people))
        rfList = list(filter(lambda x: x is not None, rfList))
        rfList = list(map(lambda x: int(x) if (type(x)==bool)|(type(x)==np.bool_) else x, rfList))
        return rfList

    @staticmethod
    def get_people_attr_static(people, rf, index):
        '''Returns a list of the alive people static attributes.
        People must be alive at the index specified.'''
        rfList = list(map( lambda x: getattr(x, "_"+rf) if x.is_alive_at_index(index) else None, people))
        rfList = list(filter(lambda x: x is not None, rfList))
        rfList = list(map(lambda x: int(x) if (type(x)==bool)|(type(x)==np.bool_) else x, rfList))
        return rfList

    # ==========================================================================
    # 4. Age / person-year extraction (delegations to Person)
    # ==========================================================================

    @staticmethod
    def get_age_key_sort_key(key):
        '''Sort key that orders age keys numerically whether they are ages (int) or
           age group strings ("100-104"), which sorted() alone orders lexicographically.'''
        return int(key.split("-")[0]) if isinstance(key, str) else key

    def get_ages_group_counter(self, agesCounter):
        '''agesCounter: Counter instance with keys the ages and values the counts for that age.
        Returns a dictionary with keys the age group string and values the counts for that age group.'''
        agesGroupCounter = dict()
        groupSize = 5 #default size of age group
        for age, count in agesCounter.items():
            groupStart = (age//groupSize)*groupSize
            groupEnd = groupStart + groupSize - 1
            groupKey = f"{groupStart}-{groupEnd}"
            agesGroupCounter[groupKey] = agesGroupCounter.get(groupKey, 0) + count
        return Counter(agesGroupCounter)

    def get_ages(self):
        '''Returns a list with the ages, from all years of the simulation, of the entire population as the elements'''
        ages = map(lambda x: x.get_ages(), self._people) #nested list of lists with ages
        ages = list(itertools.chain.from_iterable(ages)) #flattened list of ages
        return ages

    def get_first_incidence_at_risk_ages(self, outcomeType):
        '''Returns the flattened at-risk person-years for outcomeType across the population.
           Delegates to Person.get_first_incidence_at_risk_ages, which returns [] for priorToSim cases and
           truncates the rest at first in-sim event age.'''
        ages = map(lambda p: p.get_first_incidence_at_risk_ages(outcomeType), self._people)
        return list(itertools.chain.from_iterable(ages))

    def get_ages_with_outcome(self, outcomeType=OutcomeType.STROKE):
        '''Returns a list with the outcome ages of the entire population as the elements'''
        agesWithOutcome = map(lambda x: x.get_ages_with_outcome(outcomeType=outcomeType), self._people) #nested list of lists with ages
        agesWithOutcome = list(itertools.chain.from_iterable(agesWithOutcome))            #flattened list of ages
        return agesWithOutcome

    def get_age_at_first_outcome(self, outcomeType, inSim=True):
        '''First-outcome ages across all people (Nones filtered).
           NOTE: For recurrent outcomes (e.g., STROKE, MI) with inSim=True, a priorToSim case can
           contribute its in-sim recurrence age here, since that recurrence is the first in-sim
           event. That is NOT a first incidence in the cohort sense. For first-incidence analyses,
           use get_first_incidence_age instead, which excludes priorToSim cases entirely.'''
        ages = list(map(lambda x: x.get_age_at_first_outcome(outcomeType, inSim=inSim), self._people))
        ages = list(filter(lambda x: x is not None,ages))
        return ages

    def get_first_incidence_age(self, outcomeType):
        '''First in-sim outcome ages, restricted to people without a priorToSim outcome
           (the at-risk set for first incidence). Delegates to Person.get_first_incidence_age.
           Use this for first-incidence rate calculations; use get_age_at_first_outcome only when
           you genuinely want first in-sim events including recurrences.'''
        ages = map(lambda p: p.get_first_incidence_age(outcomeType), self._people)
        return list(filter(lambda a: a is not None, ages))

    def get_min_age_of_first_outcomes(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.get_min_age_of_first_outcomes(outcomeTypeList, inSim=inSim), self._people))

    def get_min_wave_of_first_outcomes(self, outcomesTypeList=[OutcomeType.STROKE]):
        return list(map(lambda x: x.get_min_wave_of_first_outcomes(outcomesTypeList=outcomesTypeList), self._people))

    def get_min_age_of_first_outcomes_or_last_age(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.get_min_age_of_first_outcomes_or_last_age(outcomeTypeList, inSim=inSim), self._people))

    def get_min_wave_of_first_outcomes_or_last_wave(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.get_min_wave_of_first_outcomes_or_last_wave(outcomeTypeList, inSim=inSim), self._people))

    # ==========================================================================
    # 5. Outcome queries (delegations to Person)
    # ==========================================================================

    def has_outcome(self, outcomeType, inSim=True):
        return list(map(lambda x: x.has_outcome(outcomeType, inSim=inSim), self._people))

    def has_any_outcome(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.has_any_outcome(outcomeTypeList, inSim=inSim), self._people))

    def has_all_outcomes(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.has_all_outcomes(outcomeTypeList, inSim=inSim), self._people))

    def has_any_outcome_by_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=0):
        return list(map(lambda x: x.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave), self._people))

    def has_cognitive_impairment(self):
        return list(map(lambda x: x.has_cognitive_impairment(), self._people))

    def has_ci(self):
        return self.has_cognitive_impairment()

    def get_outcome_count(self, outcomeType):
        return sum(self.has_outcome(outcomeType))

    def get_any_outcome_count(self, outcomeTypeList):
        return sum(self.has_any_outcome(outcomeTypeList, inSim=True))

    def get_outcome_lifetime_prevalence(self, outcomeType):
        '''Fraction of the starting cohort that ever had the outcome (priorToSim or in-sim, alive or dead).
           This is "lifetime prevalence" — distinct from cross-sectional prevalence among the currently alive.'''
        return sum(list(map(lambda x: x.has_outcome(outcomeType, inSim=False), self._people)))/self._n

    def get_any_outcome_lifetime_prevalence(self, outcomeTypeList):
        '''Fraction of the starting cohort that ever had any of the listed outcomes (priorToSim or in-sim).'''
        return sum(list(map(lambda x: x.has_any_outcome(outcomeTypeList, inSim=False), self._people)))/self._n

    def get_outcome_cumulative_incidence(self, outcomeType):
        atRisk = [p for p in self._people if not p.has_outcome_prior_to_simulation(outcomeType)]
        return sum(p.has_outcome_during_simulation(outcomeType) for p in atRisk)/len(atRisk) if len(atRisk) > 0 else 0

    def get_any_outcome_cumulative_incidence(self, outcomeTypeList):
        atRisk = [p for p in self._people if not any(p.has_outcome_prior_to_simulation(ot) for ot in outcomeTypeList)]
        return sum(p.has_any_outcome(outcomeTypeList, inSim=True) for p in atRisk)/len(atRisk) if len(atRisk) > 0 else 0

    def get_outcome_item_first(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_first(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_last(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_last(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_sum(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_sum(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_mean(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_mean(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_overall_change(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_overall_change(outcomeType, phenotypeItem, inSim=inSim), self._people))

    # ==========================================================================
    # 6. Incidence / prevalence / survival rates
    # ==========================================================================

    def get_raw_incidence_by_age(self, outcomeType, groups=False):
        '''Returns a dictionary with the keys being either age (integer, groups=False)
        or the age group (string, groups=True) and the values being the counts for that age or age group.
        Restricted to the at-risk set for first incidence (excludes people with a priorToSim outcome);
        each person's person-years are truncated at the age of their first in-sim event.'''
        pairs = self.get_first_incidence_event_ages_and_at_risk_ages(outcomeType)
        agesCounter = Counter(itertools.chain.from_iterable(atRiskAges for _, atRiskAges in pairs))
        outcomeAgesCounter = Counter(eventAge for eventAge, _ in pairs if eventAge is not None)
        if groups: #if true, then get the counter for age groups, keys are strings now, values are counts for age group category
            outcomeAgesCounter = self.get_ages_group_counter(outcomeAgesCounter)
            agesCounter = self.get_ages_group_counter(agesCounter)
        incidence = dict()
        for age in sorted(agesCounter.keys(), key=Population.get_age_key_sort_key): #sorting here results in a sorted output while printing the incidence rate
            if (age in outcomeAgesCounter.keys()) & (agesCounter[age]!=0): #second conditional avoids division by 0
                incidence[age] = outcomeAgesCounter[age]/agesCounter[age]
            else:
                incidence[age] = 0
        return incidence

    def _get_prevalence_by_age_of_people(self, outcomeType, people, groups):
        '''The prevalence-by-age calculation of get_prevalence_by_age over an explicit list of
           (alive) people, so that the same definition can be applied to any subset of the population.'''
        agesCounter = Counter(map(lambda p: p._current_age, people))
        agesWithOutcomeCounter = Counter(map(lambda p: p._current_age,
            filter(lambda p: p.has_outcome_by_age(outcomeType, p._current_age, inSim=False), people)))
        if groups:
            agesCounter = self.get_ages_group_counter(agesCounter) #converts the Counter of ages to a Counter of age groups
            agesWithOutcomeCounter = self.get_ages_group_counter(agesWithOutcomeCounter) #same thing
        prevalence = dict()
        for key in sorted(agesCounter.keys(), key=Population.get_age_key_sort_key): #sorting keys here preserves sorted insertion order for the prevalence as well
            prevalence[key] = agesWithOutcomeCounter.get(key,0) / agesCounter.get(key,0)
        return Counter(prevalence)

    def get_prevalence_by_age(self, outcomeType, groups=False, byGender=False):
        '''Cross-sectional prevalence at the current simulation snapshot.
           Each alive person contributes once at their current age: to the
           denominator always, and to the numerator if they currently have
           the condition (priorToSim outcome, or in-sim outcome at age
           <= current age). Returns a Counter keyed by age (or age group).
           byGender=True returns {"male": Counter, "female": Counter} instead, each computed
           over that gender's alive people only.'''
        alivePeople = list(filter(lambda p: p.is_alive, self._people))
        if byGender:
            return {gender.name.lower():
                        self._get_prevalence_by_age_of_people(outcomeType,
                            [p for p in alivePeople if p._gender == gender], groups)
                    for gender in NHANESGender}
        return self._get_prevalence_by_age_of_people(outcomeType, alivePeople, groups)

    def get_prevalence_by_gender(self, outcomeType):
        '''Cross-sectional prevalence pooled over all ages, one rate per gender: alive people of
           that gender who currently have the condition over all alive people of that gender.
           Returns {"male": rate, "female": rate}.'''
        alivePeople = list(filter(lambda p: p.is_alive, self._people))
        prevalence = dict()
        for gender in NHANESGender:
            genderPeople = [p for p in alivePeople if p._gender == gender]
            withOutcome = [p for p in genderPeople
                           if p.has_outcome_by_age(outcomeType, p._current_age, inSim=False)]
            prevalence[gender.name.lower()] = (len(withOutcome) / len(genderPeople)
                                               if len(genderPeople) > 0 else 0)
        return prevalence

    def get_event_rate_in_simulation(self, eventType, duration):
        events = [
            person.has_outcome_during_simulation_prior_to_wave(eventType, duration)
            for i, person in self._people.items()
        ]
        totalTime = [
            person.years_in_simulation() if person.years_in_simulation() < duration else duration
            for i, person in self._people.items()
        ]
        return np.array(events).sum() / np.array(totalTime).sum()

    def get_outcome_survival_info(self, outcomesTypeList=[OutcomeType.STROKE], personFunctionsList=[lambda x: x.get_scd_group()]):
        '''Returns a nested list, a list of lists: each sublist corresponds to a single person in the population.
        Each sublist includes information related to survival analysis, time to either censoring or outcome, and desired covariates.
        Currently, the person get_outcome_survival_info function tests if the person object has any of the outcomes provided in the list.
        Covariates are include via the personFunctionsList argument, the list must include pure functions that can be applied to a person object.'''
        return list(map(lambda x: x.get_outcome_survival_info(outcomesTypeList=outcomesTypeList, personFunctionsList=personFunctionsList), self._people))

    def get_followup_person_years_by_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Returns a list with all person years at risk for any of the outcomes in the outcome list by end of wave.
        This includes all person objects in the population even the ones that died during the simulation.'''
        return list(map(lambda x: x.get_followup_person_years_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave), self._people))

    def get_followup_events_and_person_years(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Follow-up (event, personYears) pairs for all people, delegating to the Person pair function
        so the rate numerator and denominator cannot be mismatched.'''
        return list(map(lambda x: x.get_followup_event_and_person_years(outcomesTypeList=outcomesTypeList, wave=wave), self._people))

    def get_first_incidence_event_ages_and_at_risk_ages(self, outcomeType):
        '''First-incidence (eventAge, atRiskAges) pairs for all people, delegating to the Person pair function
        so the rate numerator and denominator cannot be mismatched.'''
        return list(map(lambda x: x.get_first_incidence_event_age_and_at_risk_ages(outcomeType), self._people))

    def get_outcome_incidence_rates_at_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Returns outcome incidence rate per 1000 person-years at the end of the wave argument.
        Need to be careful with wave: wave=0 is the first wave, so set the wave to be number of years you want - 1.'''
        if wave<0:
            raise RuntimeError(f"wave {wave=} cannot be a negative number")
        if self._waveCompleted < wave:
            raise RuntimeError(f"Population has not advanced enough to reach end of {wave=}")
        pairs = self.get_followup_events_and_person_years(outcomesTypeList=outcomesTypeList, wave=wave)
        return 1000. * sum(event for event, _ in pairs) / sum(personYears for _, personYears in pairs)

    def get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Returns outcome incidence rate per 1000 person-years as a dictionary at the end of the wave argument.
        Keys are the SCD and Modality group (for now this goes from 0 to 11) and values are the incidence rates per 1000 person-years.
        Need to be careful with wave: wave=0 is the first wave, so set the wave to be number of years you want - 1
        For example, if you want to get the outcome incidence rates at the end of the first year then you will need to set wave=0.
        The defaul wave=3 is due to Kaiser group publications on stroke and dementia eg Kent2022 (about 4 years was the average follow up).
        By outcome rates, this is interpreted as the presence of any of the outcomes provided in outcomesTypeList at any year for a person.
        The calculation is as follows: for each SCD subgroup, we need to count the logical variables for each person dependent on whether they
        had any of the outcomes in the outecomesTypeList and we also need to count all the years each person was at risk of having any of the outcomes.
        For each subgroup, then we do 1000. * # of people with outcome / # of at risk person years to get the outcome incidence rate.
        This function is also designed to produce outcome incidence rates consistent with the way they were measured in
        Kent2021 (doi:10.1212/WNL.0000000000012602) and Kent2022 (DOI: 10.1161/JAHA.122.027672).'''
        if wave<0:
            raise RuntimeError(f"wave {wave=} cannot be a negative number")
        if self._waveCompleted < wave:
            raise RuntimeError(f"Population has not advanced enough to reach end of {wave=}")
        pairs = self.get_followup_events_and_person_years(outcomesTypeList=outcomesTypeList, wave=wave)
        outcomesForGroup = Counter()
        personYearsForGroup = Counter()
        for group, (outcome, personYears) in zip(self.get_scd_by_modality_group(), pairs):
            outcomesForGroup[group] += outcome
            personYearsForGroup[group] += personYears
        return {group: 1000. * outcomesForGroup[group] / personYearsForGroup[group]
                for group in sorted(outcomesForGroup)}

    # ==========================================================================
    # 7. Age/sex standardization
    # ==========================================================================

    def get_gender_age_of_all_outcomes_in_sim(self, outcomeType, personFilter=None):
        #get [(gender, age), ...] for all people and their outcomes
        genderAge = list(map( lambda x: x.get_gender_age_of_all_outcomes_in_sim(outcomeType),
                              list(filter(personFilter, self._people))))
        #remove empty lists (for Person-objects with no outcomes)
        genderAge = list(filter( lambda y: len(y)>0, genderAge))
        #flatten the list of lists
        genderAge = [x for sublist in genderAge for x in sublist]
        return genderAge

    def get_gender_age_of_all_years_in_sim(self, personFilter=None):
        genderAge = list(map( lambda x: x.get_gender_age_of_all_years_in_sim(),
                              list(filter(personFilter, self._people))))
        #flatten the list
        genderAge = [x for sublist in genderAge for x in sublist]
        return genderAge

    def get_gender_age_counts(self, genderAgeList):
        ages = dict()
        minAge = dict()
        maxAge = dict()
        counts = dict()
        for gender in NHANESGender:
            ages[gender.value] = list(map(lambda x: int(x[1]), list(filter(lambda y: y[0]==gender.value, genderAgeList))))
            if len(ages[gender.value])>0:
                minAge[gender.value] = min(ages[gender.value])
                maxAge[gender.value] = max(ages[gender.value])
            else:
                minAge[gender.value] = 18
                maxAge[gender.value] = 18
            #initialize the dictionary with 0 for all counts
            counts[gender.value] = dict(zip([i for i in range(minAge[gender.value],maxAge[gender.value])],
                                            [0 for i in range(minAge[gender.value],maxAge[gender.value])]))
            #do the counting
            for age in range(minAge[gender.value],maxAge[gender.value]+1):
                counts[gender.value][age] = len(list(filter( lambda x: x==age, ages[gender.value])))
        return counts

    def get_gender_age_counts_grouped(self, counts, ageGroups):
        #the standardized population was in groups, so I need to group my simulation counts too....
        countsGrouped = dict()
        for gender in NHANESGender:
            countsGrouped[gender.value] = [0 for i in range(len(ageGroups[gender.value]))]
            for i, ageGroup in enumerate(ageGroups[gender.value]):
                for age in ageGroup:
                    if age in counts[gender.value].keys():
                        countsGrouped[gender.value][i] += counts[gender.value][age]
        return countsGrouped

    def calculate_mean_age_sex_standardized_incidence(self, outcomeType, year=2016, personFilter=None, adultsOnly=True):
        """Calculates the gender and age standardized # of events pers 100,000 person years. """

        #standardized population age groups and percentages
        standardizedPop = StandardizedPopulation(year=year)
        if adultsOnly:
            #a standardized population includes people age 0 and older, but in the simulation we have people 18 and older
            #so I get the standardized rate in the adult population...
            ageGroups = dict()
            populationPercents = dict()
            popPercentSum = 0
            for gender in NHANESGender:
                ageGroups[gender.value] = []
                populationPercents[gender.value] = []
                for group, percent in zip(standardizedPop.ageGroups[gender.value],
                                          standardizedPop.populationPercents[gender.value]):
                    adultAges = [age for age in group if age >= 18]
                    if adultAges:
                        ageGroups[gender.value].append(adultAges)
                        #the 15-19 group keeps only its adult share, population uniform within a group
                        populationPercents[gender.value].append(percent * len(adultAges) / len(group))
                popPercentSum += sum(populationPercents[gender.value])
            #rescale the population percents
            for gender in NHANESGender:
                populationPercents[gender.value] = [x/popPercentSum for x in populationPercents[gender.value]]
        else:
            ageGroups = standardizedPop.ageGroups
            populationPercents = standardizedPop.populationPercents

        #get [ (gender, age), (gender, age),...] from simulation for all outcomes and do the counting
        outcomeGenderAge = self.get_gender_age_of_all_outcomes_in_sim(outcomeType, personFilter)
        outcomeCounts = self.get_gender_age_counts(outcomeGenderAge)

        #get [ (gender, age), (gender, age),...] from simulation for all persons and do the counting
        personGenderAge = self.get_gender_age_of_all_years_in_sim(personFilter)
        personYearCounts = self.get_gender_age_counts(personGenderAge)

        #the standardized population was in groups, so I need to group my simulation counts too....
        outcomeCountsGrouped = self.get_gender_age_counts_grouped(outcomeCounts,  ageGroups)
        personYearCountsGrouped = self.get_gender_age_counts_grouped(personYearCounts, ageGroups)

        #do the calculation
        outcomeRates = dict()
        expectedOutcomes = 0
        for gender in NHANESGender:
            outcomeRates[gender.value] = [(10**5)*x/y if y!=0 else 0 for x,y in zip(outcomeCountsGrouped[gender.value],
                                                                                    personYearCountsGrouped[gender.value])]
            expectedOutcomes += sum([x*y for x,y in zip(outcomeRates[gender.value],
                                                        populationPercents[gender.value])])
        return expectedOutcomes

    # ==========================================================================
    # 8. Treatment-strategy queries
    # ==========================================================================

    def is_in_treatment_strategy(self, tst=TreatmentStrategiesType.BP.value):
        '''Does not make sense to ask this question a person that is not alive'''
        alivePeople = filter(lambda y: y.is_alive, self._people)
        return list(map(lambda x: x.is_in_treatment_strategy(tst), alivePeople))

    def is_in_any_treatment_strategy(self):
        '''Does not make sense to ask this question a person that is not alive'''
        alivePeople = filter(lambda y: y.is_alive, self._people)
        return list(map(lambda x: x.is_in_any_treatment_strategy(), alivePeople))

    def get_treatment_strategies_with_participation(self):
        '''Returns a list of lists, with each list corresponding to a person alive'''
        alivePeople = filter(lambda y: y.is_alive, self._people)
        return list(map(lambda x: x.get_treatment_strategies_with_participation(), alivePeople))

    def has_meds_added(self, tst=TreatmentStrategiesType.BP.value):
        alivePeople = filter(lambda y: y.is_alive, self._people)
        return list(map(lambda x: x.has_meds_added(tst), alivePeople))

    def has_any_meds_added(self):
        '''It is not reasonable to return True/False, if the person is no longer alive'''
        return list(map(lambda x: x.has_any_meds_added(), filter(lambda y: y.is_alive, self._people)))

    def get_treatment_strategies_with_meds_added(self):
        '''Returns a list of lists, with each list corresponding to a person alive'''
        alivePeople = filter(lambda y: y.is_alive, self._people)
        return list(map(lambda x: x.get_treatment_strategies_with_meds_added(), alivePeople))

    def get_population_set_of_treatment_strategies_with_meds_added(self):
        '''Returns set of treatment strategies where at least 1 person has meds added'''
        tstFlattened = list(itertools.chain.from_iterable(self.get_treatment_strategies_with_meds_added()))
        return set(tstFlattened)

    def get_meds_added(self, tst=TreatmentStrategiesType.BP.value):
        alivePeople = filter(lambda x: x.is_alive, self._people)
        return list(map(lambda x: x.get_meds_added(tst), alivePeople))

    def get_scd_by_modality_group(self):
        return list(map(lambda x: x.get_scd_by_modality_group(), self._people))

    # ==========================================================================
    # 9. Person-year dataframe construction
    # ==========================================================================

    @staticmethod
    def get_outcome_flags_per_wave(person):
        """For each event outcome type, returns a 1/0 list per wave indicating whether
           the outcome occurred at that wave's age. Excludes priorToSim outcomes.
           Returns a dictionary keyed by OutcomeType."""
        nWaves = person._waveCompleted + 1
        outcomeTypes = [OutcomeType(eot.value) for eot in EventOutcomeType]
        flags = {ot: [0]*nWaves for ot in outcomeTypes}
        for ot in outcomeTypes:
            if ot not in person._outcomes:
                continue
            for outcomeAge, outcome in person._outcomes[ot]:
                if outcome.priorToSim:
                    continue
                flags[ot][person.get_wave_for_age(outcomeAge)] = 1
        return flags

    @staticmethod
    def get_outcome_history_per_wave(person):
        """For each event outcome type, returns a 1/0 list per wave indicating whether
           the person had the outcome in a previous wave (not the current wave).
           Excludes priorToSim outcomes. Returns a dictionary keyed by OutcomeType."""
        flags = Population.get_outcome_flags_per_wave(person)
        history = {}
        for ot, outcomeFlags in flags.items():
            cumulative = [0]*len(outcomeFlags)
            seen = 0
            for i in range(len(outcomeFlags)):
                cumulative[i] = seen
                seen = max(seen, outcomeFlags[i])
            history[ot] = cumulative
        return history

    def get_all_person_years_as_df(self):
        """This function creates a dataframe where each row is a person-year from the simulation.
           Thus a single person object will be represented in N rows in this dataframe where N is the
           number of years this person object lived in the simulation."""

        srfList = list(self._modelRepository[PopulationRepositoryType.STATIC_RISK_FACTORS.value]._repository.keys())
        drfList = list(self._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value]._repository.keys())
        dtList = list(self._modelRepository[PopulationRepositoryType.DEFAULT_TREATMENTS.value]._repository.keys())
        outcomeNames = [eot.value for eot in EventOutcomeType]
        outcomeHistoryNames = [eot.value + "History" for eot in EventOutcomeType]
        columnNames = ["name", "index"] + srfList + drfList + dtList + outcomeNames + outcomeHistoryNames
        nestedList = list(map(lambda x:
                          list(zip(*[
                              *[[getattr(x, "_"+"name")]*(x._waveCompleted+1)],
                              *[[getattr(x, "_"+"index")]*(x._waveCompleted+1)],
                              *[[getattr(x, "_"+attr)]*(x._waveCompleted+1) for attr in srfList],
                              *[getattr(x,"_"+attr) for attr in drfList],
                              *[getattr(x,"_"+attr) for attr in dtList],
                              *Population.get_outcome_flags_per_wave(x).values(),
                              *Population.get_outcome_history_per_wave(x).values()])),
                          self._people))
        df = pd.concat([pd.DataFrame(nestedList[i], columns=columnNames) for i in range(len(nestedList))], ignore_index=True)
        boolCols = df.select_dtypes(include="bool").columns
        df[boolCols] = df[boolCols].astype(int)
        return df

    # ==========================================================================
    # 10. Statistical helpers
    # ==========================================================================

    def get_quintiles(self, variableList):
        boundaries = np.quantile(variableList, np.linspace(0, 1, 6))
        quintiles = np.digitize(variableList, boundaries, right=False)
        quintiles[variableList == boundaries[-1]] = 5
        return quintiles

    # ==========================================================================
    # 11. Console reporting — summaries & comparisons
    # ==========================================================================

    def print_baseline_summary(self):
        print(" "*25, "Printing a summary at baseline...")
        self.print_summary_at_index(0)

    def print_lastyear_summary(self):
        print(" "*25, "Printing a summary at the last year of the simulation...")
        self.print_summary_at_index(-1)

    @staticmethod
    def _attr_stats(attrList):
        '''min/q25/median/q75/max/mean/sd of attrList, as native floats.'''
        return {
            "min": float(np.min(attrList)),
            "q25": float(np.quantile(attrList, 0.25)),
            "median": float(np.quantile(attrList, 0.5)),
            "q75": float(np.quantile(attrList, 0.75)),
            "max": float(np.max(attrList)),
            "mean": float(np.mean(attrList)),
            "sd": float(np.std(attrList)),
        }

    @staticmethod
    def _attr_proportions(attrList):
        '''Maps each category value to its proportion in attrList, in sorted-value order.'''
        counts = Counter(attrList)
        n = len(attrList)
        return {key: counts[key]/n for key in sorted(counts.keys())}

    @staticmethod
    def _print_rule():
        '''Prints the standard full-width horizontal rule used to frame reporting blocks.'''
        print(" "*25, "-"*53)

    @staticmethod
    def _print_stats_header():
        '''Prints the column header for a continuous-distribution block
           (min / 0.25 / median / 0.75 / max / mean / sd).'''
        print(" "*25, "min", " "*4, "0.25", " "*2, "med", " "*3, "0.75", " "*3, "max" , " "*2, "mean", " "*3, "sd")

    @staticmethod
    def _print_proportions_header():
        '''Prints the "proportions" sub-header and its rule used before a categorical block.'''
        print(" "*25, "proportions")
        print(" "*25, "-"*11)

    @staticmethod
    def _format_stats_row(s, prec=1):
        '''Formats a stats dict (see _attr_stats) as the 7-column min..sd row, each value in a
           width-7 field with `prec` decimals. Used by the continuous-distribution printers.'''
        return (f"{s['min']:> 7.{prec}f} {s['q25']:> 7.{prec}f} "
                f"{s['median']:> 7.{prec}f} {s['q75']:> 7.{prec}f} "
                f"{s['max']:> 7.{prec}f} {s['mean']:> 7.{prec}f} "
                f"{s['sd']:> 7.{prec}f}")

    @staticmethod
    def get_people_summary_at_index(people, index):
        '''Returns the at-index distribution summary for a Pandas Series of Person objects.
           Factor-name ordering is taken from the first Person. Returns:
             {"continuous":  {name: {"min","q25","median","q75","max","mean","sd"}},
              "proportions": {name: {value: proportion}}}
           "continuous" holds continuous dynamic risk factors followed by continuous default
           treatments. "proportions" holds categorical dynamic risk factors, then static risk
           factors, then categorical default treatments, each mapping sorted category value ->
           proportion among people alive at index. Values are native floats.'''
        p0 = people.iloc[0]
        dynamicRiskFactors = p0._dynamicRiskFactors
        defaultTreatments = p0._defaultTreatments
        staticRiskFactors = p0._staticRiskFactors
        continuous = {}
        for rf in dynamicRiskFactors:
            if rf in [crf.value for crf in ContinuousRiskFactorsType]:
                continuous[rf] = Population._attr_stats(Population.get_people_attr_at_index(people, rf, index))
        for dt in defaultTreatments:
            if dt in [cdt.value for cdt in ContinuousDefaultTreatmentsType]:
                continuous[dt] = Population._attr_stats(Population.get_people_attr_at_index(people, dt, index))
        proportions = {}
        for rf in dynamicRiskFactors:
            if rf in [crf.value for crf in CategoricalRiskFactorsType]:
                proportions[rf] = Population._attr_proportions(Population.get_people_attr_at_index(people, rf, index))
        for rf in staticRiskFactors:
            proportions[rf] = Population._attr_proportions(Population.get_people_attr_static(people, rf, index))
        for dt in defaultTreatments:
            if dt in [cdt.value for cdt in CategoricalDefaultTreatmentsType]:
                proportions[dt] = Population._attr_proportions(Population.get_people_attr_at_index(people, dt, index))
        return {"continuous": continuous, "proportions": proportions}

    def get_summary_at_index(self, index):
        '''Returns the at-index distribution summary for this population.
           See get_people_summary_at_index for the dict shape.'''
        return Population.get_people_summary_at_index(self._people, index)

    def print_summary_at_index(self, index):
        """Prints a summary of both static and dynamic risk factors at index (baseline: index=0, last year: index=-1."""
        summary = self.get_summary_at_index(index)
        print(" "*25, "Printing a summary of risk factors and default treatments...")
        self._print_stats_header()
        self._print_rule()
        for name, s in summary["continuous"].items():
            print(f"{name:>23} {self._format_stats_row(s, 1)}")
        self._print_proportions_header()
        for name, props in summary["proportions"].items():
            print(f"{name:>23}")
            for key, prop in props.items():
                print(f"{key:>23} {prop: 6.2f}")
        print(Population.get_categorical_variables_key())

    def print_baseline_summary_comparison(self, other):
        print(" "*25, "Printing a summary comparison at baseline...")
        self.print_summary_at_index_comparison(other, 0)

    def print_lastyear_summary_comparison(self, other):
        print(" "*25, "Printing a summary comparison at the last year of the simulation...")
        self.print_summary_at_index_comparison(other, -1)

    def print_summary_at_index_comparison(self, other, index):
        '''Prints a summary of both static and dynamic risk factors at index for self and other.
           other is also a Population object.
           baseline: index=0, last year: index=-1'''
        Population.print_people_summary_at_index_comparison(self._people, other._people, index)

    @staticmethod
    def print_people_summary_at_index_comparison(people, other, index):
        '''Prints a summary of both static and dynamic risk factors at index for self and other.
           people and other are both Pandas Series with Person objects.
           baseline: index=0, last year: index=-1'''
        selfSummary = Population.get_people_summary_at_index(people, index)
        otherSummary = Population.get_people_summary_at_index(other, index)
        print(" "*25, "self", " "*50,  "other")
        print(" "*25, "-"*53, " ", "-"*53)
        print(" "*25, "min", " "*4, "0.25", " "*2, "med", " "*3, "0.75", " "*3, "max" , " "*2, "mean", " "*3, "sd", "    min ", "   0.25", " "*2, "med", " "*3, "0.75", " "*3, "max", " "*2, "mean", " "*3, "sd")
        print(" "*25, "-"*53, " ", "-"*53)
        for name, s in selfSummary["continuous"].items():
            o = otherSummary["continuous"][name]
            print(f"{name:>23} {Population._format_stats_row(s, 1)} {Population._format_stats_row(o, 1)}")
        print(" "*25, "self", "  other")
        Population._print_proportions_header()
        for name, props in selfSummary["proportions"].items():
            oProps = otherSummary["proportions"][name]
            print(f"{name:>23}")
            for key, prop in props.items():
                print(f"{key:>23} {prop: 6.2f} {oProps.get(key, 0.0): 6.2f}")
        print(Population.get_categorical_variables_key())

    # ==========================================================================
    # 12. Console reporting — treatment, risk, outcome distributions
    # ==========================================================================

    def get_treatment_strategy_distributions(self):
        '''Returns the proportion distribution of each categorical "<strategy>MedsAdded" variable
           (over alive people, for treatment strategies that have meds added), plus the overall
           "anyMedsAdded" distribution. Returns {variableName: {value: proportion}} in sorted-value
           order, with "anyMedsAdded" last.
           Continuous treatment-strategy variables are not handled yet — add them here when a
           ContinuousTreatmentStrategiesType variable exists.'''
        treatmentStrategies = self.get_population_set_of_treatment_strategies_with_meds_added()
        distributions = {}
        for ts in treatmentStrategies:
            tsv = ts + "MedsAdded"
            if tsv in [ctst.value for ctst in CategoricalTreatmentStrategiesType]:
                distributions[tsv] = Population._attr_proportions(self.get_meds_added(tst=ts))
        distributions["anyMedsAdded"] = Population._attr_proportions(list(map(lambda x: int(x), self.has_any_meds_added())))
        return distributions

    def print_lastyear_treatment_strategy_distributions(self):
        '''Prints distributional information about treatment strategy variables, such as bpMedsAdded,
           statinMedsAdded, but only for the people of the population that are still alive.'''
        distributions = self.get_treatment_strategy_distributions()
        print(" "*25, "self")
        self._print_rule()
        self._print_proportions_header()
        for name, props in distributions.items():
            print(f"{name:>23}")
            for key, prop in props.items():
                print(f"{key:>23} {prop: 6.2f}")

    def get_treatment_strategy_distributions_by_risk(self, wmhSpecific=True):
        '''Returns, for each treatment strategy with meds added, the proportion of alive people in
           each 10-year-CV-risk quintile who had 0, 1, 2, 3, 4, or >4 meds added:
             {"<strategy>MedsAdded": {quintile: [p0, p1, p2, p3, p4, p_gt4]}}
           CV risk uses CVModelRepository(wmhSpecific).'''
        treatmentStrategies = self.get_population_set_of_treatment_strategies_with_meds_added()
        cvModelRepository = CVModelRepository(wmhSpecific=wmhSpecific)
        popAlive = filter(lambda x: x.is_alive, self._people)
        cvRiskList = list(map(lambda x: cvModelRepository.select_outcome_model_for_person(x).get_risk_for_person(x, years=10), popAlive))
        cvRiskQuintiles = self.get_quintiles(cvRiskList)
        distributions = {}
        for tst in treatmentStrategies:
            tsv = tst + "MedsAdded"
            medsAddedList = self.get_meds_added(tst)
            byQuintile = {}
            for ile in sorted(set(cvRiskQuintiles)):
                #meds-added counts for the people in this CV risk quintile
                maForIle = list(map(lambda y: y[1], filter(lambda x: x[0]==ile, zip(cvRiskQuintiles, medsAddedList))))
                #proportion of this quintile's people with exactly 0,1,2,3,4 meds added, then >4
                props = [float(np.mean(list(map(lambda x: 1.*(x==maNumber), maForIle)))) for maNumber in range(0,5)]
                props.append(float(np.mean(list(map(lambda x: 1.*(x>4), maForIle)))))
                byQuintile[ile] = props
            distributions[tsv] = byQuintile
        return distributions

    def print_lastyear_treatment_strategy_distributions_by_risk(self, wmhSpecific=True):
        '''Prints, per treatment strategy with meds added, a CV-risk-quintile x meds-added-count
           proportion table from get_treatment_strategy_distributions_by_risk.'''
        distributions = self.get_treatment_strategy_distributions_by_risk(wmhSpecific=wmhSpecific)
        self._print_rule()
        print(" "*25, "proportions in each quintile")
        self._print_rule()
        for tsv, byQuintile in distributions.items():
            print(" "*25, tsv)
            print(" "*6, "CV risk quintile      0       1       2       3       4      >4 ")
            for ile, props in byQuintile.items():
                printString = f"{ile:>23} "
                for p in props:
                    printString += f"{p:> 7.2f} "
                print(printString)

    def get_outcome_risk_distributions(self, outcomeTypeList=[OutcomeType.CARDIOVASCULAR]):
        '''Returns the distribution of per-person predicted risk for each outcome in outcomeTypeList,
           computed over alive people using the population's own outcome model repository (each
           model's default time horizon). Returns {ot: {"min","q25","median","q75","max","mean","sd"}}
           with native-float values.'''
        outcomeModelRepository = self._modelRepository[PopulationRepositoryType.OUTCOMES.value]
        alivePeople = list(filter(lambda x: x.is_alive, self._people))
        distributions = {}
        for ot in outcomeTypeList:
            modelRepo = outcomeModelRepository._repository[ot]
            risks = list(map(lambda x: modelRepo.select_outcome_model_for_person(x).get_risk_for_person(x), alivePeople))
            distributions[ot] = Population._attr_stats(risks)
        return distributions

    def print_outcome_risk_distributions(self, outcomeTypeList=[OutcomeType.CARDIOVASCULAR]):
        '''Prints the per-outcome risk distribution from get_outcome_risk_distributions:
           a min/0.25/med/0.75/max/mean/sd row per outcome, over alive people.'''
        distributions = self.get_outcome_risk_distributions(outcomeTypeList)
        print(" "*25, "Printing outcome risk distributions...")
        print(" "*25, "alive people count= ", f"{Population.get_alive_people_count(self._people):<8}")
        print(" "*25, "unique alive people count= ", f"{Population.get_unique_alive_people_count(self._people):<8}")
        self._print_stats_header()
        self._print_rule()
        for ot in outcomeTypeList:
            s = distributions[ot]
            print(f"{ot.value:>23} {self._format_stats_row(s, 3)}")

    def get_cv_standardized_rates(self):
        '''Returns age/sex-standardized incidence rates (per 100,000 person-years, year 2016) for the
           cardiovascular-relevant outcomes, overall and for the Black and White subpopulations:
             {ot: {"all": rate, "black": rate, "white": rate}}'''
        outcomes = [OutcomeType.MI, OutcomeType.STROKE, OutcomeType.DEATH,
                    OutcomeType.CARDIOVASCULAR, OutcomeType.NONCARDIOVASCULAR, OutcomeType.DEMENTIA]
        rates = {}
        for ot in outcomes:
            rates[ot] = {
                "all": self.calculate_mean_age_sex_standardized_incidence(ot, 2016),
                "black": self.calculate_mean_age_sex_standardized_incidence(ot, 2016, lambda y: y._black),
                "white": self.calculate_mean_age_sex_standardized_incidence(ot, 2016, lambda y: y._white),
            }
        return rates

    def print_cv_standardized_rates(self, rates=None):
        '''Prints the standardized rates from get_cv_standardized_rates: per outcome, the
           all / Black / White age-sex-standardized rates per 100,000.
           rates: the get_cv_standardized_rates dictionary, when the caller already holds it.
           Standardizing 6 outcomes over 3 subgroups walks the entire population 18 times, so a
           caller that needs the rates as well as this printout should obtain them once and pass
           them in rather than have them computed a second time here.'''
        rates = self.get_cv_standardized_rates() if rates is None else rates
        print("standardized rates (per 100,000)    all        black      white")
        for ot, r in rates.items():
            print(f"{ot.value:>30} {r['all']:> 10.1f} {r['black']:> 10.1f} {r['white']:> 10.1f}")

    def get_outcome_incidence(self, outcomeType, groups=True):
        '''Returns first-incidence rates for outcomeType as a dict:
             {"by_age": {age-or-group: rate}, "pooled_65_plus": rate, "pooled_overall": rate}
           by_age is keyed per age (or 5-year age group if groups=True). All values are restricted
           to the at-risk set (people without a priorToSim outcome) and use person-years truncated
           at each person's first in-sim event. pooled_65_plus / pooled_overall are pooled
           person-year rates over ages 65+ and over all ages.'''
        incidentRate = self.get_raw_incidence_by_age(outcomeType, groups=groups)
        outcomeAges = self.get_first_incidence_age(outcomeType)
        personYears = self.get_first_incidence_at_risk_ages(outcomeType)
        scope65plus = AgeScope(lo=65)
        outcomeAges65plus = [a for a in outcomeAges if scope65plus.contains(a)]
        personYears65plus = [a for a in personYears if scope65plus.contains(a)]
        rate65plus = len(outcomeAges65plus) / len(personYears65plus) if len(personYears65plus) > 0 else 0
        rateOverall = len(outcomeAges) / len(personYears) if len(personYears) > 0 else 0
        return {"by_age": incidentRate, "pooled_65_plus": rate65plus, "pooled_overall": rateOverall}

    def print_outcome_incidence(self, outcomeType=OutcomeType.DEMENTIA, groups=True, summary=None):
        '''Prints the first-incidence rates from get_outcome_incidence: a row per age (or age
           group, if groups=True), then pooled rate (>=65) and rate (overall).
           summary: the get_outcome_incidence dictionary, when the caller already holds it, so that
           the incidence is not computed a second time here.'''
        summary = self.get_outcome_incidence(outcomeType, groups=groups) if summary is None else summary
        self._print_rule()
        print(" "*25, f"{outcomeType.value} incidence rate (first incidence only)")
        self._print_rule()
        print(" "*19, "age", "  rate")
        for group, rate in summary["by_age"].items():
            print(f"{group:>23} {rate:7.4f}")
        print(f"{'rate (>=65)':>23} {summary['pooled_65_plus']:7.4f}")
        print(f"{'rate (overall)':>23} {summary['pooled_overall']:7.4f}")

    def get_outcome_prevalence(self, outcomeType, groups=True):
        '''Returns cross-sectional prevalence for outcomeType as a dict:
             {"by_age": {age-or-group: rate}, "pooled_65_plus": rate, "pooled_overall": rate}
           by_age is keyed per age (or 5-year age group if groups=True). Numerators include both
           priorToSim and in-sim outcomes (anyone currently with the outcome); denominators count
           only currently alive people. pooled_65_plus / pooled_overall are pooled over alive
           people aged 65+ and over all alive people.'''
        prevalence = self.get_prevalence_by_age(outcomeType, groups=groups)
        alive = list(filter(lambda p: p.is_alive, self._people))
        scope65plus = AgeScope(lo=65)
        alive65plus = [p for p in alive if scope65plus.contains(p._current_age)]
        hasOutcome = [p for p in alive if p.has_outcome_by_age(outcomeType, p._current_age, inSim=False)]
        hasOutcome65plus = [p for p in hasOutcome if scope65plus.contains(p._current_age)]
        pooled65plus = len(hasOutcome65plus) / len(alive65plus) if len(alive65plus) > 0 else 0
        pooledOverall = len(hasOutcome) / len(alive) if len(alive) > 0 else 0
        return {"by_age": dict(prevalence), "pooled_65_plus": pooled65plus, "pooled_overall": pooledOverall}

    def print_outcome_prevalence(self, outcomeType=OutcomeType.DEMENTIA, groups=True):
        '''Prints the cross-sectional prevalence from get_outcome_prevalence: a row per age (or age
           group, if groups=True), then pooled (>=65) and pooled (overall).'''
        summary = self.get_outcome_prevalence(outcomeType, groups=groups)
        self._print_rule()
        print(" "*25, f"{outcomeType.value} prevalence rate")
        self._print_rule()
        print(" "*19, "age", "  rate")
        for group, rate in summary["by_age"].items():
            print(f"{group:>23} {rate:7.4f}")
        print(f"{'pooled (>=65)':>23} {summary['pooled_65_plus']:7.4f}")
        print(f"{'pooled (overall)':>23} {summary['pooled_overall']:7.4f}")

    def print_outcome_incidence_prevalence(self, outcomeType=OutcomeType.DEMENTIA, groups=True):
        self.print_outcome_incidence(outcomeType=outcomeType, groups=groups)
        self.print_outcome_prevalence(outcomeType=outcomeType, groups=groups)

    def get_scd_cv_risk_proportions_counts(self):
        '''Returns the raw 2D histogram counts of alive people binned by their 10-year CV risk.
        Rows are CV risk that includes SCD specific information (such as WMH, SBI), columns are
        CV risk without that information. Counts are returned as a nested list of native floats.'''
        alive = filter(lambda x: x.is_alive, self._people) #just an iterator
        cvNonScdSpecificRiskList = list(map(lambda x: CVModelRepository(wmhSpecific=False).select_outcome_model_for_person(x).get_risk_for_person(x, years=10),
                                            alive))

        alive = filter(lambda x: x.is_alive, self._people) #iterator again
        cvRiskList = list(map(lambda x: CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x, years=10), alive))

        binEdges = np.array([0.   , 0.05 , 0.075, 0.1  , 0.125, 0.15 , 1.001]) #use meaningful bins
        personCounts, xEdgesActual, yEdgesActual = np.histogram2d(cvRiskList, cvNonScdSpecificRiskList,  bins=[binEdges,binEdges])
        return [[float(item) for item in row] for row in personCounts]

    @staticmethod
    def get_scd_cv_risk_proportion_matrices(counts):
        '''Given the raw 2D histogram counts (see get_scd_cv_risk_proportions_counts), builds the
        three proportion matrices that print_scd_cv_risk_proportions_table displays. The counts are
        flipped along axis 0 (so the highest SCD specific risk bin prints first) and normalized by
        the total count:
          "proportion"        : flipped, normalized counts,
          "cumulative_column" : column-wise (axis 0) cumulative sum of flipped counts, normalized,
          "cumulative_row"    : row-wise (axis 1) cumulative sum of the flipped counts, normalized.
        Each matrix is returned as a nested list of native floats.'''
        personCounts = np.array(counts)
        proportion = np.flip(personCounts, axis=0)/personCounts.sum()
        cumulativeColumn = np.flip(personCounts, axis=0).cumsum(axis=0)/personCounts.sum()  #column wise
        #cumulativeColumn = np.flip(personCounts, axis=0).cumsum(axis=0).cumsum(axis=1)/personCounts.sum() #from top left to bottom right
        cumulativeRow = np.flip(personCounts, axis=0).cumsum(axis=1)/personCounts.sum()  #row wise
        #cumulativeRow = np.flip(personCounts, axis=0).cumsum(axis=0).cumsum(axis=1)/personCounts.sum() #from top left to bottom right
        return {
            "proportion": [[float(item) for item in row] for row in proportion],
            "cumulative_column": [[float(item) for item in row] for row in cumulativeColumn],
            "cumulative_row": [[float(item) for item in row] for row in cumulativeRow],
        }

    def print_scd_cv_risk_proportions_table(self):
        '''Prints a table of proportions where the columns are CV risks without taking into account SCD specific information, such as WMH, SBI,
        and the rows are CV risks that include SCD specific information.'''
        counts = self.get_scd_cv_risk_proportions_counts()
        matrices = self.get_scd_cv_risk_proportion_matrices(counts)

        risks = ["0.0-0.05", "0.05-0.075", "0.075-0.1", "0.1-0.125", "0.125-0.15", "0.15-1.0"]

        self._print_rule()
        print(" "*25, "proportion of people in risk bins")
        self._print_rule()
        print(" "*25, "CV risk (non-SCD specific)")
        print(" "*2, "CV risk (SCD specific) " + " ".join(risks)) #     0       1       2       3       4 ")
        for i,row in enumerate(matrices["proportion"]):
            printString = f"{risks[-i-1]:>23} "
            for item in row:
                printString += f"{item:> 9.2f} "
            print(printString)

        self._print_rule()
        print(" "*25, "cumulative (column-wise) proportion of people in risk bins")
        self._print_rule()
        print(" "*25, "CV risk (non-SCD specific)")
        print(" "*2, "CV risk (SCD specific) " + " ".join(risks)) #     0       1       2       3       4 ")
        for i,row in enumerate(matrices["cumulative_column"]):
            printString = f"{risks[-i-1]:>23} "
            for item in row:
                printString += f"{item:> 9.2f} "
            print(printString)

        self._print_rule()
        print(" "*25, "cumulative (row-wise) proportion of people in risk bins")
        self._print_rule()
        print(" "*25, "CV risk (non-SCD specific)")
        print(" "*2, "CV risk (SCD specific) " + " ".join(risks)) #     0       1       2       3       4 ")
        for i,row in enumerate(matrices["cumulative_row"]):
            printString = f"{risks[-i-1]:>23} "
            for item in row:
                printString += f"{item:> 9.2f} "
            print(printString)


    def get_wmh_outcome_summary(self):
        '''Returns the proportion of people in each WMH severity category (including "unknown" for
        people whose WMH severity was not determined) and the proportion of people with an SBI.'''
        if any(len(p._outcomes.get(OutcomeType.WMH, [])) == 0 for p in self._people):
            raise RuntimeError("""get_wmh_outcome_summary requires every person to have a WMH outcome,
                                  which only Kaiser populations are built with.""")
        severityList = list(map(lambda x: x._outcomes[OutcomeType.WMH][0][1].wmhSeverity, self._people))
        severityList = [y.value if y is not None else "unknown" for y in severityList]
        sbiList =  list(map(lambda x: x._outcomes[OutcomeType.WMH][0][1].sbi, self._people))
        severity = {sev.value: sum([x==sev.value for x in severityList])/self._n for sev in WMHSeverity}
        severity["unknown"] = sum([x=="unknown" for x in severityList])/self._n
        return {"severity": severity, "sbi": sum(sbiList)/self._n}

    def print_wmh_outcome_summary(self):

        summary = self.get_wmh_outcome_summary()
        print("\n")
        print(" "*25, "Printing a summary of the WMH outcome...")
        print(" "*16, "severity proportion")
        print(" "*25, "-"*16)
        for severity in WMHSeverity:
            print(f"{severity.value:>23} {summary['severity'][severity.value]:>6.2f}")
        print(" "*15, f"unknown {summary['severity']['unknown']:>6.2f}\n")
        print(" "*21, "SBI proportion")
        print(" "*25, "-"*16)
        print(" "*18,f"TRUE {summary['sbi']:>6.2f}")

    # ==========================================================================
    # 13. Plotting
    # ==========================================================================

    def plot_outcome_incidence(self, path=None, outcomeType=OutcomeType.DEMENTIA):
        '''Produces the outcome incidence rate by age.'''
        incidentRate = self.get_raw_incidence_by_age(outcomeType)
        plt.scatter(incidentRate.keys(), incidentRate.values())
        plt.xlabel("age")
        plt.ylabel("outcome incidence rate")
        if path is None:
            plt.show()
        else:
            plt.savefig(path+"/outcome-incidence-rate.png")
            plt.clf()
            print("exported results as PNG figures")
        ageOutcome = list(map(lambda y: (y._age[-1], len(y._outcomes[outcomeType])>0),
                               list(filter(lambda x: x.is_alive, self._people))))
        nAlive = len(ageOutcome)
        ageOutcome = list(filter(lambda x: x[1]==True, ageOutcome))
        ageOutcome = [int(x[0]) for x in ageOutcome]
        plt.hist(ageOutcome)
        plt.xlabel("age")
        plt.title(f"outcome cases at end of simulation ({nAlive} Person objects alive)")
        if path is None:
            plt.show()
        else:
            plt.savefig(path+"/outcome-cases-at-end.png")
            plt.clf()
            print("exported results as PNG figures")

    def plot_vascular_rfs_last_wave(self, other, path=None):
        '''Histogram every dynamic risk factor at the last wave, overlaid with `other`
           for risk factors that have an NHANES counterpart (per PersonFactory.microsimToNhanes).
           other: a Population instance to compare against.
           path: directory to save the figure to; if None, the figure is shown interactively.'''
        dynamicRiskFactors = self._dynamicRiskFactors
        nRows = (len(dynamicRiskFactors) + 1) // 2 #ceil, round() would give 6 rows for 13 factors
        fig, ax = plt.subplots(nRows, 2, figsize=(17,15))
        row=-1
        for i,rf in enumerate(dynamicRiskFactors):
            rfList = self.get_attr_at_index(rf, -1)
            if i%2==0:
                row += 1
                col = 0
            else:
                col = 1
            if rf in PersonFactory.microsimToNhanes.keys():
                rfListNhanes = other.get_attr_at_index(rf, -1)
                ax[row,col].hist([rfList, rfListNhanes], bins=20, density=True)
            else:
                ax[row,col].hist(rfList, bins=20, density=True)
            ax[row,col].set_xlabel(rf)
            #ax[row,col].set_ylabel("probability density")
        if len(dynamicRiskFactors) % 2 == 1: #odd count leaves the last panel empty
            ax[nRows-1, 1].axis("off")
        plt.suptitle("probability densities for all dynamic risk factors")
        #plt.subplots_adjust(wspace=0.5, hspace=0.7)
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path+"/probabilities-for-all-rf.png")
            plt.clf()
            print("exported results as PNG figures")

    # ==========================================================================
    # 14. Static legend helper
    # ==========================================================================

    @staticmethod
    def get_categorical_variables_key():
        '''Returns a string that maps the integer categories to their string, and easily understandable by humans, representations'''

        raceKey = "\n" + " "*10 + "raceEthnicity  "
        for race in RaceEthnicity:
            raceKey += f"{race.value}: {race.name}, "
        raceKey = raceKey[:-2]

        edKey = "\n" + " "*14 + "education  "
        for ed in Education:
            edKey += f"{ed.value}: {ed.name}, "
        edKey = edKey[:-2]

        genderKey = "\n" + " "*17 + "gender  "
        for gender in NHANESGender:
            genderKey += f"{gender.value}: {gender.name}, "
        genderKey = genderKey[:-2]

        ssKey = "\n" + " "*10 + "smokingStatus  "
        for ss in SmokingStatus:
            ssKey += f"{ss.value}: {ss.name}, "
        ssKey = ssKey[:-2]

        booleanKey = "\n" + " "*6 + "boolean variables  0: False, 1: True"
        categoricalKey =  " "*25 + "Categorical Variables Key\n" + " "*25 + "-"*53 + raceKey + edKey + genderKey + ssKey + booleanKey
        return categoricalKey
