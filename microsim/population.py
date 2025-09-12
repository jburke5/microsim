import copy
import logging
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pandarallel import pandarallel
from collections import Counter

#from microsim.alcohol_category import AlcoholCategory
from microsim.bp_treatment_strategies import *
#from microsim.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository, 
#                                                   CohortStaticRiskFactorModelRepository,
#                                                   CohortDefaultTreatmentModelRepository)
from microsim.data_loader import (get_absolute_datafile_path,
                                  load_regression_model)
#from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.gfr_equation import GFREquation
from microsim.initialization_repository import InitializationRepository
from microsim.nhanes_risk_model_repository import NHANESRiskModelRepository
from microsim.outcome import Outcome, OutcomeType
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.person import Person
from microsim.person_factory import PersonFactory
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
#from microsim.race_ethnicity import RaceEthnicity
#from microsim.smoking_status import SmokingStatus
from microsim.statsmodel_logistic_risk_factor_model import \
    StatsModelLogisticRiskFactorModel
from microsim.stroke_outcome import StrokeOutcome
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType, CategoricalRiskFactorsType, ContinuousRiskFactorsType
#from microsim.afib_model import AFibPrevalenceModel
#from microsim.pvd_model import PVDPrevalenceModel
from microsim.treatment import DefaultTreatmentsType, TreatmentStrategiesType, CategoricalDefaultTreatmentsType, ContinuousDefaultTreatmentsType, ContinuousTreatmentStrategiesType, CategoricalTreatmentStrategiesType
from microsim.population_model_repository import PopulationRepositoryType, PopulationModelRepository
from microsim.standardized_population import StandardizedPopulation
from microsim.risk_model_repository import RiskModelRepository
from microsim.wmh_severity import WMHSeverity

class Population:
    """A Population-instance has three main parts:
           1) A set of Person-instances. The state of the Population-instance is essentially the state of all Person-instances (past and present).
           2) The models for predicting the future of these Person-instances in a default way (I explain default in a bit)
           3) Tools for analyzing and reporting the state of the Population-instance.
       people: The set of Person-instances. They are completely independent of each other.
       popModelRepository: a PopulationRepositoryType instance. Holds all rules/models for predicting the future of people.
                           The models included in this instance must create a self-consistent set of models.
                           Currently, this instance needs to have the rules for predicting dynamic risk factors, default treatment, and outcomes.
                           Static risk factors are also included for consistency and uniformity but of course static risk factors are not 
                           a function of time. 
       The Population-instance knows how to predict the future of its people but only in a default way, meaning with a default treatment
       (in order to create the self-consistent set of models). This is done with the advance method of the Population class.
       The advance method includes a treatmentStrategies argument which can be used by classes that utilize a set of Population-instances,
       eg a Trial class. A Trial-instance would then be able to apply diffferent treatmentStrategies to the Population-instances
       by passing a different argument to the Population advance method.
       _wavesCompleted: how many times the Population has predicted the future of its people (-1 is none, 0 is 1 year, 1 is 2 years).
       _people: Pandas Series of the Person-instances.
       _n: population size
       _rng: the random number generator for the Population-instance, used only for Population-level methods as all Person-instances
             have their own rng.
       Each instance will have two attributes for each PopulationRepositoryType item: the repository itself, and a list of the keys.
       For example, self._dynamicRiskFactorsRepository is the attribute that holds the repository with all models for predicting the risk factors
       and self._dynamicRiskFactors is a list that holds all those risk factors.

    Unit of people subject to treatment program over time.

    (WIP) THe basic idea is that this is a generic superclass which will manage a group of
    people. Tangible subclasses will be needed to actually assign assumptions for a given
    population. As it stands, this class doesn't do anything (other than potentially being
    useful for tests), because it isn't tied to tangible risk models. Ultimately it might
    turn into an abstract class...
    """
   
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

    def advance(self, years, treatmentStrategies=None, nWorkers=1):
        if nWorkers==1:
            self.advance_serial(years, treatmentStrategies=treatmentStrategies)
        elif nWorkers>1:
            self.advance_parallel(years, treatmentStrategies=treatmentStrategies, nWorkers=nWorkers)
        else:
            print(f"Invalid nWorkers={nWorkers} argument provided.")

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

    #Q: I think I need this for starmap
    def worker_advance(self, subPopulation, years, treatmentStrategies):
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
        peopleParts = np.array_split(self._people, nPieces)
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
                                    self._modelRepository[PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value])

    @staticmethod
    def get_people_copy(people):
        """The Person __deepcopy__ function assumes that the Person object has not been advanced to the future at all."""
        return pd.Series( list(map( lambda x: x.__deepcopy__(), people)) )

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
            blocks[cat] = pd.Series(list(filter(lambda x: (getattr(x,"_"+blockFactor)[-1]>blockBounds[cat]) &
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

    def get_age_counts(self, itemList):
        counts = dict()
        for item in range(int(min(itemList)),int(max(itemList)+1)):
            counts[item] = len(list(filter(lambda x: x==item, itemList)))
        return counts

    def get_age_at_first_outcome(self, outcomeType, inSim=True):
        #we get None from Person objects that had no outcome
        ages = list(map(lambda x: x.get_age_at_first_outcome(outcomeType, inSim=inSim), self._people))
        #remove Nones 
        ages = list(filter(lambda x: x is not None,ages))
        return ages

    def get_min_age_of_first_outcomes(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.get_min_age_of_first_outcomes(outcomeTypeList, inSim=inSim), self._people))

    def get_min_wave_of_first_outcomes(self, outcomesTypeList=[OutcomeType.STROKE]):
        return list(map(lambda x: x.get_min_wave_of_first_outcomes(outcomesTypeList=outcomesTypeList), self._people))

    def get_min_age_of_first_outcomes_or_last_age(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.get_min_age_of_first_outcomes_or_last_age(outcomeTypeList, inSim=inSim), self._people))

    def get_min_wave_of_first_outcomes_or_last_wave(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.get_min_wave_of_first_outcomes_or_last_wave(outcomeTypeList, inSim=inSim), self._people))

    def get_age_of_all_years_in_sim(self):
        ages = list(map(lambda x: getattr(x, "_"+DynamicRiskFactorsType.AGE.value), self._people))
        ages = [x for sublist in ages for x in sublist]
        return ages

    def get_raw_incidence_by_age(self, outcomeType):
        outcomeIncidenceAges = self.get_age_at_first_outcome(outcomeType)
        ages = self.get_age_of_all_years_in_sim()
        outcomeCounts = self.get_age_counts(outcomeIncidenceAges)
        ageCounts = self.get_age_counts(ages)
        outcomeIncidenceRate = dict()
        for age in ageCounts.keys():
            #second conditional avoids division by 0 
            if (age in outcomeCounts.keys()) & (ageCounts[age]!=0):
                outcomeIncidenceRate[age] = outcomeCounts[age]/ageCounts[age]
            else:
                outcomeIncidenceRate[age] = 0
        return outcomeIncidenceRate 

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
                #keep age groups with age 18 and older
                ageGroups[gender.value] = list(filter(lambda x: any(map(lambda age: age>=18, x)), standardizedPop.ageGroups[gender.value]))
                #keep the corresponding population percents
                populationPercents[gender.value] = standardizedPop.populationPercents[gender.value][-len(ageGroups[gender.value]):]
                #rescale the population percents
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

    def get_outcome_risk(self, outcomeType):
        return sum(list(map(lambda x: x.has_outcome_during_simulation(outcomeType), self._people)))/self._n

    def get_any_outcome_risk(self, outcomeTypeList):
        return sum(list(map(lambda x: x.has_any_outcome(outcomeTypeList, inSim=True), self._people)))/self._n

    def get_outcome_count(self, outcomeType):
        return sum(self.has_outcome(outcomeType))

    def has_outcome(self, outcomeType, inSim=True):
        return list(map(lambda x: x.has_outcome(outcomeType, inSim=inSim), self._people))

    def has_any_outcome(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.has_any_outcome(outcomeTypeList, inSim=inSim), self._people))

    def has_any_outcome_by_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=0):
        return list(map(lambda x: x.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave), self._people))

    def has_all_outcomes(self, outcomeTypeList, inSim=True):
        return list(map(lambda x: x.has_all_outcomes(outcomeTypeList, inSim=inSim), self._people))

    def has_cognitive_impairment(self):
        return list(map(lambda x: x.has_cognitive_impairment(), self._people))

    def has_ci(self):
        return self.has_cognitive_impairment()

    def get_outcome_item_last(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_last(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_first(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_first(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_sum(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_sum(outcomeType, phenotypeItem, inSim=inSim), self._people))

    def get_outcome_item_mean(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_mean(outcomeType, phenotypeItem, inSim=inSim), self._people))
 
    def get_outcome_item_overall_change(self, outcomeType, phenotypeItem, inSim=True):
        return list(map(lambda x: x.get_outcome_item_overall_change(outcomeType, phenotypeItem, inSim=inSim), self._people))

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

    # refactorrtag: we should probably build a specific class that loads data files...

    def calculate_mean_age_sex_standardized_mortality(self, yearOfStandardizedPopulation=2016):
        events = self.calculate_mean_age_sex_standardized_event(
            lambda x: x.is_dead(), lambda x: x.years_in_simulation(), yearOfStandardizedPopulation
        )
        return pd.Series([event[0] for event in events]).mean()

    def get_all_person_years_as_df(self):
        """This function creates a dataframe where each row is a person-year from the simulation.
           Thus a single person object will be represented in N rows in this dataframe where N is the 
           number of years this person object lived in the simulation."""
    
        srfList = list(self._modelRepository["staticRiskFactors"]._repository.keys())
        drfList = list(self._modelRepository["dynamicRiskFactors"]._repository.keys())
        dtList = list(self._modelRepository["defaultTreatments"]._repository.keys())
        columnNames = ["name"] + srfList + drfList + dtList
        nestedList = list(map(lambda x: 
                          list(zip(*[
                              *[[getattr(x, "_"+"name")]*(x._waveCompleted+1)],
                              *[[getattr(x, "_"+attr)]*(x._waveCompleted+1) for attr in srfList],
                              *[getattr(x,"_"+attr) for attr in drfList],
                              *[getattr(x,"_"+attr) for attr in dtList]])), 
                          self._people))
        df = pd.concat([pd.DataFrame(nestedList[i], columns=columnNames) for i in range(len(nestedList))], ignore_index=True)
        return df

    def get_baseline_attr(self, rf):
        return get_attr_at_index(self, rf, 0)

    def get_last_attr(self, rf):
        return get_attr_at_index(self, rf, -1)

    def get_attr_at_index(self, rf, index):
        '''Returns a list of the baseline attributes of Person objects that were alive at baseline.'''
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

    def get_outcome_survival_info(self, outcomesTypeList=[OutcomeType.STROKE], personFunctionsList=[lambda x: x.get_scd_group()]):
        '''Returns a nested list, a list of lists: each sublist corresponds to a single person in the population.
        Each sublist includes information related to survival analysis, time to either censoring or outcome, and desired covariates.
        Currently, the person get_outcome_survival_info function tests if the person object has any of the outcomes provided in the list.
        Covariates are include via the personFunctionsList argument, the list must include pure functions that can be applied to a person object.'''
        return list(map(lambda x: x.get_outcome_survival_info(outcomesTypeList=outcomesTypeList, personFunctionsList=personFunctionsList), self._people))

    def get_outcome_incidence_rates_at_end_of_wave(self, outcomesTypeList=[OutcomeType.STROKE], wave=3):
        '''Returns outcome incidence rate per 1000 person-years at the end of the wave argument.
        Need to be careful with wave: wave=0 is the first wave, so set the wave to be number of years you want - 1.'''
        if wave<0:
            raise RuntimeError(f"wave {wave=} cannot be a negative number")
        if self._waveCompleted < wave:
            raise RuntimeError(f"Population has not advanced enough to reach end of {wave=}")
        #determine if each person in the population had any of the outcomes
        anyOutcome = self.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave) #[False,True,False,False,True,...]
        #convert to integer eg [0,0,1,1,0,...1,0]
        anyOutcome = list(map(lambda y: int(y), anyOutcome))
        #get the number of years each person in the population was at risk
        waves = self.get_min_wave_of_first_outcomes_or_last_wave(outcomesTypeList) #[5,1,6,8,0,...]
        personYearsAtRisk = list(map(lambda x: min(x, wave), waves)) #with wave=3 [3,1,3,3,0,..]
        personYearsAtRisk = list(map(lambda y: y+1, personYearsAtRisk))
        popSize = len(anyOutcome) #how many people are part of the SCD and Modality group
        outcomeCounts = sum(anyOutcome) if popSize>0 else 0 #how many people had any of the outcomes
        rate = 1000. * outcomeCounts / sum(personYearsAtRisk)
        return rate

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
        #determine if each person in the population had any of the outcomes
        anyOutcome = self.has_any_outcome_by_end_of_wave(outcomesTypeList=outcomesTypeList, wave=wave) #[False,True,False,False,True,...]
        #get the number of years each person in the population was at risk
        waves = self.get_min_wave_of_first_outcomes_or_last_wave(outcomesTypeList) #[5,1,6,8,0,...]
        personYearsAtRisk = list(map(lambda x: min(x, wave), waves)) #with wave=3 [3,1,3,3,0,..]
        #get the SCD by modality group number for each person in the population
        group = self.get_scd_by_modality_group()
        rates = dict() #store rates in a dictionary
        for i in set(group): 
            #keep anyOutcome for that group only and convert to integer eg [0,0,1,1,0,...1,0]
            anyOutcomeForGroup = list(map(lambda y: int(y[1]), filter(lambda x: x[0]==i, zip(group,anyOutcome))))
            #keep the at risk person years for that group only
            personYearsAtRiskForGroup = list(map(lambda y: y[1]+1, filter(lambda x: x[0]==i, zip(group,personYearsAtRisk))))
            groupSize = len(anyOutcomeForGroup) #how many people are part of the SCD and Modality group
            groupOutcomeCounts = sum(anyOutcomeForGroup) if groupSize>0 else 0 #how many people from the group had any of the outcomes
            rates[i] = 1000. * sum(anyOutcomeForGroup) / sum(personYearsAtRiskForGroup)
        return rates

    def get_scd_by_modality_group(self):
        return list(map(lambda x: x.get_scd_by_modality_group(), self._people))

    def print_baseline_summary(self):
        self.print_summary_at_index(0)

    def print_lastyear_summary(self):
        self.print_summary_at_index(-1) 

    def print_summary_at_index(self, index):
        """Prints a summary of both static and dynamic risk factors at index (baseline: index=0, last year: index=-1."""
        print(" "*25, "Printing a summary of risk factors and default treatments...")
        print(" "*25, "min", " "*4, "0.25", " "*2, "med", " "*3, "0.75", " "*3, "max" , " "*2, "mean", " "*3, "sd")
        print(" "*25, "-"*53)
        #this is not a great solution...but...I need to be able to run this function without having to initialize a population
        #but I also need to run this function within a population and right now we are about to have two types of population...
        dynamicRiskFactors = self._dynamicRiskFactors
        for i,rf in enumerate(dynamicRiskFactors):
            if rf in [crf.value for crf in ContinuousRiskFactorsType]:
                rfList = Population.get_people_attr_at_index(self._people, rf, index)
                print(f"{rf:>23} {np.min(rfList):> 7.1f} {np.quantile(rfList, 0.25):> 7.1f} {np.quantile(rfList, 0.5):> 7.1f} {np.quantile(rfList, 0.75):> 7.1f} {np.max(rfList):> 7.1f} {np.mean(rfList):> 7.1f} {np.std(rfList):> 7.1f}")
        defaultTreatments = self._defaultTreatments
        for dt in defaultTreatments:
            if dt in [cdt.value for cdt in ContinuousDefaultTreatmentsType]:
                dtList = Population.get_people_attr_at_index(self._people, dt, index)
                print(f"{dt:>23} {np.min(dtList):> 7.1f} {np.quantile(dtList, 0.25):> 7.1f} {np.quantile(dtList, 0.5):> 7.1f} {np.quantile(dtList, 0.75):> 7.1f} {np.max(dtList):> 7.1f} {np.mean(dtList):> 7.1f} {np.std(dtList):> 7.1f}")
        print(" "*25, "proportions")
        print(" "*25, "-"*11)
        for i,rf in enumerate(dynamicRiskFactors):
            if rf in [crf.value for crf in CategoricalRiskFactorsType]:
                rfList = Population.get_people_attr_at_index(self._people, rf, index)
                print(f"{rf:>23}")
                rfValueCounts = Counter(rfList)
                for key in sorted(rfValueCounts.keys()):
                    print(f"{key:>23} {rfValueCounts[key]/len(rfList): 6.2f}")
        staticRiskFactors = self._staticRiskFactors
        for rf in staticRiskFactors:
            print(f"{rf:>23}")
            rfList = Population.get_people_attr_static(self._people, rf, index)
            rfValueCounts = Counter(rfList)
            for key in sorted(rfValueCounts.keys()):
                print(f"{key:>23} {rfValueCounts[key]/len(rfList): 6.2f}")
        for dt in defaultTreatments:
            if dt in [cdt.value for cdt in CategoricalDefaultTreatmentsType]:
                print(f"{dt:>23}")
                dtList = Population.get_people_attr_at_index(self._people, dt, index)
                dtValueCounts = Counter(dtList)
                for key in sorted(dtValueCounts.keys()):
                    print(f"{key:>23} {dtValueCounts[key]/len(dtList): 6.2f}")  


    def print_baseline_summary_comparison(self, other):
        self.print_summary_at_index_comparison(other, 0)

    def print_lastyear_summary_comparison(self, other):
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
        print(" "*25, "self", " "*50,  "other")
        print(" "*25, "-"*53, " ", "-"*53)
        print(" "*25, "min", " "*4, "0.25", " "*2, "med", " "*3, "0.75", " "*3, "max" , " "*2, "mean", " "*3, "sd", "    min ", "   0.25", " "*2, "med", " "*3, "0.75", " "*3, "max", " "*2, "mean", " "*3, "sd")
        print(" "*25, "-"*53, " ", "-"*53)
        #this is not a great solution...but...I need to be able to run this function without having to initialize a population
        #but I also need to run this function within a population and right now we are about to have two types of population...
        dynamicRiskFactors = people.iloc[0]._dynamicRiskFactors
        for i,rf in enumerate(dynamicRiskFactors):
            if rf in [crf.value for crf in ContinuousRiskFactorsType]:
                #rfList = self.get_attr_at_index(rf, index)
                rfList = Population.get_people_attr_at_index(people, rf, index)
                #rfListOther = other.get_attr_at_index(rf, index)
                rfListOther = Population.get_people_attr_at_index(other, rf, index)
                print(f"{rf:>23} {np.min(rfList):> 7.1f} {np.quantile(rfList, 0.25):> 7.1f} {np.quantile(rfList, 0.5):> 7.1f} {np.quantile(rfList, 0.75):> 7.1f} {np.max(rfList):> 7.1f} {np.mean(rfList):> 7.1f} {np.std(rfList):> 7.1f} {np.min(rfListOther):> 7.1f} {np.quantile(rfListOther, 0.25):> 7.1f} {np.quantile(rfListOther, 0.5):> 7.1f} {np.quantile(rfListOther, 0.75):> 7.1f} {np.max(rfListOther):> 7.1f} {np.mean(rfListOther):> 7.1f} {np.std(rfListOther):> 7.1f}")
        defaultTreatments = people.iloc[0]._defaultTreatments
        for dt in defaultTreatments:
            if dt in [cdt.value for cdt in ContinuousDefaultTreatmentsType]:
                dtList = Population.get_people_attr_at_index(people, dt, index)
                dtListOther = Population.get_people_attr_at_index(other, dt, index)
                print(f"{dt:>23} {np.min(dtList):> 7.1f} {np.quantile(dtList, 0.25):> 7.1f} {np.quantile(dtList, 0.5):> 7.1f} {np.quantile(dtList, 0.75):> 7.1f} {np.max(dtList):> 7.1f} {np.mean(dtList):> 7.1f} {np.std(dtList):> 7.1f} {np.min(dtListOther):> 7.1f} {np.quantile(dtListOther, 0.25):> 7.1f} {np.quantile(dtListOther, 0.5):> 7.1f} {np.quantile(dtListOther, 0.75):> 7.1f} {np.max(dtListOther):> 7.1f} {np.mean(dtListOther):> 7.1f} {np.std(dtListOther):> 7.1f}")
        print(" "*25, "self", "  other")
        print(" "*25, "proportions")
        print(" "*25, "-"*11)
        for i,rf in enumerate(dynamicRiskFactors):
            if rf in [crf.value for crf in CategoricalRiskFactorsType]:
                #rfList = self.get_attr_at_index(rf, index)
                rfList = Population.get_people_attr_at_index(people, rf, index)
                #rfListOther = other.get_attr_at_index(rf, index)
                rfListOther = Population.get_people_attr_at_index(other, rf, index)
                print(f"{rf:>23}")
                rfValueCounts = Counter(rfList)
                rfValueCountsOther = Counter(rfListOther)
                for key in sorted(rfValueCounts.keys()):
                    print(f"{key:>23} {rfValueCounts[key]/len(rfList): 6.2f} {rfValueCountsOther[key]/len(rfListOther): 6.2f}")
        staticRiskFactors = people.iloc[0]._staticRiskFactors
        for rf in staticRiskFactors:
            print(f"{rf:>23}")
            #rfList = list(map( lambda x: getattr(x, "_"+rf.value), self._people))
            #rfList = list(map( lambda x: getattr(x, "_"+rf), people))
            rfList = Population.get_people_attr_static(people, rf, index)
            rfValueCounts = Counter(rfList)
            #rfListOther = list(map( lambda x: getattr(x, "_"+rf.value), other._people))
            #rfListOther = list(map( lambda x: getattr(x, "_"+rf), other))
            rfListOther = Population.get_people_attr_static(other, rf, index)
            rfValueCountsOther = Counter(rfListOther)
            for key in sorted(rfValueCounts.keys()):
                print(f"{key:>23} {rfValueCounts[key]/len(rfList): 6.2f} {rfValueCountsOther[key]/len(rfListOther): 6.2f}")
        for dt in defaultTreatments:
            if dt in [cdt.value for cdt in CategoricalDefaultTreatmentsType]:
                print(f"{dt:>23}")
                dtList = Population.get_people_attr_at_index(people, dt, index)
                dtListOther = Population.get_people_attr_at_index(other, dt, index)
                dtValueCounts = Counter(dtList)
                dtValueCountsOther = Counter(dtListOther)
                for key in sorted(dtValueCounts.keys()):
                    print(f"{key:>23} {dtValueCounts[key]/len(dtList): 6.2f} {dtValueCountsOther[key]/len(dtListOther): 6.2f}")

    def print_lastyear_treatment_strategy_distributions(self):
        '''Prints distributional information about treatment strategy variables, such as bpMedsAdded, statinsAdded,
        but only for the people of the population that are still alive.'''
        #at this point there is no continuous treatment-related variable...so uncomment when there is one...
        #print(" "*25, "self")
        #print(" "*25, "-"*53)
        #print(" "*25, "min", " "*4, "0.25", " "*2, "med", " "*3, "0.75", " "*3, "max" , " "*2, "mean", " "*3, "sd")
        #print(" "*25, "-"*53)
        treatmentStrategies = self._people.iloc[0]._treatmentStrategies.keys()
        #for ts in treatmentStrategies:
        #    tsVariables = self._people.iloc[0]._treatmentStrategies[ts].keys()
        #    for tsv in tsVariables:
        #        if (tsv in [ctst.value for ctst in ContinuousTreatmentStrategiesType]) & (tsv!="status"):
        #            tsvList = list(map(lambda x: x._treatmentStrategies[ts][tsv], self._people))
        #            print(f"{tsv:>23} {np.min(tsvList):> 7.1f} {np.quantile(tsvList, 0.25):> 7.1f} {np.quantile(tsvList, 0.5):> 7.1f} {np.quantile(tsvList, 0.75):> 7.1f} {np.max(tsvList):> 7.1f} {np.mean(tsvList):> 7.1f} {np.std(tsvList):> 7.1f}")
        print(" "*25, "self")
        print(" "*25, "-"*53)
        print(" "*25, "proportions")
        print(" "*25, "-"*11)
        for ts in treatmentStrategies:
            tsVariables = self._people.iloc[0]._treatmentStrategies[ts].keys()
            for tsv in tsVariables:
                if (tsv in [ctst.value for ctst in CategoricalTreatmentStrategiesType]) & (tsv!="status"):
                    alivePeople = filter(lambda x: x.is_alive, self._people)
                    tsvList = list(map(lambda x: x._treatmentStrategies[ts][tsv], alivePeople))
                    print(f"{tsv:>23}")
                    tsvValueCounts = Counter(tsvList)
                    for key in sorted(tsvValueCounts.keys()):
                        print(f"{key:>23} {tsvValueCounts[key]/len(tsvList): 6.2f}")

    def print_cv_standardized_rates(self):
        outcomes = [OutcomeType.MI, OutcomeType.STROKE, OutcomeType.DEATH,
                    OutcomeType.CARDIOVASCULAR, OutcomeType.NONCARDIOVASCULAR, OutcomeType.DEMENTIA]
        standardizedRates = list(map(lambda x: self.calculate_mean_age_sex_standardized_incidence(x, 2016), outcomes))
        standardizedRatesBlack = list(map(
                                  lambda x: self.calculate_mean_age_sex_standardized_incidence(x,2016, lambda y: y._black),
                                  outcomes))
        standardizedRatesWhite = list(map(
                                  lambda x: self.calculate_mean_age_sex_standardized_incidence(x,2016, lambda y: y._white),
                                  outcomes))
        print("standardized rates (per 100,000)    all        black      white")
        for i in range(len(outcomes)):
            print(f"{outcomes[i].value:>30} {standardizedRates[i]:> 10.1f} {standardizedRatesBlack[i]:> 10.1f} {standardizedRatesWhite[i]:> 10.1f}")

    def print_outcome_incidence(self, path=None, outcomeType=OutcomeType.DEMENTIA):
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
 
    def print_vascular_rfs_over_time(self, other, path=None):
        '''This function takes a population and analyzes the distribution of its risk factors.
           These distributions are then compared to the same distributions from other. 
           other: a Population instance.
           tmpDir: a complete path, if the user wants to save the plots instead of displaying them.'''
        dynamicRiskFactors = self._dynamicRiskFactors
        nRows = round(len(dynamicRiskFactors)/2)
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
        plt.suptitle("probability densities for all dynamic risk factors")
        #plt.subplots_adjust(wspace=0.5, hspace=0.7)
        plt.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path+"/probabilities-for-all-rf.png")
            plt.clf()
            print("exported results as PNG figures")
        self.print_lastyear_summary_comparison(other)
 
    def print_scd_cv_risk_proportions_table(self):
        '''Prints a table of proportions where the columns are CV risks without taking into account SCD specific information, such as WMH, SBI,
        and the rows are CV risks that include SCD specific information.'''
        alive = filter(lambda x: x.is_alive, self._people) #just an iterator
        cvNonScdSpecificRiskList = list(map(lambda x: CVModelRepository(wmhSpecific=False).select_outcome_model_for_person(x).get_risk_for_person(x, years=10),
                                            alive))
    
        alive = filter(lambda x: x.is_alive, self._people) #iterator again
        cvRiskList = list(map(lambda x: CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x, years=10), alive))    
    
        binEdges = np.array([0.   , 0.05 , 0.075, 0.1  , 0.125, 0.15 , 1.001]) #use meaningful bins
        personCounts, xEdgesActual, yEdgesActual = np.histogram2d(cvRiskList, cvNonScdSpecificRiskList,  bins=[binEdges,binEdges])
    
        risks = ["0.0-0.05", "0.05-0.075", "0.075-0.1", "0.1-0.125", "0.125-0.15", "0.15-1.0"]
    
        print(" "*25, "-"*53)
        print(" "*25, "proportion of people in risk bins")
        print(" "*25, "-"*53)
        print(" "*25, "CV risk (non-SCD specific)")
        print(" "*2, "CV risk (SCD specific) " + " ".join(risks)) #     0       1       2       3       4 ") 
        for i,row in enumerate(np.flip(personCounts, axis=0)/personCounts.sum()):
            printString = f"{risks[-i-1]:>23} "
            for item in row:
                printString += f"{item:> 9.2f} " 
            print(printString)
    
        print(" "*25, "-"*53)
        print(" "*25, "cumulative (column-wise) proportion of people in risk bins")
        print(" "*25, "-"*53)
        print(" "*25, "CV risk (non-SCD specific)")
        print(" "*2, "CV risk (SCD specific) " + " ".join(risks)) #     0       1       2       3       4 ") 
        for i,row in enumerate(np.flip(personCounts, axis=0).cumsum(axis=0)/personCounts.sum()):  #column wise
        #for i,row in enumerate(np.flip(personCounts, axis=0).cumsum(axis=0).cumsum(axis=1)/personCounts.sum()): #from top left to bottom right
            printString = f"{risks[-i-1]:>23} "
            for item in row:
                printString += f"{item:> 9.2f} " 
            print(printString) 

        print(" "*25, "-"*53)
        print(" "*25, "cumulative (row-wise) proportion of people in risk bins")
        print(" "*25, "-"*53)
        print(" "*25, "CV risk (non-SCD specific)")
        print(" "*2, "CV risk (SCD specific) " + " ".join(risks)) #     0       1       2       3       4 ") 
        for i,row in enumerate(np.flip(personCounts, axis=0).cumsum(axis=1)/personCounts.sum()):  #row wise
        #for i,row in enumerate(np.flip(personCounts, axis=0).cumsum(axis=0).cumsum(axis=1)/personCounts.sum()): #from top left to bottom right
            printString = f"{risks[-i-1]:>23} "
            for item in row:
                printString += f"{item:> 9.2f} " 
            print(printString)


    def print_wmh_outcome_summary(self):

        severityList = list(map(lambda x: x._outcomes[OutcomeType.WMH][0][1].wmhSeverity, self._people))
        severityList = [y.value if y is not None else "unknown" for y in severityList]
        sbiList =  list(map(lambda x: x._outcomes[OutcomeType.WMH][0][1].sbi, self._people))
        print("\n")
        print(" "*25, "Printing a summary of the WMH outcome...")
        print(" "*16, "severity proportion")
        print(" "*25, "-"*16)
        for severity in WMHSeverity:
            print(f"{severity.value:>23} {sum([x==severity.value for x in severityList])/self._n:>6.2f}")
        print(" "*15, f"unknown {sum([x=='unknown' for x in severityList])/self._n:>6.2f}\n")
        print(" "*21, "SBI proportion")
        print(" "*25, "-"*16)
        print(" "*18,f"TRUE {sum(sbiList)/self._n:>6.2f}")








