import copy
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from collections import Counter

from microsim.alcohol_category import AlcoholCategory
from microsim.bp_treatment_strategies import *
from microsim.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository, 
                                                   CohortStaticRiskFactorModelRepository,
                                                   CohortDefaultTreatmentModelRepository)
from microsim.cv_outcome_determination import CVOutcomeDetermination
from microsim.data_loader import (get_absolute_datafile_path,
                                  load_regression_model)
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.gfr_equation import GFREquation
from microsim.initialization_repository import InitializationRepository
from microsim.nhanes_risk_model_repository import NHANESRiskModelRepository
from microsim.outcome import Outcome, OutcomeType
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.person import Person
from microsim.person_factory import PersonFactory, microsimToNhanes
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.statsmodel_logistic_risk_factor_model import \
    StatsModelLogisticRiskFactorModel
from microsim.sim_settings import simSettings
from microsim.stroke_outcome import StrokeOutcome
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType, CategoricalRiskFactorsType, ContinuousRiskFactorsType
from microsim.afib_model import AFibPrevalenceModel
from microsim.pvd_model import PVDPrevalenceModel
from microsim.treatment import DefaultTreatmentsType, TreatmentStrategiesType
from microsim.population_model_repository import PopulationRepositoryType, PopulationModelRepository
from microsim.standardized_population import StandardizedPopulation
from microsim.risk_model_repository import RiskModelRepository

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
    def get_unique_people_count(people):
        return len(set(list(map(lambda x: x._name, people))))

    def get_attr(self, attr):
        return list(map(lambda x: getattr(x, "_"+attr), self._people))

    def get_age_counts(self, itemList):
        counts = dict()
        for item in range(int(min(itemList)),int(max(itemList)+1)):
            counts[item] = len(list(filter(lambda x: x==item, itemList)))
        return counts

    def get_age_at_first_outcome(self, outcomeType):
        #we get None from Person objects that had no outcome
        ages = list(map(lambda x: x.get_age_at_first_outcome(outcomeType), self._people))
        #remove Nones 
        ages = list(filter(lambda x: x is not None,ages))
        return ages

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
            minAge[gender.value] = min(ages[gender.value])
            maxAge[gender.value] = max(ages[gender.value])
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

    def calculate_mean_age_sex_standardized_incidence(self, outcomeType, year=2016, personFilter=None):
        """Calculates the gender and age standardized # of events pers 100,000 person years. """

        #standardized population age groups and percentages
        standardizedPop = StandardizedPopulation(year=2016)
        
        #get [ (gender, age), (gender, age),...] from simulation for all outcomes and do the counting
        outcomeGenderAge = self.get_gender_age_of_all_outcomes_in_sim(outcomeType, personFilter)
        outcomeCounts = self.get_gender_age_counts(outcomeGenderAge)

        #get [ (gender, age), (gender, age),...] from simulation for all persons and do the counting
        personGenderAge = self.get_gender_age_of_all_years_in_sim(personFilter)
        personYearCounts = self.get_gender_age_counts(personGenderAge)

        #the standardized population was in groups, so I need to group my simulation counts too....
        outcomeCountsGrouped = self.get_gender_age_counts_grouped(outcomeCounts,  standardizedPop.ageGroups)
        personYearCountsGrouped = self.get_gender_age_counts_grouped(personYearCounts, standardizedPop.ageGroups)

        #do the calculation
        outcomeRates = dict()
        expectedOutcomes = 0
        for gender in NHANESGender:
            outcomeRates[gender.value] = [(10**5)*x/y if y!=0 else 0 for x,y in zip(outcomeCountsGrouped[gender.value],
                                                                                    personYearCountsGrouped[gender.value])]
            expectedOutcomes += sum([x*y for x,y in zip(outcomeRates[gender.value],
                                                        standardizedPop.populationPercents[gender.value])])
        return expectedOutcomes

    def get_outcome_risk(self, outcomeType):
        return sum(list(map(lambda x: x.has_outcome_during_simulation(outcomeType), self._people)))/self._n

    def has_outcome(self, outcomeType):
        return list(map(lambda x: x.has_outcome(outcomeType), self._people))

    def has_any_outcome(self, outcomeTypeList):
        return list(map(lambda x: x.has_any_outcome(outcomeTypeList), self._people))

    def has_all_outcomes(self, outcomeTypeList):
        return list(map(lambda x: x.has_all_outcomes(outcomeTypeList), self._people))

    def has_cognitive_impairment(self):
        return list(map(lambda x: x.has_cognitive_impairment(), self._people))

    def has_ci(self):
        return self.has_cognitive_impairment()

    def get_outcome_item_last(self, outcomeType, phenotypeItem):
        return list(map(lambda x: x.get_outcome_item_last(outcomeType, phenotypeItem)))

    def get_outcome_item_first(self, outcomeType, phenotypeItem):
        return list(map(lambda x: x.get_outcome_item_first(outcomeType, phenotypeItem)))

    def get_outcome_item_sum(self, outcomeType, phenotypeItem):
        return list(map(lambda x: x.get_outcome_item_sum(outcomeType, phenotypeItem)))

    def get_outcome_item_mean(self, outcomeType, phenotypeItem):
        return list(map(lambda x: x.get_outcome_item_mean(outcomeType, phenotypeItem)))
 
    def get_outcome_item_overall_change(self, outcomeType, phenotypeItem):
        return list(map(lambda x: x.get_outcome_item_overall_change(outcomeType, phenotypeItem)))

    #def reset_to_baseline(self):
    #    self._totalWavesAdvanced = 0
    #    self._currentWave = 0
    #    self._bpTreatmentStrategy = None
    #    for person in self._people:
    #        person.reset_to_baseline()

    def set_bp_treatment_strategy(self, bpTreatmentStrategy):
        self._bpTreatmentStrategy = bpTreatmentStrategy
        for person in self._people:
            person._bpTreatmentStrategy = bpTreatmentStrategy

    #def get_people_alive_at_the_start_of_the_current_wave(self):
    #    return self.get_people_alive_at_the_start_of_wave(self._currentWave)

    #def get_people_alive_at_the_start_of_wave(self, wave):
    #    peopleAlive = []
    #    for person in self._people:
    #        if person.alive_at_start_of_wave(wave):
    #            peopleAlive.append(person)
    #    return pd.Series(peopleAlive)

    #def get_people_that_are_currently_alive(self):
    #    return pd.Series([not person.is_dead() for _, person in self._people.items()])

    #def get_number_of_patients_currently_alive(self):
    #    self.get_people_that_are_currently_alive().sum()

    #def get_events_in_most_recent_wave(self, eventType):
    #    peopleWithEvents = []
    #    for _, person in self._people.items():
    #        if person.has_outcome_at_age(eventType, person._age[-1]):
    #            peopleWithEvents.append(person)
    #    return peopleWithEvents

    #def generate_starting_mean_patient(self):
    #    df = self.get_people_initial_state_as_dataframe()
    #    return Person(
    #        age=int(round(df.age.mean())),
    #        gender=NHANESGender(df.gender.mode()),
    #        raceEthnicity=NHANESRaceEthnicity(df.raceEthnicity.mode()),
    #        sbp=df.sbp.mean(),
    #        dbp=df.dbp.mean(),
    #        a1c=df.a1c.mean(),
    #        hdl=df.hdl.mean(),
    #        totChol=df.totChol.mean(),
    #        bmi=df.bmi.mean(),
    #        ldl=df.ldl.mean(),
    #        trig=df.trig.mean(),
    #        waist=df.waist.mean(),
    #        anyPhysicalActivity=df.anyPhysicalActivity.mode(),
    #        education=Education(df.education.mode()),
    #        smokingStatus=SmokingStatus(df.smokingStatus.mode()),
    #        antiHypertensiveCount=int(round(df.antiHypetensiveCount().mean())),
    #        statin=df.statin.mode(),
    #        otherLipidLoweringMedicationCount=int(
    #            round(df.otherLipidLoweringMedicationCount.mean())
    #        ),
    #        initializeAfib=(lambda _: False),
    #        selfReportStrokeAge=None,
    #        selfReportMIAge=None,
    #        randomEffects=self._outcome_model_repository.get_random_effects(),
    #    )

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

    def get_person_attributes_from_person(self, person, timeVaryingCovariates):
        attrForPerson = person.get_current_state_as_dict()
        try:
            attrForPerson["populationIndex"] = person._populationIndex
        except AttributeError:
            pass  # populationIndex is not necessary for advancing; can continue safely without it

        timeVaryingAttrsForPerson = person.get_tvc_state_as_dict(timeVaryingCovariates)
        attrForPerson.update(timeVaryingAttrsForPerson)
        return attrForPerson

    def get_people_current_state_as_dataframe(self):
            timeVaryingCovariatesAndOutcomes = self._timeVaryingCovariates
            timeVaryingCovariatesAndOutcomes.append("gcp")
            return pd.DataFrame(
                list(
                    self.applyMethodSeries(self._people,
                        self.get_person_attributes_from_person,
                        timeVaryingCovariates=timeVaryingCovariatesAndOutcomes,
                    )
                )
            )

    def get_people_current_state_and_summary_as_dataframe(self):
        df = self.get_people_current_state_as_dataframe()
        # iterate through variables that vary over time
        tvcMeans = {}
        for var in self._timeVaryingCovariates:
            tvcMeans["mean" + var.capitalize()] = [
                pd.Series(getattr(person, "_" + var)).mean()
                for i, person in self._people.items()
            ]   
        df = pd.concat([df, pd.DataFrame(tvcMeans)], axis=1)
        #I thought I needed these, I no longer do
        #the GCP Stroke model needs the median of some quantities (will not get the medians for all TVCs for now) 
        #varMedians = {}
        #for var in ["bmi", "gcp", "waist"]:
        #    varMedians["median" + var.capitalize()] = [
        #        pd.Series(getattr(person, "_" + var)).median()
        #        for i, person in self._people.items()
        #    ]   
        #df = pd.concat([df, pd.DataFrame(varMedians)], axis=1)
        return df

    def get_people_initial_state_as_dataframe(self):
        return pd.DataFrame(
            {
                "age": [person._age[0] for person in self._people],
                "gender": [person._gender for person in self._people],
                "raceEthnicity": [person._raceEthnicity for person in self._people],
                "sbp": [person._sbp[0] for person in self._people],
                "dbp": [person._dbp[0] for person in self._people],
                "a1c": [person._a1c[0] for person in self._people],
                "hdl": [person._hdl[0] for person in self._people],
                "ldl": [person._ldl[0] for person in self._people],
                "trig": [person._trig[0] for person in self._people],
                "totChol": [person._totChol[0] for person in self._people],
                "creatinine": [person._creatinine[0] for person in self._people],
                "bmi": [person._bmi[0] for person in self._people],
                "anyPhysicalActivity": [person._anyPhysicalActivity[0] for person in self._people],
                "education": [person._education.value for person in self._people],
                "afib": [person._afib[0] for person in self._people],
                "antiHypertensiveCount": [
                    person._antiHypertensiveCount[0] for person in self._people
                ],
                "statin": [person._statin[0] for person in self._people],
                "otherLipidLoweringMedicationCount": [
                    person._otherLipidLoweringMedicationCount[0] for person in self._people
                ],
                "waist": [person._waist[0] for person in self._people],
                "smokingStatus": [person._smokingStatus for person in self._people],
                "miPriorToSim": [person._selfReportMIPriorToSim for person in self._people],
                "strokePriorToSim": [
                    person._selfReportStrokePriorToSim for person in self._people
                ],
                "totalQalys": [np.array(person._qalys).sum() for person in self._people],
                "totalBPMedsAdded" : [np.zeros(len(self._people))],
                "bpMedsAdded" : [np.zeros(len(self._people))]
            }
        )

    def get_summary_df(self):
        data = {}
        for year in range(1,self._currentWave+1):
            data[f'mi{year}'] = [x.has_mi_during_wave(year) for _, x in self._people.items()]
            data[f'stroke{year}'] = [x.has_stroke_during_wave(year) for _, x in self._people.items()]
            data[f'dead{year}'] = [x.dead_at_end_of_wave(year) for _, x in self._people.items()]
            data[f'dementia{year}'] = [x.has_outcome_during_wave(year, OutcomeType.DEMENTIA) for _, x in self._people.items()]
            data[f'gcp{year}'] = [np.nan if x.dead_at_start_of_wave(year) else x._gcp[year-1] for _, x in self._people.items()]
            data[f'sbp{year}'] = [np.nan if x.dead_at_start_of_wave(year) else x._sbp[year-1] for _, x in self._people.items()]
            data[f'dbp{year}'] = [np.nan if x.dead_at_start_of_wave(year) else x._dbp[year-1] for _, x in self._people.items()]  
            data[f'bpMeds{year}'] = [np.nan if x.dead_at_start_of_wave(year) else x._antiHypertensiveCount[year-1] for _, x in self._people.items()]
            data[f'bpMedsAdded{year}'] = [np.nan if x.dead_at_start_of_wave(year) else x._bpMedsAdded[year-1] for _, x in self._people.items()]
            data[f'totalBPMeds{year}'] = [np.nan if x.dead_at_start_of_wave(year) else x._bpMedsAdded[year-1]+x._antiHypertensiveCount[year-1] for _, x in self._people.items()]
            data[f'totalBPMedsAdded{year}'] = [np.array(x._bpMedsAdded).sum() for _, x in self._people.items()]

        data['baseAge'] = [x._age[0] for _, x in self._people.items()]
        data['id'] = [x._populationIndex  for _, x in self._people.items()] 
        data['nhanesIndex'] = [x.dfIndex  for _, x in self._people.items()] 
        data['finalAge'] = [x._age[-1]  for _, x in self._people.items()]
        data['education'] = [x._education  for _, x in self._people.items()]
        data['gender'] = [x._gender  for _, x in self._people.items()]
        data['raceEthnicity'] = [x._raceEthnicity  for _, x in self._people.items()]
        data['smokingStatus'] = [x._smokingStatus  for _, x in self._people.items()]

        data['baselineSBP'] = [x._sbp[0] for _, x in self._people.items()]
        data['baselineDBP'] = [x._dbp[0] for _, x in self._people.items()]
        data['black'] = [x._raceEthnicity==4 for _, x in self._people.items()]
        data['dementiaFreeYears'] = [x.get_age_at_first_outcome(OutcomeType.DEMENTIA) - x._age[0] if x._dementia else x._age[-1] - x._age[0]  for _, x in self._people.items()]
        data['deadAtEndOfSim'] = [x._alive[-1]==False for _, x in self._people.items()]
        return pd.DataFrame(data)

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
        #rfList = list(map( lambda x: getattr(x, "_"+rf.value)[index] if x.is_alive else None, self._people))
        rfList = list(map( lambda x: getattr(x, "_"+rf.value)[index] if x.is_alive else None, people))
        rfList = list(filter(lambda x: x is not None, rfList))
        rfList = list(map(lambda x: int(x) if (type(x)==bool)|(type(x)==np.bool_) else x, rfList))
        return rfList

    def print_baseline_summary(self):
        self.print_summary_at_index(0)

    def print_lastyear_summary(self):
        self.print_summary_at_index(-1) 

    def print_summary_at_index(self, index):
        """Prints a summary of both static and dynamic risk factors at index (baseline: index=0, last year: index=-1."""
        print(" "*50, "  min ", "  0.25", " "*1, "med", " "*2, "0.75", " "*2, "max", " "*1, "mean", " "*2, "sd")
        for i,rf in enumerate(DynamicRiskFactorsType):
            rfList = self.get_attr_at_index(rf, index)
            print(f"{rf.value:>50} {np.min(rfList):> 6.1f} {np.quantile(rfList, 0.25):> 6.1f} {np.quantile(rfList, 0.5):> 6.1f} {np.quantile(rfList, 0.75):> 6.1f} {np.max(rfList):> 6.1f} {np.mean(rfList):> 6.1f} {np.std(rfList):> 6.1f}")
        print(" "*50, "  proportions") 
        for rf in StaticRiskFactorsType:
            print(f"{rf.value:>50}")
            rfList = list(map( lambda x: getattr(x, "_"+rf.value), self._people))
            rfValueCounts = Counter(rfList)
            for key in sorted(rfValueCounts.keys()):
                print(f"{key.value:>50} {rfValueCounts[key]/self._n: 6.2f}")

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
        for i,rf in enumerate(DynamicRiskFactorsType):
            #rfList = self.get_attr_at_index(rf, index)
            rfList = Population.get_people_attr_at_index(people, rf, index)
            #rfListOther = other.get_attr_at_index(rf, index)
            rfListOther = Population.get_people_attr_at_index(other, rf, index)
            print(f"{rf.value:>23} {np.min(rfList):> 7.1f} {np.quantile(rfList, 0.25):> 7.1f} {np.quantile(rfList, 0.5):> 7.1f} {np.quantile(rfList, 0.75):> 7.1f} {np.max(rfList):> 7.1f} {np.mean(rfList):> 7.1f} {np.std(rfList):> 7.1f} {np.min(rfListOther):> 7.1f} {np.quantile(rfListOther, 0.25):> 7.1f} {np.quantile(rfListOther, 0.5):> 7.1f} {np.quantile(rfListOther, 0.75):> 7.1f} {np.max(rfListOther):> 7.1f} {np.mean(rfListOther):> 7.1f} {np.std(rfListOther):> 7.1f}")
        print(" "*25, "self", "  other")
        print(" "*25, "proportions")
        print(" "*25, "-"*11)
        for rf in StaticRiskFactorsType:
            print(f"{rf.value:>23}")
            #rfList = list(map( lambda x: getattr(x, "_"+rf.value), self._people))
            rfList = list(map( lambda x: getattr(x, "_"+rf.value), people))
            rfValueCounts = Counter(rfList)
            #rfListOther = list(map( lambda x: getattr(x, "_"+rf.value), other._people))
            rfListOther = list(map( lambda x: getattr(x, "_"+rf.value), other))
            rfValueCountsOther = Counter(rfListOther)
            for key in sorted(rfValueCounts.keys()):
                print(f"{key:>23} {rfValueCounts[key]/people.shape[0]: 6.2f} {rfValueCountsOther[key]/other.shape[0]: 6.2f}")

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

    def print_dementia_incidence(self):
        dementiaIncidentRate = self.get_raw_incidence_by_age(OutcomeType.DEMENTIA)
        for key in sorted(dementiaIncidentRate.keys()):
            print(f"{key:>50} {dementiaIncidentRate[key]: 6.3f}")
 
#NOTE: this class used to be a subclass of NHANESDirectSamplePopulation which is no longer in use...
#so we will need to update this class...and also see how it relates to the StandardizedPopulation class as well
class NHANESAgeStandardPopulation:
#class NHANESAgeStandardPopulation(NHANESDirectSamplePopulation):
    def __init__(self, n, year, rng=None):
        nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        weights = self.get_weights(year)
        weights["gender"] = weights["female"] + 1
        weights = pd.merge(nhanes, weights, how="left", on=["age", "gender"]).popWeight
        super().__init__(n=n, year=year, weights=weights, rng=rng)

    def get_weights(self, year):
        standard = self.build_age_standard(year)
        return self.get_population_weighted_standard(standard)

    def get_population_weighted_standard(self, standard):
        rows = []
        for age in range(1, 151):
            for female in range(0, 2):
                dfRow = standard.loc[
                    (age >= standard.lowerAgeBound)
                    & (age <= standard.upperAgeBound)
                    & (standard.female == female)
                ]
                upperAge = dfRow["upperAgeBound"].values[0]
                lowerAge = dfRow["lowerAgeBound"].values[0]
                totalPop = dfRow["standardPopulation"].values[0]
                rows.append(
                    {"age": age, "female": female, "pop": totalPop / (upperAge - lowerAge + 1)}
                )
        df = pd.DataFrame(rows)
        df["popWeight"] = df["pop"] / df["pop"].sum()
        return df


class ClonePopulation(Population):
    """Simple class to build a "Population" seeded by mulitple copies of the same person"""

    def __init__(self, person, n):
        self._outcome_model_repository = OutcomeModelRepository()
        self._qaly_assignment_strategy = QALYAssignmentStrategy()
        self._risk_model_repository = CohortRiskModelRepository()
        self._initialization_repository = InitializationRepository()
        self.n = n

        # trying to make sure that cloned peopel are setup the same way as people are
        # when sampled from NHANES
        clonePerson = build_person(pd.Series({'age' : person._age[0],
                                            'gender': int(person._gender),
                                            'raceEthnicity':int(person._raceEthnicity),
                                            'meanSBP' :person._sbp[0],
                                            'meanDBP' :person._dbp[0],
                                            'a1c':person._a1c[0],
                                            'hdl':person._hdl[0],
                                            'ldl':person._ldl[0],
                                            'trig':person._trig[0],
                                            'tot_chol':person._totChol[0],
                                            'bmi':person._bmi[0],
                                            'waist':person._waist[0],
                                            'anyPhysicalActivity':person._anyPhysicalActivity[0],
                                            'smokingStatus': int(person._smokingStatus),
                                            'alcoholPerWeek': person._alcoholPerWeek[0],
                                            'education' : int(person._education),
                                            'antiHypertensive': person._antiHypertensiveCount[0],
                                            'statin': person._statin[0],
                                            'otherLipidLowering' : person._otherLipidLoweringMedicationCount[0],
                                            'serumCreatinine' : person._creatinine[0],
                                            'selfReportStrokeAge' : -1, 
                                            'selfReportMIAge' : -1,
                                            'diedBy2015' : 0}), 
                                            self._outcome_model_repository, 
                                            randomEffects=person._randomEffects,
                                            rng=person._rng)

        # for factors that were initialized on the first person, we have to set them the same way on teh clones
        clonePerson._afib[0] = person._afib[0]
        initializers = self._initialization_repository.get_initializers()
        for initializerName, _ in initializers.items():
            fromAttr = getattr(person, initializerName)
            toAttr = getattr(clonePerson, initializerName)
            toAttr.clear()
            toAttr.extend(fromAttr)

        
        people = pd.Series([copy.deepcopy(clonePerson) for i in range(0, n)])

        #pandarallel.initialize(verbose=1)
        for i in range(0, len(people)):
            people.iloc[i]._populationIndex = i
        super().__init__(people)
