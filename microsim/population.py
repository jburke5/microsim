import copy
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
from pandarallel import pandarallel

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
from microsim.outcome_model_type import OutcomeModelType
from microsim.person import Person
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.statsmodel_logistic_risk_factor_model import \
    StatsModelLogisticRiskFactorModel
from microsim.sim_settings import simSettings
from microsim.stroke_outcome import StrokeOutcome
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.afib_model import AFibPrevalenceModel
from microsim.pvd_model import PVDPrevalenceModel
from microsim.treatment import DefaultTreatmentsType, TreatmentStrategiesType
from microsim.population_model_repository import PopulationRepositoryType, PopulationModelRepository
from microsim.standardized_population import StandardizedPopulation

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
        for repositoryType in PopulationRepositoryType:
            #set each repository as a Population-instance attribute
            setattr(self, "_"+repositoryType.value+"Repository", getattr(popModelRepository, "_"+repositoryType.value+"Repository"))
            #set the keys of each repository as a Population-instance attribute
            setattr(self, "_"+repositoryType.value, list(getattr(popModelRepository, "_"+repositoryType.value+"Repository")._repository.keys()))

    def advance(self, years, treatmentStrategies=None):
        list(map(lambda x: x.advance(years, 
                                     getattr(self,"_"+PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value+"Repository"),
                                     getattr(self,"_"+PopulationRepositoryType.DEFAULT_TREATMENTS.value+"Repository"),     
                                     getattr(self,"_"+PopulationRepositoryType.OUTCOMES.value+"Repository"), 
                                     treatmentStrategies),
                 self._people))
        #note: need to remember that each Person-instance will have their own _waveCompleted attribute, which may be different than the
        #      Population-level _waveCompleted attribute
        self._waveCompleted += years

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

    def reset_to_baseline(self):
        self._totalWavesAdvanced = 0
        self._currentWave = 0
        self._bpTreatmentStrategy = None
        for person in self._people:
            person.reset_to_baseline()

    # trying to work this out. if we do get it worked out, then we probably want to rebuild the person to use systematic data structrures
    # (i.e. static attributes, time-varying attributes)
    # also, will need to thikn about ways to make sure that the dataframe version of reality stays synced with teh "patient-based" version of reality
    # for now, will build a DF at the beginnign and then update the peopel at the end...
    def advance_vectorized(self, years, rng=None):
        #rng = np.random.default_rng(rng)
        # get dataframe of people...
        df = self.get_people_current_state_and_summary_as_dataframe()
        alive = df.loc[df.dead == False]
        # might not need this row...depends o n whethe we do an bulk update on people or an wave-abased update
        waveAtStartOfAdvance = self._currentWave

        for yearIndex in range(years):
            logging.info(f"processing year: {self._currentWave}")
            alive = alive.loc[alive.dead == False]
            # if everybody has died, break out of the loop, no need to keep moving forward
            if len(alive) == 0:
                break
            self._currentWave += 1

            ########################## RISK FACTORS 

            riskFactorsDict = {}
            # advance risk factors on *Next variables
            #import pdb; pdb.set_trace()
            for rf in self._riskFactors:
                # print(f"### Risk Factor: {rf}")
                riskFactorsDict[rf + "Next"] = self.applyMethod(alive,
                    self._risk_model_repository.get_model(rf).estimate_next_risk_vectorized,
                    axis="columns",
                    rng=rng, 
                )

            # bring *Next risk factors modifications to the dataframe
            alive = pd.concat([alive.reset_index(drop=True), pd.DataFrame(riskFactorsDict).reset_index(drop=True)], axis='columns', ignore_index=False)

            #now that new risk factors have been calculated, copy the updated *Next values for risk factors to the current (last) values, 
            #while leaving *Next values untouched for now, so all estimates of risks will utilize the last, and most up to date, values
            alive = self.move_people_df_riskFactors_forward(alive)
 
            ########################## RISKS WITH UPDATED RISK FACTORS AND BEFORE TREATMENT

            #now that we have the updated risk factors, but have not applied the treatment yet, obtain the untreated risks
            #untreated risks are used in recalibration of mi and stroke outcomes
            alive = self.estimate_risks(alive, "untreated")

            ########################## TREATMENT (WHICH INDIRECTLY MODIFIES RISK FACTORS)

            treatmentsDict = {}
            # advance treatment on *Next variables (antiHypertensiveCountNext, statinNext)
            for treatment in self._treatments:
                # print(f"### Treatment: {treatment}")
                treatmentsDict[treatment + "Next"] = alive.apply(
                    self._risk_model_repository.get_model(treatment).estimate_next_risk_vectorized,
                    axis="columns",
                    rng=rng,
                )

            # bring *Next treatment modifications to the dataframe
            alive = pd.concat([alive.reset_index(drop=True), pd.DataFrame(treatmentsDict).reset_index(drop=True)], axis='columns', ignore_index=False)
            
            # initialize BP medications (on *Next variables)
            # bp meds added in this wave are 0...(bpMedsAddedNext variable)
            # total bp meds are carried forward from the prior wave...(totalBPMedsAddedNext variable)
            medsData = pd.DataFrame({'bpMedsAddedNext' :  pd.Series(np.zeros(len(alive))), 
                                    'totalBPMedsAddedNext' : pd.Series(alive['totalBPMedsAdded'])})

            # bring preliminary *Next bp medications to the dataframe
            alive = pd.concat([alive.reset_index(drop=True), medsData], axis='columns', ignore_index=False)

            # use the BP treatment strategy to modify *Next values (sbpNext, dbpNext, bpMedsAddedNext, totalBPMedsAddedNext)
            if self._bpTreatmentStrategy is not None:
                alive = alive.apply(
                    self._bpTreatmentStrategy.get_changes_vectorized, axis="columns"
                )

            #now that treatment has been applied, copy the updated *Next values for treatment to the current (last) values, 
            alive = self.move_people_df_treatment_forward(alive) 
            #because treatment can potentially affect the *Next risk factors, we need to move these again now
            #while leaving *Next values untouched for now, so all estimates of risks will utilize the last, and most up to date, values
            alive = self.move_people_df_riskFactors_forward(alive)
            #now that we are completely done with changing risk factors, move their means forward too
            alive = self.move_people_df_riskFactorsMeans_forward(alive)

            ########################## RISKS WITH UPDATED RISK FACTORS AFTER TREATMENT

            #now that we have updated risk factors and applied the treatment, obtain the risks of this group to use in recalibration
            alive = self.estimate_risks(alive, "treated")

            ########################## CLINICAL OUTCOMES

            #the order of the outcomes potentially becomes important if outcomes depend on each other (eg when gcp outcome depends on stroke outcome)

            # add these variables here to speed up performance...better than adding one at a time
            # in the advance method...
            outcomeVars = {}
            # first, setup outcome variables
            for outcome in ["stroke", "mi", "dementia", "dead", "cvDeath", 'nonCVDeath']:
                outcomeVars[outcome + "Next"] = [False] * len(alive)
            outcomeVars["miFatal"] =[False] * len(alive)

            outcomeVars = StrokeOutcome.add_outcome_vars(outcomeVars, len(alive))

            #outcomeVars["qalyNext"] =np.zeros(len(alive))
            #outcomeVars["ageAtFirstDementia"] = [np.nan] * len(df)
            alive = pd.concat([alive.reset_index(drop=True), pd.DataFrame(outcomeVars) ], axis='columns')

            ############# NON-FATAL OUTCOMES

            # advance outcomes

            # calculate GCP and its slope
            gcp = {}
            gcp["gcpNext"] = alive.apply(
                self._outcome_model_repository.get_gcp_vectorized, axis="columns", rng=rng
            )
            gcp["gcpSlope"] = gcp['gcpNext'] - alive['gcp']
            alive.drop(columns=['gcpSlope'], inplace=True)
            alive = pd.concat([alive.reset_index(drop=True), pd.DataFrame(gcp)], axis='columns')

            # if the whole popluation has demeentia, nobody new can get dementia...            
            alive.dementia = alive.dementia.astype('bool_')
            if len(alive.loc[~alive.dementia]) > 0:
                newDementia = alive.loc[~alive.dementia].apply(
                    self._outcome_model_repository.get_dementia_vectorized, rng=rng, axis="columns"
                )
                alive["dementiaNext"] = newDementia
            else:
                alive["dementiaNext"] = np.repeat(False, len(alive))

            alive.loc[alive["dementiaNext"] == 1, "ageAtFirstDementia"] = alive.age

            ############# (POTENTIALLY) FATAL AND FATAL OUTCOMES

            # first determine if there is a cv event
            # find preliminary mi and stroke outcomes, these will be corrected in a bit with recalibration
            # background: outcome_model_repository is based on models originally designed at the person class level
            # background: the recalibration method is applied at the population class level because it addresses a population-level issue
            alive = alive.apply(
                self._outcome_model_repository.assign_cv_outcome_vectorized, axis="columns", rng=rng
            )

            #perform recalibration, we are essentially modifying mi and stroke outcomes we obtained just above 
            numberAliveBeforeRecal = len(alive)
            alive = self.apply_recalibration_standards(alive, rng=rng)
            if len(alive) != numberAliveBeforeRecal:
                raise Exception(
                    f"number alive: {len(alive)} not equal to alive before recal {numberAliveBeforeRecal}"
                )

            #now that we know with certainty all cv-based deaths, find the non-cv-based deaths and finalize the deadNext variable 
            alive["cvDeathNext"] = alive["deadNext"]
            alive["nonCVDeathNext"] = alive.apply(
                self._outcome_model_repository.assign_non_cv_mortality_vectorized, axis="columns", rng=rng
            )
            alive["deadNext"] = alive["nonCVDeathNext"] | alive["cvDeathNext"]

            ############# FUNCTIONS OF CLINICAL OUTCOMES

            # calculate and assign the QALYs
            qaly = {}
            qaly["qalyNext"] = alive.apply(
                QALYAssignmentStrategy().get_qalys_vectorized, axis="columns"
            )
            alive = pd.concat([alive.reset_index(drop=True), pd.DataFrame(qaly)], axis='columns')

            ############# OUTCOME UPDATES

            alive = self.move_people_df_outcomes_forward(alive)
            alive = self.move_people_df_outcomesMeans_forward(alive)

            ########################## UPDATES
            
            #if they survived until now, they are now 1 year older
            alive.loc[~alive.dead, "age"] = alive.age + 1

            self._totalWavesAdvanced += 1

            alive["totalYearsInSim"] = alive["totalYearsInSim"] + 1

            alive.apply(self.push_updates_back_to_people, axis="columns")

            ########################## DATAFRAME CLEAN-UP

            # for efficieicny, we could try to do this all at the end...but, its a bit cleanear  to do it wave by wave
            nextCols = [col for col in alive.columns if "Next" in col]
            alive.drop(columns=nextCols, inplace=True)
            fatalCols = [col for col in alive.columns if "Fatal" in col]
            alive.drop(columns=fatalCols, inplace=True)
            riskCols = ['treatedcombinedRisks', 'treatedstrokeProbabilities', 'treatedstrokeRisks', 'treatedmiRisks', 
                        'untreatedcombinedRisks', 'untreatedstrokeProbabilities', 'untreatedstrokeRisks', 'untreatedmiRisks']
            alive.drop(columns=riskCols, inplace=True)
        return alive, df

    def push_updates_back_to_people(self, x):
        updatedIndices = set()
        peopleSet = set()
        person = self._people.iloc[int(x.populationIndex)]
        if x.populationIndex in updatedIndices or person in peopleSet:
            raise Exception(f"Population index: {x.populationIndex} already updated")
        else:
            updatedIndices.add(x.populationIndex)
            peopleSet.add(person)
        return self.update_person(person, x)

    def update_person(self, person, x):
        if person.is_dead():
            raise Exception(f"Trying to update a dead person: {person}")
        for rf in self._riskFactors:
            attr = getattr(person, "_" + rf)
            attr.append(x[rf + str(self._currentWave)])

        for treatment in self._treatments:
            attr = getattr(person, "_" + treatment)
            attr.append(x[treatment + str(self._currentWave)])

        # advance outcomes - this will add CV eath
        for outcomeName, outcomeType in {
            "stroke": OutcomeType.STROKE,
            "mi": OutcomeType.MI,
            "dementia": OutcomeType.DEMENTIA,
        }.items():
            if x[outcomeName + "Next"]:
                fatal = False if outcomeName == "dementia" else x[outcomeName + "Fatal"]
                # only one dementia event per person
                if outcomeName == "dementia" and person._dementia:
                    break
                elif outcomeName == "stroke":
                    outcomePhenotypeColumns = [i + str(self._currentWave) for i in StrokeOutcome.phenotypeItems ]
                    outcomePhenotypeDict = dict(zip( StrokeOutcome.phenotypeItems, x.loc[outcomePhenotypeColumns].tolist() ))
                    person.add_outcome_event(Outcome(outcomeType, fatal, **outcomePhenotypeDict)) #Outcome.properties will hold the stroke phenotype as a dict
                    person._randomEffects["gcpStroke"] = x.gcpStrokeRandomEffect
                    person._randomEffects["gcpStrokeSlope"] = x.gcpStrokeSlopeRandomEffect
                else:
                    person.add_outcome_event(Outcome(outcomeType, fatal))

        person._gcp.append(x.gcp)
        person._qalys.append(x.qalyNext)
        person._bpMedsAdded.append(x.bpMedsAddedNext)

        # add non CV death to person objects
        if x.nonCVDeathNext:
            person._alive.append(False)

        # only advance age in survivors
        if not x.deadNext:
            person._age.append(person._age[-1] + 1)
            person._alive.append(True)
        return person

    def move_people_df_riskFactors_forward(self, df):  
        factorsToChange = copy.copy(self._riskFactors)

        newVariables = {} #will hold data from columns that need to be added to df

        for rf in factorsToChange:
            # the curent value is stored in the variable name
            df[rf] = df[rf + "Next"]
            if (rf + str(self._currentWave) in df.columns):       #if column already exists, eg if you have already updated risk factors once in this iteration
                df[rf + str(self._currentWave)] = df[rf + "Next"] #then update values in place 
            else:                                                 #if this is the first time in the advanced_vectorized iteration I am updating risk factors
                newVariables[rf + str(self._currentWave)] = df[rf + "Next"] #then I need to create this column 

        df["current_diabetes"] = df["a1c"] > 6.5
        df["gfr"] = df.apply(GFREquation().get_gfr_for_person_vectorized, axis="columns") #GFR calculation requires an updated age

        # assign ages for new events
        # df.loc[(df.ageAtFirstStroke.isnull()) & (df.strokeNext), 'ageAtFirstStroke'] = df.age
        # df.loc[(df.ageAtFirstMI.isnull()) & (df.miNext), 'ageAtFirstMI'] = df.age
        # df.loc[(df.ageAtFirstDementia.isnull()) & (df.dementiaNext), 'ageAtFirstDementia'] = df.age

        if (newVariables == {}):
            return df #I have not created any new columns, just modified existing df column data
        else:
            return pd.concat([df.reset_index(drop=True), pd.DataFrame(newVariables).reset_index(drop=True)], axis='columns', ignore_index=False)

    def move_people_df_riskFactorsMeans_forward(self, df):
        factorsToChange = copy.copy(self._riskFactors)

        for rf in factorsToChange:
            df["mean" + rf.capitalize()] = (
                df["mean" + rf.capitalize()] * (df["totalYearsInSim"] + 1) + df[rf + "Next"]
            ) / (df["totalYearsInSim"] + 2)

        # assign ages for new events
        # df.loc[(df.ageAtFirstStroke.isnull()) & (df.strokeNext), 'ageAtFirstStroke'] = df.age
        # df.loc[(df.ageAtFirstMI.isnull()) & (df.miNext), 'ageAtFirstMI'] = df.age
        # df.loc[(df.ageAtFirstDementia.isnull()) & (df.dementiaNext), 'ageAtFirstDementia'] = df.age

        #note: I used these when I thought I needed median bmi and waist, I think now the above for loop will do
        #with GCPStrokeModel we started including meanBmi so I need to update this as well
        #note: because bmi is updated prior to generating stroke outcomes I use range(0,self._currentWave) (compare this with the medianGcp update)
        #df["meanBmi"] = df.apply(lambda y: y[["bmi" + f"{x}" for x in range(0,self._currentWave)]].mean(), axis=1)
        #df["meanWaist"] = df.apply(lambda y: y[["waist" + f"{x}" for x in range(0,self._currentWave)]].mean(), axis=1)

        df["meanSbpSinceLastStroke"] = df.apply(lambda y: None if pd.isna(y.waveAtLastStroke) 
                                                               else y[["sbp" + f"{x}" for x in range(round(y.waveAtLastStroke),self._currentWave)]].mean(), axis=1)
        df["meanLdlSinceLastStroke"] = df.apply(lambda y: None if pd.isna(y.waveAtLastStroke) 
                                                               else y[["ldl" + f"{x}" for x in range(round(y.waveAtLastStroke),self._currentWave)]].mean(), axis=1)
        df["meanA1cSinceLastStroke"] = df.apply(lambda y: None if pd.isna(y.waveAtLastStroke) 
                                                               else y[["a1c" + f"{x}" for x in range(round(y.waveAtLastStroke),self._currentWave)]].mean(), axis=1)
        return df

    def move_people_df_treatment_forward(self, df):
        factorsToChange = copy.copy(self._treatments)
        
        newVariables = {}
        
        for rf in factorsToChange:
            # the curent value is stored in the variable name
            df[rf] = df[rf + "Next"]
            newVariables[rf + str(self._currentWave)] = df[rf + "Next"]
            df["mean" + rf.capitalize()] = (
                df["mean" + rf.capitalize()] * (df["totalYearsInSim"] + 1) + df[rf + "Next"]
            ) / (df["totalYearsInSim"] + 2)

        df["current_bp_treatment"] = df["antiHypertensiveCount"] >= 1
        df["bpMedsAdded"] = df["bpMedsAddedNext"]
        df["totalBPMedsAdded"] = df["totalBPMedsAddedNext"]
        
        # assign ages for new events
        # df.loc[(df.ageAtFirstStroke.isnull()) & (df.strokeNext), 'ageAtFirstStroke'] = df.age
        # df.loc[(df.ageAtFirstMI.isnull()) & (df.miNext), 'ageAtFirstMI'] = df.age
        # df.loc[(df.ageAtFirstDementia.isnull()) & (df.dementiaNext), 'ageAtFirstDementia'] = df.age

        return pd.concat([df.reset_index(drop=True), pd.DataFrame(newVariables).reset_index(drop=True)], axis='columns', ignore_index=False)

    def move_people_df_outcomes_forward(self, df): #includes clinical outcomes and function of clinical outcomes (QALYs)

        newVariables = {}

        for outcome in ["mi", "stroke"]:
            df[outcome + "InSim"] = df[outcome + "InSim"] | df[outcome + "Next"]
            df[outcome] = df[outcome + "InSim"] | df[outcome + "PriorToSim"]
            newVariables[outcome + str(self._currentWave)] = df[outcome + "Next"]
        for item in StrokeOutcome.phenotypeItems:
            newVariables[item + str(self._currentWave)] = df[item + "Next"]
        df["dead"] = df["dead"] | df["deadNext"]
        newVariables["dead" + str(self._currentWave)] = df["deadNext"]

        df["dementia"] = df["dementiaNext"] | df["dementia"]
        newVariables["gcp" + str(self._currentWave)] = df['gcpNext']
        df['gcp'] = df['gcpNext']
        df["totalQalys"] = df["totalQalys"] + df["qalyNext"]

        return pd.concat([df.reset_index(drop=True), pd.DataFrame(newVariables).reset_index(drop=True)], axis='columns', ignore_index=False)

    def move_people_df_outcomesMeans_forward(self, df):
        #the GCPStrokeModel needs only the updated median GCP
        #note: there is no gcp0 but baseGcp on the df 
        #note: because gcp is updated after we generate stroke outcomes I use range(1,self._currentWave+1) (compare this with the medianBmi update)
        df["meanGcp"] = df.apply(lambda y: y[["baseGcp"]+["gcp" + f"{x}" for x in range(1,self._currentWave+1)]].mean(), axis=1)
        return df

    def set_bp_treatment_strategy(self, bpTreatmentStrategy):
        self._bpTreatmentStrategy = bpTreatmentStrategy
        for person in self._people:
            person._bpTreatmentStrategy = bpTreatmentStrategy

    def apply_recalibration_standards(self, recalibration_df, rng=None):
        # treatment_standard is a dictionary of outcome types and effect sizees
        if self._bpTreatmentStrategy is not None:
            if self._bpTreatmentStrategy.get_treatment_recalibration_for_population() is not None:
                recalibration_df = self.recalibrate_bp_treatment(recalibration_df, rng=rng)
        return recalibration_df

    # should the estiamted treatment effect be based on the number of events in the population
    # (i.e. # events treated / # of events untreated)
    # of should it be based on teh predicted reisks
    # the problem with the first approach is that its going to depend a lot on small sample sizes...
    # and we don't necessarily want to take out that random error...that random error reflects
    # genuine uncertainty.
    # so, i thikn it should be based on the model-predicted risks...

    def recalibrate_bp_treatment(self, recalibration_df, rng=None):
        #logging.info(f"*** before recalibration, mi count: {recalibration_df.miNext.sum()}, stroke count: {recalibration_df.strokeNext.sum()}")
        treatment_outcome_standard = (
            self._bpTreatmentStrategy.get_treatment_recalibration_for_population()
        )

        #logging.info(f"######## BP meds After redo: {recalibration_df.totalBPMedsAddedNext.value_counts()}")
        totalBPMedsAddedCapped = recalibration_df['totalBPMedsAddedNext'].copy()
        totalBPMedsAddedCapped.loc[totalBPMedsAddedCapped >= BaseTreatmentStrategy.MAX_BP_MEDS] = BaseTreatmentStrategy.MAX_BP_MEDS
        #recalibration_df.loc[recalibration_df['totalBPMedsAddedNext'] >= BaseTreatmentStrategy.MAX_BP_MEDS, 'totalBPMedsAddedCapped'] = BaseTreatmentStrategy.MAX_BP_MEDS
        recalibrationVars = {"rolledBackEventType" : [None] * len(recalibration_df),
                            'totalBPMedsAddedCapped' : totalBPMedsAddedCapped}       
        recalibration_df = pd.concat([recalibration_df.reset_index(drop=True), pd.DataFrame(recalibrationVars).reset_index(drop=True)], axis='columns', ignore_index=False)
        
        #recalibration_df["rolledBackEventType"] = None
        # total meds added represents the total number of medication effects that we'll recalibrate for
        # it is the lesser of the total number of BP meds actually added (totalBpMedsAdded) or the max cap
        # so, if a treamtent strategy adds 10 medications, they'll effect the BP...but, they 
        # wont' have an additional efect on event reduction over the medication cap
        #logging.info(f"######## BP meds After redo: {recalibration_df.totalBPMedsAddedNext.value_counts()}")

        # recalibrate within each group of added medicaitons so that we can stratify the treamtnet effects
        for i in range(1, BaseTreatmentStrategy.MAX_BP_MEDS + 1):
            #logging.info(f"Roll back for med count: {i}")
            recalibrationPopForMedCount = recalibration_df.loc[recalibration_df.totalBPMedsAddedCapped == i]
            # the change standards are for a single medication
            recalibration_standard_for_med_count = treatment_outcome_standard.copy()
            for key, value in recalibration_standard_for_med_count.items():
                recalibration_standard_for_med_count[key] = value**i

            if len(recalibrationPopForMedCount) > 0:
                # recalibrate stroke
                recalibratedForMedCount = self.create_or_rollback_events_to_correct_calibration(
                    recalibration_standard_for_med_count,
                    "treatedstrokeRisks",
                    "untreatedstrokeRisks",
                    "stroke",
                    OutcomeType.STROKE,
                    CVOutcomeDetermination()._will_have_fatal_stroke,
                    recalibrationPopForMedCount,
                    rng=rng,
                )

                recalibration_df.loc[
                    recalibratedForMedCount.index, "strokeNext"
                ] = recalibratedForMedCount["strokeNext"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "strokeFatal"
                ] = recalibratedForMedCount["strokeFatal"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "deadNext"
                ] = recalibratedForMedCount["deadNext"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "ageAtFirstStroke"
                ] = recalibratedForMedCount["ageAtFirstStroke"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "rolledBackEventType"
                ] = recalibratedForMedCount["rolledBackEventType"]

                # recalibrate MI
                recalibratedForMedCount = self.create_or_rollback_events_to_correct_calibration(
                    recalibration_standard_for_med_count,
                    "treatedmiRisks",
                    "untreatedmiRisks",
                    "mi",
                    OutcomeType.MI,
                    CVOutcomeDetermination()._will_have_fatal_mi,
                    recalibrationPopForMedCount,
                    rng=rng,
                )
                recalibration_df.loc[
                    recalibratedForMedCount.index, "miNext"
                ] = recalibratedForMedCount["miNext"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "miFatal"
                ] = recalibratedForMedCount["miFatal"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "deadNext"
                ] = recalibratedForMedCount["deadNext"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "ageAtFirstMI"
                ] = recalibratedForMedCount["ageAtFirstMI"]
                recalibration_df.loc[
                    recalibratedForMedCount.index, "rolledBackEventType"
                ] = recalibratedForMedCount["rolledBackEventType"]

        #logging.info(f"*** after recalibration, mi count: {recalibration_df.miNext.sum()}, stroke count: {recalibration_df.strokeNext.sum()}")
        #recalibration_df.drop(columns=['treatedcombinedRisks', 'treatedstrokeProbabilities', 'treatedstrokeRisks', 'treatedmiRisks', 
         #           'untreatedcombinedRisks', 'untreatedstrokeProbabilities', 'untreatedstrokeRisks', 'untreatedmiRisks', 'totalBPMedsAddedCapped', 'rolledBackEventType'], inplace=True)
        recalibration_df.drop(columns=['totalBPMedsAddedCapped', 'rolledBackEventType'], inplace=True)
        return recalibration_df

    def estimate_risks(self, recalibration_df, prefix):
        combinedRisks = recalibration_df.apply(
            self._outcome_model_repository.get_risk_for_person_vectorized,
            axis="columns",
            args=(OutcomeModelType.CARDIOVASCULAR, 1),
        )
        strokeProbabilities = recalibration_df.apply(
            CVOutcomeDetermination().get_stroke_probability, axis="columns", vectorized=True
        )
        strokeRisks = (
            combinedRisks * strokeProbabilities
        )
        miRisks = combinedRisks * (
            1 - strokeProbabilities
        )

        risksAndProbs = pd.DataFrame({prefix + "combinedRisks" : combinedRisks, prefix + "strokeProbabilities" : strokeProbabilities,
                        prefix + "strokeRisks" : strokeRisks, prefix + "miRisks" : miRisks})
        
        return pd.concat([recalibration_df.reset_index(drop=True), risksAndProbs.reset_index(drop=True)], axis='columns', ignore_index=False)


    def create_or_rollback_events_to_correct_calibration(
        self,
        treatment_outcome_standard,
        treatedRiskVar,
        untreatedRiskVar,
        eventVar,
        outcomeType,
        fatalityDetermination,
        recalibration_pop,
        rng=None
    ):
        #logging.info(f"create or rollback {outcomeType}, standard: {treatment_outcome_standard[outcomeType]}")

        modelEstimatedRR = (
            recalibration_pop[treatedRiskVar].mean() / recalibration_pop[untreatedRiskVar].mean()
        )
        nextEventVar = eventVar + "Next"
        ageAtFirstVar = (
            "ageAtFirst" + eventVar.upper()
            if len(eventVar) == 2
            else "ageAtFirst" + eventVar.capitalize()
        )
        # use the delta between that effect and the calibration standard to recalibrate the pop.
        delta = modelEstimatedRR - treatment_outcome_standard[outcomeType]
        eventsForPeople = recalibration_pop.loc[recalibration_pop[nextEventVar] == True]

        numberOfEventStatusesToChange = abs(
            int(round(delta * len(eventsForPeople) / modelEstimatedRR))
        )
        nonEventsForPeople = recalibration_pop.loc[recalibration_pop[nextEventVar] == False]
        # key assumption: "treatment" is applied to a population as opposed to individuals within a population
        # analyses can be setup either way...build two populations and then set different treatments
        # or build a ur-population adn then set different treamtents within them
        # this is, i thikn, the first time where a coding decision is tied to one of those structure.
        # it would not, i think, be hard to change. but, just spelling it out here.

        # if negative, the model estimated too few events, if positive, too mnany
        #logging.info(f"bp recalibration, delta: {delta}, number of statuses to change: {numberOfEventStatusesToChange}")
        #logging.info(f"bp recalibration, delta: {delta} = {modelEstimatedRR} - {treatment_outcome_standard[outcomeType]}")
        #logging.info(f"bp recalibration, treated mean: {recalibration_pop[treatedRiskVar].mean()}, untreated mean: {recalibration_pop[untreatedRiskVar].mean()}")

        if delta < 0:
            if numberOfEventStatusesToChange > 0:
                new_events = nonEventsForPeople.sample(
                    n=numberOfEventStatusesToChange,
                    replace=False,
                    weights=nonEventsForPeople[untreatedRiskVar].values,
                )
                recalibration_pop.loc[new_events.index, nextEventVar] = True
                recalibration_pop.loc[
                    new_events.index, eventVar + "Fatal"
                ] = recalibration_pop.loc[new_events.index].apply(
                    fatalityDetermination, axis="columns", args=(True,None,rng)
                )
                recalibration_pop.loc[new_events.index, ageAtFirstVar] = np.fmin(
                    recalibration_pop.loc[new_events.index].age,
                    recalibration_pop.loc[new_events.index][ageAtFirstVar],
                )

        elif delta > 0:
            if numberOfEventStatusesToChange > len(eventsForPeople):
                numberOfEventStatusesToChange = len(eventsForPeople)
            if numberOfEventStatusesToChange > 0:
                events_to_rollback = eventsForPeople.sample(
                    n=numberOfEventStatusesToChange,
                    replace=False,
                    weights=1 - eventsForPeople[untreatedRiskVar].values,
                )
                recalibration_pop.loc[events_to_rollback.index, nextEventVar] = False
                recalibration_pop.loc[events_to_rollback.index, eventVar + "Fatal"] = False
                recalibration_pop.loc[events_to_rollback.index, "deadNext"] = False
                recalibration_pop.loc[events_to_rollback.index, ageAtFirstVar] = np.minimum(
                    recalibration_pop.loc[events_to_rollback.index].age,
                    recalibration_pop.loc[events_to_rollback.index][ageAtFirstVar],
                )
                recalibration_pop.loc[events_to_rollback.index, "rolledBackEventType"] = eventVar
        return recalibration_pop

    def get_people_alive_at_the_start_of_the_current_wave(self):
        return self.get_people_alive_at_the_start_of_wave(self._currentWave)

    def get_people_alive_at_the_start_of_wave(self, wave):
        peopleAlive = []
        for person in self._people:
            if person.alive_at_start_of_wave(wave):
                peopleAlive.append(person)
        return pd.Series(peopleAlive)

    def get_people_that_are_currently_alive(self):
        return pd.Series([not person.is_dead() for _, person in self._people.items()])

    def get_number_of_patients_currently_alive(self):
        self.get_people_that_are_currently_alive().sum()

    def get_events_in_most_recent_wave(self, eventType):
        peopleWithEvents = []
        for _, person in self._people.items():
            if person.has_outcome_at_age(eventType, person._age[-1]):
                peopleWithEvents.append(person)
        return peopleWithEvents

    def generate_starting_mean_patient(self):
        df = self.get_people_initial_state_as_dataframe()
        return Person(
            age=int(round(df.age.mean())),
            gender=NHANESGender(df.gender.mode()),
            raceEthnicity=NHANESRaceEthnicity(df.raceEthnicity.mode()),
            sbp=df.sbp.mean(),
            dbp=df.dbp.mean(),
            a1c=df.a1c.mean(),
            hdl=df.hdl.mean(),
            totChol=df.totChol.mean(),
            bmi=df.bmi.mean(),
            ldl=df.ldl.mean(),
            trig=df.trig.mean(),
            waist=df.waist.mean(),
            anyPhysicalActivity=df.anyPhysicalActivity.mode(),
            education=Education(df.education.mode()),
            smokingStatus=SmokingStatus(df.smokingStatus.mode()),
            antiHypertensiveCount=int(round(df.antiHypetensiveCount().mean())),
            statin=df.statin.mode(),
            otherLipidLoweringMedicationCount=int(
                round(df.otherLipidLoweringMedicationCount.mean())
            ),
            initializeAfib=(lambda _: False),
            selfReportStrokeAge=None,
            selfReportMIAge=None,
            randomEffects=self._outcome_model_repository.get_random_effects(),
        )

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

    def get_raw_incidence_by_age(self, eventType):
        popDF = self.get_people_current_state_as_dataframe()

        for year in range(1, self._totalWavesAdvanced + 1):
            eventVarName = "event" + str(year)
            ageVarName = "age" + str(year)
            popDF[ageVarName] = popDF["baseAge"] + year
            popDF[eventVarName] = [
                person.has_outcome_during_wave(year, OutcomeType.DEMENTIA)
                for person in self._people
            ]

        popDF = popDF[
            list(filter(lambda x: x.startswith("age") or x.startswith("event"), popDF.columns))
        ]
        popDF["id"] = popDF.index
        popDF.drop(columns=["age"], inplace=True)
        longAgesEvents = pd.wide_to_long(df=popDF, stubnames=["age", "event"], i="id", j="wave")

        agesAliveDF = self.get_people_current_state_as_dataframe()
        for year in range(1, self._totalWavesAdvanced + 1):
            aliveVarName = "alive" + str(year)
            ageVarName = "age" + str(year)
            agesAliveDF[ageVarName] = agesAliveDF["baseAge"] + year
            agesAliveDF[aliveVarName] = [
                person.alive_at_start_of_wave(year) for i, person in self._people.items()
            ]

        agesAliveDF = agesAliveDF[
            list(
                filter(lambda x: x.startswith("age") or x.startswith("alive"), agesAliveDF.columns)
            )
        ]
        agesAliveDF.drop(columns=["age"], inplace=True)
        agesAliveDF["id"] = agesAliveDF.index
        longAgesDead = pd.wide_to_long(
            df=agesAliveDF, stubnames=["age", "alive"], i="id", j="wave"
        )
        return (
            longAgesEvents.groupby("age")["event"].sum()
            / longAgesDead.groupby("age")["alive"].sum()
        )

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

    def use_pandarallel(self,flag): #must be able to change these attributes for instances that used pandarallel and will be passed to multiprocessing
        if flag:
            self.applyMethod = pd.DataFrame.parallel_apply
            self.applyMethodSeries = pd.Series.parallel_apply
        else:
            self.applyMethod = pd.DataFrame.apply
            self.applyMethodSeries = pd.Series.apply

def initializeAFib(person):
    #the intercept of this model was modified in order to have agreement with the 2019 global burden of disease data
    #optimization of the intercept was performed on the afibModelRecalibrations notebook
    model = load_regression_model("BaselineAFibModel")
    statsModel = StatsModelLogisticRiskFactorModel(model)
    return person._rng.uniform() < statsModel.estimate_next_risk(person)


def build_person(x, initializationModelRepository, rng=None):
    """Takes all Person-instance-related data via x and initializationModelRepository and organizes it,
       passes the organized data to the Person class and returns a Person instance.
       A random number generator is needed."""

    name = x.name
    
    personStaticRiskFactors = {
                        StaticRiskFactorsType.RACE_ETHNICITY.value: NHANESRaceEthnicity(int(x.raceEthnicity)),
                        StaticRiskFactorsType.EDUCATION.value: Education(int(x.education)),
                        StaticRiskFactorsType.GENDER.value: NHANESGender(int(x.gender)),
                        StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus(int(x.smokingStatus))}
    
    #TO DO: find a way to include everything here, including the rfs that need initialization
    #the PVD model would be easy to implement, eg with an estimate_next_risk_for_patient_characteristics function
    #but the AFIB model would be more difficult because it relies on statsmodel_logistic_risk file
    #for now include None, in order to create the risk factor lists correctly at the Person instance
    personDynamicRiskFactors = {
                        DynamicRiskFactorsType.AGE.value: x.age,
                        DynamicRiskFactorsType.SBP.value: x.meanSBP,
                        DynamicRiskFactorsType.DBP.value: x.meanDBP,
                        DynamicRiskFactorsType.A1C.value: x.a1c,
                        DynamicRiskFactorsType.HDL.value: x.hdl,
                        DynamicRiskFactorsType.LDL.value: x.ldl,
                        DynamicRiskFactorsType.TRIG.value: x.trig,
                        DynamicRiskFactorsType.TOT_CHOL.value: x.tot_chol,
                        DynamicRiskFactorsType.BMI.value: x.bmi,
                        DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: x.anyPhysicalActivity,
                        DynamicRiskFactorsType.AFIB.value: None,
                        DynamicRiskFactorsType.WAIST.value: x.waist,
                        DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.get_category_for_consumption(x.alcoholPerWeek),
                        DynamicRiskFactorsType.CREATININE.value: x.serumCreatinine,
                        DynamicRiskFactorsType.PVD.value: None}
    
    #Q: do we need otherLipid treatment? I am not bringing it to the Person objects for now.
    personDefaultTreatments = {
                        DefaultTreatmentsType.STATIN.value: round(x.statin),
                        #DefaultTreatmentsType.OTHER_LIPID_LOWERING_MEDICATION_COUNT.value: x.otherLipidLowering,
                        DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: x.antiHypertensive}
    
    personTreatmentStrategies = dict(zip([strategy.value for strategy in TreatmentStrategiesType],
                                         [[0] for strategy in range(len(TreatmentStrategiesType))]))
    
    personOutcomes = dict(zip([outcome for outcome in OutcomeType],
                                  [list() for outcome in range(len(OutcomeType))]))
    #add pre-simulation stroke outcomes
    selfReportStrokeAge=x.selfReportStrokeAge
    #Q: we should not add the stroke outcome in case of "else"?
    if selfReportStrokeAge is not None and selfReportStrokeAge > 1:
            selfReportStrokeAge = selfReportStrokeAge if selfReportStrokeAge <= x.age else x.age
            #personOutcomes[OutcomeType.STROKE].append((-1, Outcome(OutcomeType.STROKE, False)))
            personOutcomes[OutcomeType.STROKE].append((selfReportStrokeAge, StrokeOutcome(False, None, None, None, selfReported=True)))
    #add pre-simulation mi outcomes
    selfReportMIAge=rng.integers(18, x.age) if x.selfReportMIAge == 99999 else x.selfReportMIAge
    if selfReportMIAge is not None and selfReportMIAge > 1:
            selfReportMIAge = selfReportMIAge if selfReportMIAge <= x.age else x.age
            personOutcomes[OutcomeType.MI].append((selfReportMIAge, Outcome(OutcomeType.MI, False, selfReported=True)))

    person = Person(name, 
                   personStaticRiskFactors,
                   personDynamicRiskFactors, 
                   personDefaultTreatments,
                   personTreatmentStrategies,
                   personOutcomes)    
    
    #TO DO: find a way to initialize these rfs above with everything else
    person._pvd = [initializationModelRepository[DynamicRiskFactorsType.PVD].estimate_next_risk(person)]
    person._afib = [initializationModelRepository[DynamicRiskFactorsType.AFIB].estimate_next_risk(person)]
    return person


def build_people_using_nhanes_for_sampling(nhanes, n, rng=None, filter=None, random_seed=None, weights=None):
    """Creates a Pandas Series collection of Person instances.
       A random number generator is required."""

    if weights is None:
        weights = nhanes.WTINT2YR
    repeated_sample = nhanes.sample(n, weights=weights, random_state=random_seed, replace=True)
    initializationModelRepository = {DynamicRiskFactorsType.AFIB: AFibPrevalenceModel(), 
                                     DynamicRiskFactorsType.PVD: PVDPrevalenceModel()}
    people = pd.DataFrame.apply(repeated_sample,
                                build_person, initializationModelRepository=initializationModelRepository, rng=rng, axis="columns")

    #sets the unique identifier for each Person instance
    list(map(lambda person, i: setattr(person, "_index", i), people, range(n))) 

    #Q: why are we not applying the filter on the nhanes df, before we create all the Person instances?
    if filter is not None:
        people = people.loc[people.apply(filter)]
    return people


class NHANESDirectSamplePopulation(Population):
    """Simple base class to sample with replacement from 2015/2016 NHANES"""

    def __init__(
        self,
        #Q: there is an issue with making this n to be the population size
        #   the population class must set the pop size depending on what is passed on to it
        #   because this n may be modified by the filter later on, the way it is currently set up
        n,
        year,
        popModelRepository=None,
        filter=None,
        generate_new_people=True,
        model_reposistory_type="cohort",
        random_seed=None,
        weights=None,
    ):

        #Q: why are we not applying the filter to the NHANES dataframe? Instead we spend all the time creating Person-instances
        # we will later reject....
        nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        nhanes = nhanes.loc[nhanes.year == year]
        self._rng = np.random.default_rng()
        people = build_people_using_nhanes_for_sampling(
            nhanes,
            n,
            rng=self._rng,
            filter=filter,
            random_seed=random_seed,
            weights=weights,
        )
        #This is the default, self-consistent set of models for advancing an NHANES Population
        #Q: how to create one for the NHANESRiskModelRepository? Do we even need to do this? I have never used that....
        if popModelRepository is None:
            popModelRepository = PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                                           CohortDefaultTreatmentModelRepository(),
                                                           OutcomeModelRepository(),
                                                           CohortStaticRiskFactorModelRepository()) 
        super().__init__(people, popModelRepository)
        self.year = year

    def copy(self):
        newPop = NHANESDirectSamplePopulation(self.n, self.year, False)
        newPop._people = copy.deepcopy(self._people)
        return newPop

    def _initialize_risk_models(self, model_repository_type):
        if model_repository_type == "cohort":
            self._risk_model_repository = CohortRiskModelRepository()
        elif model_repository_type == "nhanes":
            self._risk_model_repository = NHANESRiskModelRepository()
        else:
            raise Exception("unknwon risk model repository type" + model_repository_type)

class PersonListPopulation(Population):
    def __init__(self, people):

        super().__init__(pd.Series(people))
        self.n = len(people)
        self._qaly_assignment_strategy = QALYAssignmentStrategy()
        self._outcome_model_repository = OutcomeModelRepository()
        self._risk_model_repository = CohortRiskModelRepository()
        # population index is used for efficiency in the population, need to set it on 
        # each person when a new population is setup
        for i, person in self._people.items():
            person._populationIndex = i
        # if the people have already been advanced, have the population start at that point
        self._currentWave = len(people[0]._age)-1


class NHANESAgeStandardPopulation(NHANESDirectSamplePopulation):
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
