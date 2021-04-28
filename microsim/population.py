from microsim.person import Person
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.gender import NHANESGender
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.nhanes_risk_model_repository import NHANESRiskModelRepository
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel
from microsim.data_loader import load_regression_model, get_absolute_datafile_path
from microsim.outcome_model_type import OutcomeModelType
from microsim.cv_outcome_determination import CVOutcomeDetermination
from microsim.outcome import Outcome, OutcomeType
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.initialization_repository import InitializationRepository

import pandas as pd
from pandarallel import pandarallel
import copy
import multiprocessing as mp
import numpy as np
import logging


class Population:
    """
    Unit of people subject to treatment program over time.

    (WIP) THe basic idea is that this is a generic superclass which will manage a group of
    people. Tangible subclasses will be needed to actually assign assumptions for a given
    population. As it stands, this class doesn't do anything (other than potentially being
    useful for tests), because it isn't tied to tangible risk models. Ultimately it might
    turn into an abstract class...
    """

    _ageStandards = {}

    def __init__(self, people):
        self._people = people
        self._ageStandards = {}
        # luciana tag: discuss with luciana...want to keep track of the sim wave htat is currently running, while running
        # and also the total number of years advanced...need to think about how to do this is a way that will be safe
        # this approach has major risks if you forget to update one of these variables
        self._totalWavesAdvanced = 0
        self._currentWave = 0
        self._bpTreatmentStrategy = None
        self.num_of_processes = 8

        self._riskFactors = ['sbp', 'dbp', 'a1c', 'hdl', 'ldl', 'trig', 'totChol',
                             'bmi', 'anyPhysicalActivity', 'afib', 'waist', 'alcoholPerWeek']
        # , 'otherLipidLoweringMedicationCount']
        self._treatments = ['antiHypertensiveCount', 'statin']
        self._timeVaryingCovariates = copy.copy(self._riskFactors)
        self._timeVaryingCovariates.append('age')
        self._timeVaryingCovariates.extend(self._treatments)
        self._timeVaryingCovariates.append('bpMedsAdded')

    def reset_to_baseline(self):
        self._totalWavesAdvanced = 0
        self._currentWave = 0
        self._bpTreatmentStrategy = None
        for person in self._people:
            person.reset_to_baseline()

    def advance(self, years):
        for yearIndex in range(years):
            logging.info(f"processing year: {yearIndex}")
            self._currentWave += 1
            for person in self._people:
                self.advance_person(person)
            self.apply_recalibration_standards()
            self._totalWavesAdvanced += 1

    # trying to work this out. if we do get it worked out, then we probably want to rebuild the person to use systematic data structrures
    # (i.e. static attributes, time-varying attributes)
    # also, will need to thikn about ways to make sure that the dataframe version of reality stays synced with teh "patient-based" version of reality
    # for now, will build a DF at the beginnign and then update the peopel at the end...
    def advance_vectorized(self, years):
        # get dataframe of people...
        df = self.get_people_current_state_and_summary_as_dataframe()
        alive = df.loc[df.dead == False]
        pandarallel.initialize(verbose=1)
        # might not need this row...depends o n whethe we do an bulk update on people or an wave-abased update
        waveAtStartOfAdvance = self._currentWave
        for yearIndex in range(years):
            logging.info(f"processing year: {yearIndex}")
            alive = alive.loc[alive.dead == False]
            self._currentWave += 1

            # advance risk factors
            for rf in self._riskFactors:
                #print(f"### Risk Factor: {rf}")
                alive[rf + "Next"] = alive.parallel_apply(self._risk_model_repository.get_model(
                    rf).estimate_next_risk_vectorized, axis='columns')

            # advance treatment
            for treatment in self._treatments:
                #print(f"### Treatment: {treatment}")
                alive[treatment + "Next"] = alive.apply(self._risk_model_repository.get_model(
                    treatment).estimate_next_risk_vectorized, axis='columns')

            # apply treatment modifications
            alive['bpMedsAddedNext'] = 0
            if self._bpTreatmentStrategy is not None:
                alive = alive.apply(
                    self._bpTreatmentStrategy.get_changes_vectorized, axis='columns')

            # advance outcomes
            # first, setup outcome variables
            for outcome in ['stroke', 'mi', 'dementia', 'dead', 'gcp']:
                alive[outcome + 'Next'] = 0
            alive['strokeFatal'] = 0
            alive['miFatal'] = 0

            # first determine if there is a cv event
            alive = alive.apply(
                self._outcome_model_repository.assign_cv_outcome_vectorized, axis='columns')
            
            # reselect on survivors...because we don't want to set non-cv mortality on poepel that died from cv events.
            alive['gcp'] = alive.apply(
                self._outcome_model_repository.get_gcp_vectorized, axis='columns')
            newDementia = alive.loc[~alive.dementia].apply(
                self._outcome_model_repository.get_dementia_vectorized, axis='columns')
            alive['dementiaNext']  = newDementia
            alive.loc[alive['dementiaNext']==1, 'ageAtFirstDementia'] = alive.age
            alive['dementia'] = newDementia | alive['dementia']
            alive['nonCVDeathNext'] = alive.apply(
                self._outcome_model_repository.assign_non_cv_mortality_vectorized, axis='columns')
            alive['cvDeathNext'] =  alive['deadNext']
            alive['deadNext'] = alive['nonCVDeathNext'] | alive['cvDeathNext']

            alive = self.apply_recalibration_standards(alive)

            alive['qalyNext'] = alive.apply(
                QALYAssignmentStrategy().get_qalys_vectorized, axis='columns')

            alive.loc[~alive.dead, 'age'] = alive.age + 1
            self._totalWavesAdvanced += 1

            #### OK ...now i have it...
            #### this is where it goes sideways...
            ### we rollback the even ton the person...
            ### but, then we don't include that person in the DF because they were already excluded from the DF
            ### in the next wave...because theyr'e "dead" in teh alive DF, but alive on the peson.

            ### fix #1 — recalibrate on teh DF adn then push to people after that.
            ### fix #2 — after recalibration, track which people were recalibrated and update teh alive DF
            alive = self.move_people_df_forward(alive)
            # for efficieicny, we could try to do this all at the end...but, its a bit cleanear  to do it wave by wave
            alive.apply(self.push_updates_back_to_people, axis='columns')
            nextCols = [col for col in alive.columns if "Next" in col]
            alive.drop(columns=nextCols, inplace=True)

            # for efficiency...probaly  want to also push this forward
        return alive, df


    def push_updates_back_to_people(self, x):
        person = self._people.iloc[int(x.populationIndex)]
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

        # advance outcomes
        for outcomeName, outcomeType in {'stroke':  OutcomeType.STROKE, 'mi': OutcomeType.MI, 'dementia': OutcomeType.DEMENTIA}.items():
            if x[outcomeName + "Next"]:
                fatal = False if outcomeName == "dementia" else x[outcomeName + "Fatal"]
                person.add_outcome_event(Outcome(outcomeType, fatal))

        person._gcp.append(x.gcp)
        person._qalys.append(x.qalyNext)
        person._bpMedsAdded.append(x.bpMedsAddedNext)
        if not person.is_dead():
            person._age.append(x.age)
            person._alive.append(True)
        return person

    def move_people_df_forward(self, df):
        factorsToChange = copy.copy(self._riskFactors)
        factorsToChange.extend(self._treatments)

        for rf in factorsToChange:
            # the curent value is stored in the variable name
            df[rf] = df[rf + 'Next']
            df[rf + str(self._currentWave)] = df[rf + 'Next']
            df['mean' + rf.capitalize()] = (df['mean' + rf.capitalize()] *
                                            (df['totalYearsInSim']+1) + df[rf + 'Next']) / (df['totalYearsInSim']+2)
        for outcome in ['mi', 'stroke']:
            df[outcome + 'InSim'] = df[outcome + 'InSim'] | df[outcome + "Next"]
            df[outcome + str(self._currentWave)] = df[outcome + "Next"]
        df['dead'] = df['dead'] | df['deadNext']
        df['dead' + str(self._currentWave)] = df['deadNext']

        df['totalYearsInSim'] = df['totalYearsInSim'] + 1
        df['current_diabetes'] = df['a1c'] > 6.5
        df['current_bp_treatment'] = df['antiHypertensiveCount'] >= 1
        df['totalQalys'] = df['totalQalys'] + df['qalyNext']
        # assign ages for new events
        #df.loc[(df.ageAtFirstStroke.isnull()) & (df.strokeNext), 'ageAtFirstStroke'] = df.age
        #df.loc[(df.ageAtFirstMI.isnull()) & (df.miNext), 'ageAtFirstMI'] = df.age
        #df.loc[(df.ageAtFirstDementia.isnull()) & (df.dementiaNext), 'ageAtFirstDementia'] = df.age

        return df

    def advance_person(self, person):
        if not person.is_dead():
            person.advance_year(self._risk_model_repository,
                                self._outcome_model_repository,
                                self._qaly_assignment_strategy)
        return person

    def advance_people(self, people):
        return people.apply(self.advance_person)

    def advance_multi_process(self, years):
        for i in range(years):
            self._currentWave += 1
            logging.info(f"processing year: {i}")
            data_split = np.array_split(self._people, self.num_of_processes)
            pool = mp.Pool(self.num_of_processes)
            self._people = pd.concat(pool.map(self.advance_people, data_split))
            pool.close()
            pool.join()

            self.apply_recalibration_standards()
            self._totalWavesAdvanced += 1

    def set_bp_treatment_strategy(self, bpTreatmentStrategy):
        self._bpTreatmentStrategy = bpTreatmentStrategy
        for person in self._people:
            person._bpTreatmentStrategy = bpTreatmentStrategy

    def apply_recalibration_standards(self, recalibration_df):
        # treatment_standard is a dictionary of outcome types and effect sizees
        if (self._bpTreatmentStrategy is not None):
            if (self._bpTreatmentStrategy.get_treatment_recalibration_for_population() is not None):
                recalibration_df = self.recalibrate_bp_treatment(recalibration_df)
        return recalibration_df

    # should the estiamted treatment effect be based on the number of events in the population
    # (i.e. # events treated / # of events untreated)
    # of should it be based on teh predicted reisks
    # the problem with the first approach is that its going to depend a lot on small sample sizes...
    # and we don't necessarily want to take out that random error...that random error reflects
    # genuine uncertainty.
    # so, i thikn it should be based on the model-predicted risks...

    def recalibrate_bp_treatment(self, recalibration_df):
        treatment_outcome_standard = self._bpTreatmentStrategy.get_treatment_recalibration_for_population()
        # estimate risk for the people alive at the start of the wave
        #recalibration_pop = self.get_people_alive_at_the_start_of_the_current_wave()
        self.estimate_risks(recalibration_df, "treated")

        # rollback the treatment effect.
        # redtag: would like to apply to this to a deeply cloned population, but i can't get that to work
        # so, for now, applying it to the actual population and then rolling the effect back later.
        recalibration_df = recalibration_df.apply(self._bpTreatmentStrategy.rollback_changes_vectorized, axis='columns')

        # estimate risk after applying the treamtent effect
        self.estimate_risks(recalibration_df, "untreated")

        # hacktag related to above — roll back the treatment effect...
        recalibration_df = recalibration_df.apply(self._bpTreatmentStrategy.get_changes_vectorized, axis='columns')

        maxMeds = 5
        # recalibrate within each group of added medicaitons so that we can stratify the treamtnet effects
        for i in range(1, maxMeds+1):
            recalibrationPopForMedCount = recalibration_df.loc[(recalibration_df.bpMedsAddedNext==i) | (recalibration_df.bpMedsAddedNext.values>=maxMeds)]            
            # the change standards are for a single medication
            recalibration_standard_for_med_count = treatment_outcome_standard.copy()
            for key, value in recalibration_standard_for_med_count.items():
                recalibration_standard_for_med_count[key] = value**i

            if len(recalibrationPopForMedCount) > 0:
                # recalibrate stroke
                recalibratedForMedCount = self.create_or_rollback_events_to_correct_calibration(recalibration_standard_for_med_count,
                                                                      "treatedstrokeRisks",
                                                                      "untreatedstrokeRisks",  
                                                                      "stroke",
                                                                      OutcomeType.STROKE,
                                                                      CVOutcomeDetermination()._will_have_fatal_stroke,
                                                                      recalibrationPopForMedCount)

                recalibration_df.loc[recalibratedForMedCount.index, 'strokeNext'] = recalibratedForMedCount['strokeNext']
                recalibration_df.loc[recalibratedForMedCount.index, 'strokeFatal'] = recalibratedForMedCount['strokeFatal']
                recalibration_df.loc[recalibratedForMedCount.index, 'ageAtFirstStroke'] = recalibratedForMedCount['ageAtFirstStroke']

                # recalibrate MI
                recalibratedForMedCount = self.create_or_rollback_events_to_correct_calibration(recalibration_standard_for_med_count,
                                                                      "treatedmiRisks",
                                                                      "untreatedmiRisks",
                                                                      "mi",
                                                                      OutcomeType.MI,
                                                                      CVOutcomeDetermination()._will_have_fatal_mi,
                                                                      recalibrationPopForMedCount)
                recalibration_df.loc[recalibratedForMedCount.index, 'miNext'] = recalibratedForMedCount['miNext']
                recalibration_df.loc[recalibratedForMedCount.index, 'miFatal'] = recalibratedForMedCount['miFatal']
                recalibration_df.loc[recalibratedForMedCount.index, 'ageAtFirstMI'] = recalibratedForMedCount['ageAtFirstMI']
        return recalibration_df

                

    def estimate_risks(self, recalibration_df, prefix):
        recalibration_df[prefix + 'combinedRisks'] = recalibration_df.apply(self._outcome_model_repository.get_risk_for_person_vectorized, axis='columns', args=(OutcomeModelType.CARDIOVASCULAR, 1))
        recalibration_df[prefix + 'strokeProbabilities'] = recalibration_df.apply(CVOutcomeDetermination().get_stroke_probability, axis='columns', vectorized=True)
        recalibration_df[prefix + 'strokeRisks'] = recalibration_df[prefix + 'combinedRisks'] * recalibration_df[prefix + 'strokeProbabilities']
        recalibration_df[prefix + 'miRisks'] = recalibration_df[prefix + 'combinedRisks'] * (1-recalibration_df[prefix + 'strokeProbabilities'])
        return recalibration_df

    def create_or_rollback_events_to_correct_calibration(self,
                                                         treatment_outcome_standard,
                                                         treatedRiskVar,
                                                         untreatedRiskVar,
                                                         eventVar,
                                                         outcomeType,
                                                         fatalityDetermination,
                                                         recalibration_pop):
        modelEstimatedRR = recalibration_pop[treatedRiskVar].mean()/recalibration_pop[untreatedRiskVar].mean()
        nextEventVar = eventVar + "Next"
        ageAtFirstVar = "ageAtFirst" + eventVar.upper() if len(eventVar) == 2 else "ageAtFirst" + eventVar.capitalize()
        # use the delta between that effect and the calibration standard to recalibrate the pop.
        delta = modelEstimatedRR - treatment_outcome_standard[outcomeType]
        eventsForPeople = recalibration_pop.loc[recalibration_pop[nextEventVar]==True]
                
        numberOfEventStatusesToChange = abs(
            int(round(delta * len(eventsForPeople)/modelEstimatedRR)))
        nonEventsForPeople = recalibration_pop.loc[recalibration_pop[nextEventVar]==False]
        # key assumption: "treatment" is applied to a population as opposed to individuals within a population
        # analyses can be setup either way...build two populations and then set different treatments
        # or build a ur-population adn then set different treamtents within them
        # this is, i thikn, the first time where a coding decision is tied to one of those structure.
        # it would not, i think, be hard to change. but, just spelling it out here.

        # if negative, the model estimated too few events, if positive, too mnany
        if delta < 0:
            if numberOfEventStatusesToChange > 0:
                new_events = nonEventsForPeople.sample(n=numberOfEventStatusesToChange,
                                                            replace=False,
                                                            weights=nonEventsForPeople[untreatedRiskVar].values)
                recalibration_pop.loc[new_events.index, nextEventVar] = True
                recalibration_pop.loc[new_events.index, eventVar + 'Fatal'] = recalibration_pop.loc[new_events.index].apply(fatalityDetermination, axis='columns', args=(True,))
                recalibration_pop.loc[new_events.index, ageAtFirstVar] = np.fmin(recalibration_pop.loc[new_events.index].age, recalibration_pop.loc[new_events.index][ageAtFirstVar])
            

        elif delta > 0:
            if numberOfEventStatusesToChange > len(eventsForPeople):
                numberOfEventStatusesToChange = len(eventsForPeople)
            if numberOfEventStatusesToChange > 0:
                events_to_rollback = eventsForPeople.sample(n=numberOfEventStatusesToChange,
                                                                                   replace=False,
                                                                                   weights=1-eventsForPeople[untreatedRiskVar].values)
                recalibration_pop.loc[events_to_rollback.index, nextEventVar] = False
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
        return pd.Series([not person.is_dead() for _, person in self._people.iteritems()])

    def get_number_of_patients_currently_alive(self):
        self.get_people_that_are_currently_alive().sum()

    def get_events_in_most_recent_wave(self, eventType):
        peopleWithEvents = []
        for _, person in self._people.iteritems():
            if person.has_outcome_at_age(eventType, person._age[-1]):
                peopleWithEvents.append(person)
        return peopleWithEvents

    def generate_starting_mean_patient(self):
        df = self.get_people_initial_state_as_dataframe()
        return Person(age=int(round(df.age.mean())),
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
                          round(df.otherLipidLoweringMedicationCount.mean())),
                      initializeAfib=(lambda _: False),
                      selfReportStrokeAge=None,
                      selfReportMIAge=None,
                      randomEffects=self._outcome_model_repository.get_random_effects())

    def get_event_rate_in_simulation(self, eventType, duration):
        events = [person.has_outcome_during_simulation_prior_to_wave(eventType, duration) for i, person in self._people.iteritems()]
        totalTime = [person.years_in_simulation() if person.years_in_simulation() < duration  else duration for i, person in self._people.iteritems()]
        return np.array(events).sum() / np.array(totalTime).sum()
    
    def get_raw_incidence_by_age(self, eventType):
        popDF = self.get_people_current_state_as_dataframe()

        for year in range(1, self._totalWavesAdvanced + 1):
            eventVarName = 'event' + str(year)
            ageVarName = 'age' + str(year)
            popDF[ageVarName] = popDF['baseAge'] + year
            popDF[eventVarName] = [person.has_outcome_during_wave(
                year, OutcomeType.DEMENTIA) for person in self._people]

        popDF = popDF[list(filter(lambda x: x.startswith(
            'age') or x.startswith('event'), popDF.columns))]
        popDF['id'] = popDF.index
        popDF.drop(columns=['age'], inplace=True)
        longAgesEvents = pd.wide_to_long(df=popDF, stubnames=['age', 'event'], i='id', j='wave')

        agesAliveDF = self.get_people_current_state_as_dataframe()
        for year in range(1, self._totalWavesAdvanced + 1):
            aliveVarName = 'alive' + str(year)
            ageVarName = 'age' + str(year)
            agesAliveDF[ageVarName] = agesAliveDF['baseAge'] + year
            agesAliveDF[aliveVarName] = [person.alive_at_start_of_wave(
                year) for i, person in self._people.iteritems()]

        agesAliveDF = agesAliveDF[list(filter(lambda x: x.startswith(
            'age') or x.startswith('alive'), agesAliveDF.columns))]
        agesAliveDF.drop(columns=['age'], inplace=True)
        agesAliveDF['id'] = agesAliveDF.index
        longAgesDead = pd.wide_to_long(df=agesAliveDF, stubnames=[
                                       'age', 'alive'], i='id', j='wave')
        return longAgesEvents.groupby('age')['event'].sum()/longAgesDead.groupby('age')['alive'].sum()

    # refactorrtag: we should probably build a specific class that loads data files...

    def build_age_standard(self, yearOfStandardizedPopulation):
        if yearOfStandardizedPopulation in Population._ageStandards:
            return copy.deepcopy(Population._ageStandards[yearOfStandardizedPopulation])

        datafile_path = get_absolute_datafile_path("us.1969_2017.19ages.adjusted.txt")
        ageStandard = pd.read_csv(datafile_path, header=0, names=['raw'])
        # https://seer.cancer.gov/popdata/popdic.html
        ageStandard['year'] = ageStandard['raw'].str[0:4]
        ageStandard['year'] = ageStandard.year.astype(int)
        # format changes in 1990...so, we'll go forward from there...
        ageStandard = ageStandard.loc[ageStandard.year >= 1990]
        ageStandard['state'] = ageStandard['raw'].str[4:6]
        ageStandard['state'] = ageStandard['raw'].str[4:6]
        # 1 = white, 2 = black, 3 = american indian/alaskan, 4 = asian/pacific islander
        ageStandard['race'] = ageStandard['raw'].str[13:14]
        ageStandard['hispanic'] = ageStandard['raw'].str[14:15]
        ageStandard['female'] = ageStandard['raw'].str[15:16]
        ageStandard['female'] = ageStandard['female'].astype(int)
        ageStandard['female'] = ageStandard['female'].replace({1: 0, 2: 1})
        ageStandard['ageGroup'] = ageStandard['raw'].str[16:18]
        ageStandard['ageGroup'] = ageStandard['ageGroup'].astype(int)
        ageStandard['standardPopulation'] = ageStandard['raw'].str[18:26]
        ageStandard['standardPopulation'] = ageStandard['standardPopulation'].astype(int)
        ageStandard['lowerAgeBound'] = (ageStandard.ageGroup - 1) * 5
        ageStandard['upperAgeBound'] = (ageStandard.ageGroup * 5) - 1
        ageStandard['lowerAgeBound'] = ageStandard['lowerAgeBound'].replace({-5: 0, 0: 1})
        ageStandard['upperAgeBound'] = ageStandard['upperAgeBound'].replace({-1: 0, 89: 150})
        ageStandardYear = ageStandard.loc[ageStandard.year == yearOfStandardizedPopulation]
        ageStandardGroupby = ageStandardYear[['female',
                                              'standardPopulation',
                                              'lowerAgeBound',
                                              'upperAgeBound',
                                              'ageGroup']].groupby(['ageGroup',
                                                                    'female'])
        ageStandardHeaders = ageStandardGroupby.first()[['lowerAgeBound', 'upperAgeBound']]
        ageStandardHeaders['female'] = ageStandardHeaders.index.get_level_values(1)
        ageStandardPopulation = ageStandardYear[['female', 'standardPopulation', 'ageGroup']]
        ageStandardPopulation = ageStandardPopulation.groupby(['ageGroup', 'female']).sum()
        ageStandardPopulation = ageStandardHeaders.join(ageStandardPopulation, how='inner')
        # cache the age standard populations...they're not that big and it takes a while
        # to build one
        ageStandardPopulation['outcomeCount'] = 0
        ageStandardPopulation['simPersonYears'] = 0
        ageStandardPopulation['simPeople'] = 0
        Population._ageStandards[yearOfStandardizedPopulation] = copy.deepcopy(
            ageStandardPopulation)

        return ageStandardPopulation

    def tabulate_age_specific_rates(self, ageStandard):
        ageStandard['percentStandardPopInGroup'] = ageStandard['standardPopulation'] / \
            (ageStandard['standardPopulation'].sum())
        ageStandard['ageSpecificRate'] = ageStandard['outcomeCount'] * \
            100000 / ageStandard['simPersonYears']
        ageStandard['ageSpecificContribution'] = ageStandard['ageSpecificRate'] * \
            ageStandard['percentStandardPopInGroup']
        return ageStandard

    # return the age standardized # of events per 100,000 person years
    def calculate_mean_age_sex_standardized_incidence(
            self, outcomeType, yearOfStandardizedPopulation=2016,
            subPopulationSelector=None, subPopulationDFSelector=None):

        # the age selector picks the first outcome (_outcomes(outcomeTYpe)[0]) and the age is the
        # first element within the returned tuple (the second [0])
        events = self.calculate_mean_age_sex_standardized_event(
            lambda x: x.has_outcome_during_simulation(outcomeType),
            lambda x: x.get_outcomes_during_simulation(outcomeType)[0][0] - x._age[0] + 1,
            yearOfStandardizedPopulation,
            subPopulationSelector,
            subPopulationDFSelector)
        return (pd.Series([event[0] for event in events]).mean(),
                pd.Series([event[1] for event in events]).sum())

    def calculate_mean_age_sex_standardized_mortality(self, yearOfStandardizedPopulation=2016):
        events = self.calculate_mean_age_sex_standardized_event(lambda x: x.is_dead(),
                                                                lambda x: x.years_in_simulation(),
                                                                yearOfStandardizedPopulation)
        return pd.Series([event[0] for event in events]).mean()

    def get_events_for_event_type(self, eventSelector, eventAgeIdentifier, subPopulationSelector=None, subPopulationDFSelector=None):
        # build a dataframe to represent the population
        popDF = self.get_people_current_state_as_dataframe(parallel=False)
        popDF['female'] = popDF['gender'] - 1

        # calculated standardized event rate for each year
        for year in range(1, self._totalWavesAdvanced + 1):
            eventVarName = 'event' + str(year)
            ageVarName = 'age' + str(year)
            popDF[ageVarName] = popDF['baseAge'] + year
            if subPopulationDFSelector is not None:
                popDF['subpopFilter'] = popDF.apply(subPopulationDFSelector, axis='columns')
                popDF = popDF.loc[popDF.subpopFilter == 1]
            popDF[eventVarName] = [eventSelector(person) and eventAgeIdentifier(
                person) == year for person in filter(subPopulationSelector, self._people)]
        return popDF

    def  calculate_mean_age_sex_standardized_event(self, eventSelector, eventAgeIdentifier,
                                                  yearOfStandardizedPopulation=2016,
                                                  subPopulationSelector=None,
                                                  subPopulationDFSelector=None):
        # calculated standardized event rate for each year
        popDF = self.get_events_for_event_type(eventSelector, eventAgeIdentifier,
                                          subPopulationSelector,
                                          subPopulationDFSelector)
        popDF['female'] = popDF['gender'] - 1

        eventsPerYear = []

        for year in range(1, self._totalWavesAdvanced + 1):
            eventVarName = 'event' + str(year)
            ageVarName = 'age' + str(year)
            popDF[ageVarName] = popDF['baseAge'] + year
            if subPopulationDFSelector is not None:
                popDF['subpopFilter'] = popDF.apply(subPopulationDFSelector, axis='columns')
                popDF = popDF.loc[popDF.subpopFilter == 1]
            popDF[eventVarName] = [eventSelector(person) and eventAgeIdentifier(
                person) == year for person in filter(subPopulationSelector, self._people)] 
            dfForAnnualEventCalc = popDF[[ageVarName, 'female', eventVarName]]
            dfForAnnualEventCalc.rename(
                columns={
                    ageVarName: 'age',
                    eventVarName: 'event'},
                inplace=True)
            eventsPerYear.append(
                self.get_standardized_events_for_year(
                    dfForAnnualEventCalc,
                    yearOfStandardizedPopulation))

        return eventsPerYear

    def get_standardized_events_for_year(self, peopleDF, yearOfStandardizedPopulation):
        ageStandard = self.build_age_standard(yearOfStandardizedPopulation)
        # limit to the years where there are people
        # if the simulation runs for 50 years...there will be empty cells in all of the
        # young person categories
        ageStandard = ageStandard.loc[ageStandard.lowerAgeBound >= peopleDF.age.min()]

        # take the dataframe of peoplein teh population and tabnulate events relative
        # to the age standard (max age is 85 in the age standard...)
        peopleDF.loc[peopleDF['age'] > 85, 'age'] = 85
        peopleDF.loc[:,'ageGroup'] = (peopleDF['age'] // 5) + 1
        peopleDF.loc[:,'ageGroup'] = peopleDF['ageGroup'].astype(int)
        # tabulate events by group
        eventsByGroup = peopleDF.groupby(['ageGroup', 'female'])['event'].sum()
        personYears = peopleDF.groupby(['ageGroup', 'female'])['age'].count()
        # set those events on the age standard
        ageStandard['outcomeCount'] = eventsByGroup
        ageStandard['simPersonYears'] = personYears

        ageStandard = self.tabulate_age_specific_rates(ageStandard)
        return((ageStandard.ageSpecificContribution.sum(), ageStandard.outcomeCount.sum()))

    def get_person_attributes_from_person(self, person, timeVaryingCovariates):
        attrForPerson = {'age': person._age[-1],
                         'baseAge': person._age[0],
                         'gender': person._gender,
                         'raceEthnicity': person._raceEthnicity,
                         'black': person._raceEthnicity == 4,
                         'sbp': person._sbp[-1],
                         'dbp': person._dbp[-1],
                         'a1c': person._a1c[-1],
                         'current_diabetes': person._a1c[-1] > 6.5,
                         'hdl': person._hdl[-1],
                         'ldl': person._ldl[-1],
                         'trig': person._trig[-1],
                         'totChol': person._totChol[-1],
                         'bmi': person._bmi[-1],
                         'anyPhysicalActivity': person._anyPhysicalActivity[-1],
                         'education': person._education.value,
                         'afib': person._afib[-1],
                         'alcoholPerWeek': person._alcoholPerWeek[-1],
                         'antiHypertensiveCount': person._antiHypertensiveCount[-1],
                         'current_bp_treatment': person._antiHypertensiveCount[-1] > 0,
                         'statin': person._statin[-1],
                         'otherLipidLoweringMedicationCount': person._otherLipidLoweringMedicationCount[-1],
                         'waist': person._waist[-1],
                         'smokingStatus': person._smokingStatus,
                         'current_smoker': person._smokingStatus == 2,
                         'dead': person.is_dead(),
                         'gcpRandomEffect': person._randomEffects['gcp'],
                         'miPriorToSim': person._selfReportMIPriorToSim,
                         'mi': person._selfReportMIPriorToSim or person.has_mi_during_simulation(),
                         'stroke': person._selfReportStrokePriorToSim or person.has_stroke_during_simulation(),
                         'ageAtFirstStroke': person.get_age_at_first_outcome(OutcomeType.STROKE),
                         'ageAtFirstMI': person.get_age_at_first_outcome(OutcomeType.MI),
                         'ageAtFirstDementia': person.get_age_at_first_outcome(OutcomeType.DEMENTIA),
                         'miInSim': person.has_mi_during_simulation(),
                         'strokePriorToSim': person._selfReportStrokePriorToSim,
                         'strokeInSim': person.has_stroke_during_simulation(),
                         'dementia': person._dementia,
                         'gcp': person._gcp[-1],
                         'baseGcp': person._gcp[0],
                         'gcpSlope': person._gcp[-1] - person._gcp[-2] if len(person._gcp) >= 2 else 0,
                         'totalYearsInSim': person.years_in_simulation(),
                         'totalQalys': np.array(person._qalys).sum(),
                         'bpMedsAdded' : person._bpMedsAdded[-1]}
        try:
            attrForPerson['populationIndex'] = person._populationIndex
        except AttributeError:
            pass  # populationIndex is not necessary for advancing; can continue safely without it

        for var in timeVaryingCovariates:
            attr = getattr(person, "_" + var)
            for wave in range(0, len(attr)):
                attrForPerson[var + str(wave)] = attr[wave]
        return attrForPerson

    def get_people_current_state_as_dataframe(self, parallel=True):
        if parallel:
            pandarallel.initialize(verbose=1)
            return pd.DataFrame.from_dict(self._people.parallel_apply(self.get_person_attributes_from_person, timeVaryingCovariates=self._timeVaryingCovariates).array)
        else:
            return pd.DataFrame.from_dict(self._people.apply(self.get_person_attributes_from_person, timeVaryingCovariates=self._timeVaryingCovariates).array)

    def get_people_current_state_and_summary_as_dataframe(self):
        df = self.get_people_current_state_as_dataframe()
        # iterate through variables that vary over time
        for var in self._timeVaryingCovariates:
            df['mean' + var.capitalize()] = [pd.Series(getattr(person, "_" + var)).mean()
                                             for i, person in self._people.iteritems()]
        return df

    def get_people_initial_state_as_dataframe(self):
        return pd.DataFrame({'age': [person._age[0] for person in self._people],
                             'gender': [person._gender for person in self._people],
                             'raceEthnicity': [person._raceEthnicity for person in self._people],
                             'sbp': [person._sbp[0] for person in self._people],
                             'dbp': [person._dbp[0] for person in self._people],
                             'a1c': [person._a1c[0] for person in self._people],
                             'hdl': [person._hdl[0] for person in self._people],
                             'ldl': [person._ldl[0] for person in self._people],
                             'trig': [person._trig[0] for person in self._people],
                             'totChol': [person._totChol[0] for person in self._people],
                             'bmi': [person._bmi[0] for person in self._people],
                             'anyPhysicalActivity': [person._anyPhysicalActivity[0] for person in self._people],
                             'education': [person._education.value for person in self._people],
                             'afib': [person._afib[0] for person in self._people],
                             'antiHypertensiveCount': [person._antiHypertensiveCount[0] for person in self._people],
                             'statin': [person._statin[0] for person in self._people],
                             'otherLipidLoweringMedicationCount': [person._otherLipidLoweringMedicationCount[0] for person in self._people],
                             'waist': [person._waist[0] for person in self._people],
                             'smokingStatus': [person._smokingStatus for person in self._people],
                             'miPriorToSim': [person._selfReportMIPriorToSim for person in self._people],
                             'strokePriorToSim': [person._selfReportStrokePriorToSim for person in self._people],
                             'totalQalys': [np.array(person._qalys).sum() for person in self._people]})


def initializeAFib(person):
    model = load_regression_model("BaselineAFibModel")
    statsModel = StatsModelLogisticRiskFactorModel(model)
    return statsModel.estimate_next_risk(person)


def build_person(x, outcome_model_repository):
    return Person(
        age=x.age,
        gender=NHANESGender(int(x.gender)),
        raceEthnicity=NHANESRaceEthnicity(int(x.raceEthnicity)),
        sbp=x.meanSBP,
        dbp=x.meanDBP,
        a1c=x.a1c,
        hdl=x.hdl,
        ldl=x.ldl,
        trig=x.trig,
        totChol=x.tot_chol,
        bmi=x.bmi,
        waist=x.waist,
        anyPhysicalActivity=x.anyPhysicalActivity,
        smokingStatus=SmokingStatus(int(x.smokingStatus)),
        alcohol=AlcoholCategory.get_category_for_consumption(x.alcoholPerWeek),
        education=Education(int(x.education)),
        antiHypertensiveCount=x.antiHypertensive,
        statin=x.statin,
        otherLipidLoweringMedicationCount=x.otherLipidLowering,
        initializeAfib=initializeAFib,
        initializationRepository=InitializationRepository(),
        selfReportStrokeAge=x.selfReportStrokeAge,
        selfReportMIAge=np.random.randint(
            18, x.age) if x.selfReportMIAge == 99999 else x.selfReportMIAge,
        randomEffects=outcome_model_repository.get_random_effects(),
        dfIndex=x.index,
        diedBy2015=x.diedBy2015==True)


def build_people_using_nhanes_for_sampling(nhanes, n, outcome_model_repository,  filter=None, random_seed=None, weights=None):
    if weights is None:
        weights = nhanes.WTINT2YR
    repeated_sample = nhanes.sample(
        n,
        weights=weights,
        random_state=random_seed,
        replace=True)
    pandarallel.initialize(verbose=1)
    people = repeated_sample.parallel_apply(
        build_person, outcome_model_repository=outcome_model_repository, axis='columns')
    
    for i in range(0, len(people)):
        people.iloc[i]._populationIndex = i

    if filter is not None:
        people = people.loc[people.apply(filter)]

    return people


class NHANESDirectSamplePopulation(Population):
    """ Simple base class to sample with replacement from 2015/2016 NHANES """

    def __init__(
            self,
            n,
            year,
            filter=None,
            generate_new_people=True,
            model_reposistory_type="cohort",
            random_seed=None,
            weights=None):

        nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        nhanes = nhanes.loc[nhanes.year == year]
        self._outcome_model_repository = OutcomeModelRepository()
        people = build_people_using_nhanes_for_sampling(
            nhanes, n, self._outcome_model_repository,  filter=filter, random_seed=random_seed, weights=weights)
        super().__init__(people)
        self._qaly_assignment_strategy = QALYAssignmentStrategy()
        self.n = n
        self.year = year
        self._initialize_risk_models(model_reposistory_type)

    def copy(self):
        newPop = NHANESDirectSamplePopulation(self.n, self.year, False)
        newPop._people = copy.deepcopy(self._people)
        return newPop

    def _initialize_risk_models(self, model_repository_type):
        if (model_repository_type == "cohort"):
            self._risk_model_repository = CohortRiskModelRepository()
        elif (model_repository_type == "nhanes"):
            self._risk_model_repository = NHANESRiskModelRepository()
        else:
            raise Exception('unknwon risk model repository type' + model_repository_type)

class NHANESAgeStandardPopulation(NHANESDirectSamplePopulation):
        def __init__(self, n, year):
            nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
            weights = self.get_weights(year)
            weights['gender'] = weights['female'] + 1
            weights =  pd.merge(nhanes, weights, how='left', on=['age', 'gender']).popWeight
            super().__init__(n=n, year=year, weights = weights)

        def get_weights(self, year):
            standard = self.build_age_standard(year)
            return self.get_population_weighted_standard(standard)

        def get_population_weighted_standard(self, standard):
            rows = []
            for age in range(1, 151):
                for female in range (0, 2):
                    dfRow = standard.loc[(age >= standard.lowerAgeBound ) & (age <= standard.upperAgeBound) & (standard.female==female)]
                    upperAge = dfRow['upperAgeBound'].values[0]
                    lowerAge = dfRow['lowerAgeBound'].values[0]
                    totalPop = dfRow['standardPopulation'].values[0]
                    rows.append({'age' : age, 'female' : female, 'pop' : totalPop/(upperAge-lowerAge+1)})
            df = pd.DataFrame(rows)
            df['popWeight'] = df['pop']/df['pop'].sum()
            return df




class ClonePopulation(Population):
    """ Simple class to build a "Population" seeded by mulitple copies of the same person"""

    def __init__(self,person, n):
        self._outcome_model_repository = OutcomeModelRepository()
        self._qaly_assignment_strategy = QALYAssignmentStrategy()
        self._risk_model_repository = CohortRiskModelRepository()
        self.n = n

        people = pd.Series([copy.deepcopy(person) for i in range(0, n)])
        
        pandarallel.initialize(verbose=1)
        for i in range(0, len(people)):
            people.iloc[i]._populationIndex = i
        super().__init__(people)
