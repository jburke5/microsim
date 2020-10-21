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

import pandas as pd
import copy
import multiprocessing as mp
import numpy as np
from functools import partial


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
        self._risk_model_repository = None
        self._outcome_model_repository = None
        self._qaly_assignment_strategy = None
        self._ageStandards = {}
        # luciana tag: discuss with luciana...want to keep track of the sim wave htat is currently running, while running
        # and also the total number of years advanced...need to think about how to do this is a way that will be safe
        # this approach has major risks if you forget to update one of these variables
        self._totalWavesAdvanced = 0
        self._currentWave = 0
        self._bpTreatmentStrategy = None
        self.num_of_processes = 8

    def reset_to_baseline(self):
        self._totalWavesAdvanced = 0
        self._currentWave = 0
        self._bpTreatmentStrategy = None
        for person in self._people:
            person.reset_to_baseline()

    def advance(self, years):
        for yearIndex in range(years):
            print(f"processing year: {yearIndex}")
            self._currentWave += 1
            for person in self._people:
                self.advance_person(person)
            self.apply_recalibration_standards()
            self._totalWavesAdvanced += 1

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
            print(f"processing year: {i}")
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

    def apply_recalibration_standards(self):
        # treatment_standard is a dictionary of outcome types and effect sizees
        if (self._bpTreatmentStrategy is not None):
            if (self._bpTreatmentStrategy.get_treatment_recalibration_for_population() is not None):
                self.recalibrate_bp_treatment()

    # should the estiamted treatment effect be based on the number of events in the population
    # (i.e. # events treated / # of events untreated)
    # of should it be based on teh predicted reisks
    # the problem with the first approach is that its going to depend a lot on small sample sizes...
    # and we don't necessarily want to take out that random error...that random error reflects
    # genuine uncertainty.
    # so, i thikn it should be based on the model-predicted risks...

    def recalibrate_bp_treatment(self):
        treatment_outcome_standard = self._bpTreatmentStrategy.get_treatment_recalibration_for_population()
        # estimate risk for the people alive at the start of the wave
        recalibration_pop = self.get_people_alive_at_the_start_of_the_current_wave()
        treatedStrokeRisks, treatedMIRisks = self.estimate_risks(recalibration_pop)

        # rollback the treatment effect.
        # redtag: would like to apply to this to a deeply cloned population, but i can't get that to work
        # so, for now, applying it to the actual population and then rolling the effect back later.
        for _, person in recalibration_pop.iteritems():
            treatment_change_standard, _, effect_of_treatment_standard = self._bpTreatmentStrategy.get_changes_for_person(person)
            person._sbp[-1] = person._sbp[-1] - \
                effect_of_treatment_standard['_sbp'] * person._bpMedsAdded[-1]
            person._dbp[-1] = person._dbp[-1] - \
                effect_of_treatment_standard['_dbp'] * person._bpMedsAdded[-1]
            person._antiHypertensiveCount[-1] = person._antiHypertensiveCount[-1] - \
                treatment_change_standard['_antiHypertensiveCount'] * person._bpMedsAdded[-1]

        # estimate risk after applying the treamtent effect
        untreatedStrokeRisks, untreatedMIRisks = self.estimate_risks(recalibration_pop)

        # hacktag related to above — roll back the treatment effect...
        for _, person in recalibration_pop.iteritems():
            treatment_change_standard, _, effect_of_treatment_standard = self._bpTreatmentStrategy.get_changes_for_person(person)
            person._sbp[-1] = person._sbp[-1] + \
                effect_of_treatment_standard['_sbp'] * person._bpMedsAdded[-1]
            person._dbp[-1] = person._dbp[-1] + \
                effect_of_treatment_standard['_dbp'] * person._bpMedsAdded[-1]
            person._antiHypertensiveCount[-1] = person._antiHypertensiveCount[-1] + \
                treatment_change_standard['_antiHypertensiveCount'] * person._bpMedsAdded[-1]

        maxMeds = 5
        # recalibrate within each group of added medicaitons so that we can stratify the treamtnet effects
        for i in range(1, maxMeds+1):
            peopleWithBPMedsAdded = [((person._bpMedsAdded[-1] ==
                                       i) or (person._bpMedsAdded[-1] >= maxMeds)) for _, person in recalibration_pop.iteritems()]
            recalibrationPopForMedCount = recalibration_pop.loc[peopleWithBPMedsAdded]
            treatedStrokeRisksForMedCount = treatedStrokeRisks.loc[peopleWithBPMedsAdded]
            untreatedStrokeRisksForMedCount = untreatedStrokeRisks.loc[peopleWithBPMedsAdded]
            treatedMIRisksForMedCount = treatedMIRisks.loc[peopleWithBPMedsAdded]
            untreatedMIRisksForMedCount = untreatedMIRisks.loc[peopleWithBPMedsAdded]
            # the change standards are for a single medication
            recalibration_standard_for_med_count = treatment_outcome_standard.copy()
            for key, value in recalibration_standard_for_med_count.items():
                recalibration_standard_for_med_count[key] = value**i

            if len(recalibrationPopForMedCount) > 0:
                # recalibrate stroke
                self.create_or_rollback_events_to_correct_calibration(recalibration_standard_for_med_count,
                                                                    treatedStrokeRisksForMedCount,
                                                                    untreatedStrokeRisksForMedCount,
                                                                    OutcomeType.STROKE,
                                                                    CVOutcomeDetermination()._will_have_fatal_stroke,
                                                                    recalibrationPopForMedCount)

                # recalibrate MI
                self.create_or_rollback_events_to_correct_calibration(recalibration_standard_for_med_count,
                                                                    treatedMIRisksForMedCount,
                                                                    untreatedMIRisksForMedCount,
                                                                    OutcomeType.MI,
                                                                    CVOutcomeDetermination()._will_have_fatal_mi,
                                                                    recalibrationPopForMedCount)

    def estimate_risks(self, recalibration_pop):
        combinedRisks = pd.Series([self._outcome_model_repository.get_risk_for_person(
            person, OutcomeModelType.CARDIOVASCULAR, 1) for _, person in recalibration_pop.iteritems()])
        strokeProbabilities = pd.Series(
            [CVOutcomeDetermination().get_stroke_probability(person) for _, person in recalibration_pop.iteritems()])

        strokeRisks = combinedRisks * strokeProbabilities
        miRisks = combinedRisks * (1-strokeProbabilities)
        return strokeRisks, miRisks

    def create_or_rollback_events_to_correct_calibration(self,
                                                         treatment_outcome_standard,
                                                         treatedRisks,
                                                         untreatedRisks,
                                                         outcomeType,
                                                         fatalityDetermination,
                                                         recalibration_pop):
        modelEstimatedRR = treatedRisks.mean()/untreatedRisks.mean()
        # use the delta between that effect and the calibration standard to recalibrate the pop.
        delta = modelEstimatedRR - treatment_outcome_standard[outcomeType]
        eventsForPeople = [person.has_outcome_during_wave(
            self._currentWave, outcomeType) for _, person in recalibration_pop.iteritems()]
        numberOfEventStatusesToChange = abs(
            int(round(delta * pd.Series(eventsForPeople).sum()/modelEstimatedRR)))
        nonEventsForPeople = [not item for item in eventsForPeople]
        # key assumption: "treatment" is applied to a population as opposed to individuals within a population
        # analyses can be setup either way...build two populations and then set different treatments
        # or build a ur-population adn then set different treamtents within them
        # this is, i thikn, the first time where a coding decision is tied to one of those structure.
        # it would not, i think, be hard to change. but, just spelling it out here.

        # if negative, the model estimated too few events, if positive, too mnany
        if delta < 0:
            if numberOfEventStatusesToChange > 0:
                new_events = recalibration_pop.loc[nonEventsForPeople].sample(n=numberOfEventStatusesToChange,
                                                                              replace=False,
                                                                              weights=pd.Series(untreatedRisks).loc[nonEventsForPeople].values)
                for _, event in new_events.iteritems():
                    event.add_outcome_event(Outcome(outcomeType, fatalityDetermination(event)))

        # redtag - two problems here...1. rolling back events in people that may not have events
        # 2. probably usign the wrong weights...need to roll back inversely proportionately to the likeliood of an event, riht?

        elif delta > 0:
            if numberOfEventStatusesToChange > pd.Series(eventsForPeople).sum():
                numberOfEventStatusesToChange = pd.Series(eventsForPeople).sum()
            if numberOfEventStatusesToChange > 0:
                events_to_rollback = recalibration_pop.loc[eventsForPeople].sample(n=numberOfEventStatusesToChange,
                                                                                   replace=False,
                                                                                   weights=pd.Series(1-untreatedRisks).loc[eventsForPeople].values)
                for _, event in events_to_rollback.iteritems():
                    event.rollback_most_recent_event(outcomeType)

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
                      selfReportMIAge=None)

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
            lambda x: x._outcomes[outcomeType][0][0] - x._age[0] + 1,
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
        popDF = self.get_people_current_state_as_dataframe()
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

    def calculate_mean_age_sex_standardized_event(self, eventSelector, eventAgeIdentifier,
                                                  yearOfStandardizedPopulation=2016,
                                                  subPopulationSelector=None,
                                                  subPopulationDFSelector=None):
        # calculated standardized event rate for each year
        popDF = get_events_for_event_type(eventSelector, eventAgeIdentifier,
                                          subPopulationSelector=None,
                                          subPopulationDFSelector=None)
        eventsPerYear = []

        for year in range(1, self._totalWavesAdvanced + 1):
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
        peopleDF['ageGroup'] = (peopleDF['age'] // 5) + 1
        peopleDF['ageGroup'] = peopleDF['ageGroup'].astype(int)
        # tabulate events by group
        eventsByGroup = peopleDF.groupby(['ageGroup', 'female'])['event'].sum()
        personYears = peopleDF.groupby(['ageGroup', 'female'])['age'].count()
        # set those events on the age standard
        ageStandard['outcomeCount'] = eventsByGroup
        ageStandard['simPersonYears'] = personYears

        ageStandard = self.tabulate_age_specific_rates(ageStandard)
        return((ageStandard.ageSpecificContribution.sum(), ageStandard.outcomeCount.sum()))

    def get_people_current_state_as_dataframe(self):
        return pd.DataFrame({'age': [person._age[-1] for person in self._people],
                             'baseAge': [person._age[0] for person in self._people],
                             'gender': [person._gender for person in self._people],
                             'raceEthnicity': [person._raceEthnicity for person in self._people],
                             'sbp': [person._sbp[-1] for person in self._people],
                             'dbp': [person._dbp[-1] for person in self._people],
                             'a1c': [person._a1c[-1] for person in self._people],
                             'hdl': [person._hdl[-1] for person in self._people],
                             'ldl': [person._ldl[-1] for person in self._people],
                             'trig': [person._trig[-1] for person in self._people],
                             'totChol': [person._totChol[-1] for person in self._people],
                             'bmi': [person._bmi[-1] for person in self._people],
                             'anyPhysicalActivity': [person._anyPhysicalActivity[-1] for person in self._people],
                             'education': [person._education.value for person in self._people],
                             'aFib': [person._afib[-1] for person in self._people],
                             'antiHypertensive': [person._antiHypertensiveCount[-1] for person in self._people],
                             'statin': [person._statin[-1] for person in self._people],
                             'otherLipidLoweringMedicationCount': [person._otherLipidLoweringMedicationCount[-1] for person in self._people],
                             'waist': [person._waist[-1] for person in self._people],
                             'smokingStatus': [person._smokingStatus for person in self._people],
                             'dead': [person.is_dead() for person in self._people],
                             'miPriorToSim': [person._selfReportMIPriorToSim for person in self._people],
                             'miInSim': [person.has_mi_during_simulation() for person in self._people],
                             'strokePriorToSim': [person._selfReportStrokePriorToSim for person in self._people],
                             'strokeInSim': [person.has_stroke_during_simulation() for person in self._people],
                             'totalYearsInSim': [person.years_in_simulation() for person in self._people]})

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
                             'aFib': [person._afib[0] for person in self._people],
                             'antiHypertensive': [person._antiHypertensiveCount[0] for person in self._people],
                             'statin': [person._statin[0] for person in self._people],
                             'otherLipidLoweringMedicationCount': [person._otherLipidLoweringMedicationCount[0] for person in self._people],
                             'waist': [person._waist[0] for person in self._people],
                             'smokingStatus': [person._smokingStatus for person in self._people],
                             'miPriorToSim': [person._selfReportMIPriorToSim for person in self._people],
                             'strokePriorToSim': [person._selfReportStrokePriorToSim for person in self._people]})


def initializeAFib(person):
    model = load_regression_model("BaselineAFibModel")
    statsModel = StatsModelLogisticRiskFactorModel(model)
    return statsModel.estimate_next_risk(person)


def build_person(x):
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
        selfReportStrokeAge=x.selfReportStrokeAge,
        selfReportMIAge=x.selfReportMIAge,
        dfIndex=x.index,
        diedBy2015=x.diedBy2015)


def build_people_using_nhanes_for_sampling(nhanes, n, filter=None, random_seed=None):
    repeated_sample = nhanes.sample(
        n,
        weights=nhanes.WTINT2YR,
        random_state=random_seed,
        replace=True)
    # people = repeated_sample.apply(build_person, axis=1)
    people = parallelize_on_rows(repeated_sample, build_person)
    if filter is not None:
        people = people.loc[people.apply(filter)]

    return people

# from https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply


def parallelize(data, func, number_of_processes=8):
    data_split = np.array_split(data, number_of_processes)
    pool = mp.Pool(number_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, number_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), number_of_processes)


class NHANESDirectSamplePopulation(Population):
    """ Simple base class to sample with replacement from 2015/2016 NHANES """

    def __init__(
            self,
            n,
            year,
            filter=None,
            generate_new_people=True,
            model_reposistory_type="cohort",
            random_seed=None):
        nhanes = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        nhanes = nhanes.loc[nhanes.year == year]
        super().__init__(build_people_using_nhanes_for_sampling(
            nhanes, n, filter=filter, random_seed=random_seed))
        self.n = n
        self.year = year
        self._initialize_risk_models(model_reposistory_type)
        self._outcome_model_repository = OutcomeModelRepository()
        self._qaly_assignment_strategy = QALYAssignmentStrategy()

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
