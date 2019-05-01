from mcm.person import Person
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.gender import NHANESGender
from mcm.cohort_risk_model_repository import CohortRiskModelRepository
from mcm.nhanes_risk_model_repository import NHANESRiskModelRepository
from mcm.outcome_model_repository import OutcomeModelRepository

import pandas as pd
import os
import copy


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
        self._ageStandards = {}
        self._totalYearsAdvanced = 0

    def advance(self, years):
        for _ in range(years):
            for person in self._people:
                if not person.is_dead():
                    person.advance_year(self._risk_model_repository,
                                        self._outcome_model_repository)
            self.apply_recalibration_standards()
        self._totalYearsAdvanced += years

    def apply_recalibration_standards(self):
        pass

    # refactorrtag: we should probably build a specific class that loads data files...
    def build_age_standard(self, yearOfStandardizedPopulation):
        if yearOfStandardizedPopulation in Population._ageStandards:
            return copy.deepcopy(Population._ageStandards[yearOfStandardizedPopulation])

        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        model_spec_path = os.path.normpath(
            os.path.join(
                abs_module_path,
                "./data/",
                "us.1969_2017.19ages.adjusted.txt"))

        ageStandard = pd.read_csv(model_spec_path, header=0, names=['raw'])
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
        ageStandard['ageSpecificRate'] = ageStandard['outcomeCount'] * 100000 / \
            ageStandard['simPersonYears']
        ageStandard['ageSpecificContribution'] = ageStandard['ageSpecificRate'] * \
            ageStandard['percentStandardPopInGroup']
        return ageStandard

    # return the age standardized # of events per 100,000 person years
    def calculate_mean_age_sex_standardized_incidence(
            self, outcomeType, yearOfStandardizedPopulation=2016, subPopulationSelector=None):

        # the age selector picks the first outcome (_outcomes(outcomeTYpe)[0]) and the age is the
        # first element within the returned tuple (the second [0])
        events = self.calculate_mean_age_sex_standardized_event(
            lambda x: x.has_outcome_during_simulation(outcomeType),
            lambda x: x._outcomes[outcomeType][0][0] - x._age[0]+1,
            yearOfStandardizedPopulation)
        return (pd.Series([event[0] for event in events]).mean(),
                pd.Series([event[1] for event in events]).sum())

    def calculate_mean_age_sex_standardized_mortality(self, yearOfStandardizedPopulation=2016):
        events = self.calculate_mean_age_sex_standardized_event(lambda x: x.is_dead(),
                                                                lambda x: x.years_in_simulation(),
                                                                yearOfStandardizedPopulation)
        return pd.Series([event[0] for event in events]).mean()

    def calculate_mean_age_sex_standardized_event(self, eventSelector, eventAgeIdentifier,
                                                  yearOfStandardizedPopulation=2016):
        # build a dataframe to represent the population
        popDF = pd.DataFrame({"index": self._people.index,
                              "baseAge": [person._age[0] for person in self._people],
                              "female": [person._gender - 1 for person in self._people]
                              })

        eventsPerYear = []
        # calculated standardized event rate for each year
        for year in range(1, self._totalYearsAdvanced + 1):
            eventVarName = 'event' + str(year)
            ageVarName = 'age' + str(year)
            popDF[ageVarName] = popDF['baseAge'] + year
            popDF[eventVarName] = [eventSelector(person) and eventAgeIdentifier(
                person) == year for person in self._people]
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
                             'smokingStatus': [person._smokingStatus for person in self._people],
                             'dead': [person.is_dead() for person in self._people],
                             'miPriorToSim': [person.has_mi_prior_to_simulation() for person in self._people],
                             'miInSim': [person.has_mi_during_simulation() for person in self._people],
                             'strokePriorToSim': [person.has_stroke_prior_to_simulation() for person in self._people],
                             'strokeInSim': [person.has_stroke_during_simulation() for person in self._people],
                             'totalYearsInSim': [person.years_in_simulation() for person in self._people]})


def build_people_using_nhanes_for_sampling(nhanes, n, random_seed=None):
    repeated_sample = nhanes.sample(
        n, weights=nhanes.WTINT2YR, random_state=random_seed, replace=True)
    people = repeated_sample.apply(
        lambda x: Person(
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
            smokingStatus=SmokingStatus(int(x.smokingStatus)),
            selfReportStrokeAge=x.selfReportStrokeAge,
            selfReportMIAge=x.selfReportMIAge,
            dfIndex=x.index,
            diedBy2015=x.diedBy2015), axis=1)
    return people


class NHANESDirectSamplePopulation(Population):
    """ Simple base class to sample with replacement from 2015/2016 NHANES """

    def __init__(self, n, year, model_reposistory_type="cohort", random_seed=None):
        nhanes = pd.read_stata("mcm/data/fullyImputedDataset.dta")
        nhanes = nhanes.loc[nhanes.year == year]
        super().__init__(build_people_using_nhanes_for_sampling(
            nhanes, n, random_seed=random_seed))
        self.n = n
        self._initialize_risk_models(model_reposistory_type)
        self._outcome_model_repository = OutcomeModelRepository()

    def _initialize_risk_models(self, model_repository_type):
        if (model_repository_type == "cohort"):
            self._risk_model_repository = CohortRiskModelRepository()
        elif (model_repository_type == "nhanes"):
            self._risk_model_repository = NHANESRiskModelRepository()
        else:
            raise Exception('unknwon risk model repository type' + model_repository_type)
