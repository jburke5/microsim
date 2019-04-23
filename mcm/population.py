from mcm.person import Person
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.gender import NHANESGender
from mcm.cohort_risk_model_repository import CohortRiskModelRepository
from mcm.nhanes_risk_model_repository import NHANESRiskModelRepository
from mcm.outcome_model_repository import OutcomeModelRepository

import pandas as pd
import os


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
            return Population._ageStandards[yearOfStandardizedPopulation].copy()
        
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
        Population._ageStandards[yearOfStandardizedPopulation] = ageStandardPopulation
        ageStandardPopulation['outcomeCount'] = 0
        ageStandardPopulation['simPersonYears'] = 0

        return ageStandardPopulation

    def assign_incident_outcomes(self, ageStandard, outcomeType):
        for person in self._people:
            if len(person._outcomes[outcomeType]) > 0:
                ageStandard.loc[((ageStandard['lowerAgeBound'] <= person._age[0]) &
                                 (ageStandard['upperAgeBound'] >= person._age[0]) &
                                 (ageStandard['female'] == (person._gender == NHANESGender.FEMALE))),
                                'outcomeCount'] += 1
            ageStandard.loc[((ageStandard['lowerAgeBound'] <= person._age[0]) &
                             (ageStandard['upperAgeBound'] >= person._age[0]) &
                             (ageStandard['female'] == (person._gender == NHANESGender.FEMALE))),
                            'simPersonYears'] += person.years_in_simulation()
        return ageStandard

    def build_age_sex_standardized_dataframe(self, outcomeType, yearOfStandardizedPopulation=2016):
        ageStandard = self.build_age_standard(yearOfStandardizedPopulation)
        ageStandard = self.assign_incident_outcomes(ageStandard, outcomeType)
        ageStandard = self.tabulate_age_specific_rates(ageStandard)
        return ageStandard

    def tabulate_age_specific_rates(self, ageStandard):
        ageStandard['percentStandardPopInGroup'] = ageStandard['standardPopulation'] / \
            (ageStandard['standardPopulation'].sum())
        ageStandard['ageSpecificRate'] = ageStandard['outcomeCount'] * 100000 / \
            ageStandard['simPersonYears'] 
        ageStandard['ageSpecificContribution'] = ageStandard['ageSpecificRate'] * \
            ageStandard['percentStandardPopInGroup']
        # should the < 18 group be included? not sure if they're included in formal incidence calcs
        # we'll slightly underestimate if we include them because they're not in the sim pop...
        #ageStandard = ageStandard.loc[ageStandard.lowerAgeBound >= 18]
        return ageStandard

    # return the age standardized # of events per 100,000 person years
    def calculate_age_sex_standardized_incidence(
            self, outcomeType, yearOfStandardizedPopulation=2016):

        df = self.build_age_sex_standardized_dataframe(outcomeType, yearOfStandardizedPopulation)
        return (df['ageSpecificContribution'].sum(), df['outcomeCount'].sum(), df)

    def calculate_age_sex_standardized_mortality(self, yearOfStandardizedPopulation=2016):
        ageStandard = self.build_age_standard(yearOfStandardizedPopulation)

        for person in self._people:
            if person.is_dead():
                ageStandard.loc[((ageStandard['lowerAgeBound'] <= person._age[0]) &
                                 (ageStandard['upperAgeBound'] >= person._age[0]) &
                                 (ageStandard['female'] == (person._gender == NHANESGender.FEMALE))),
                                'outcomeCount'] += 1
            ageStandard.loc[((ageStandard['lowerAgeBound'] <= person._age[0]) &
                             (ageStandard['upperAgeBound'] >= person._age[0]) &
                             (ageStandard['female'] == (person._gender == NHANESGender.FEMALE))),
                            'simPersonYears'] += person.years_in_simulation()
        ageStandard = self.tabulate_age_specific_rates(ageStandard)
        return (ageStandard['ageSpecificContribution'].sum(), ageStandard['outcomeCount'].sum(), ageStandard)


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
