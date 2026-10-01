import pandas as pd

from microsim.common.data_loader import get_absolute_datafile_path
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.risk_factor import StaticRiskFactorsType

class StandardizedPopulation:
    '''This class will represent distributions of standardized populations.
       These distributions are useful in estimating expected standardized outcomes from the simulation population
       which in general may have a different distribution than a standardized population.
       The benefit of using such a class is that you can swap different standardized populations 
       on a single simulation population to estimate expected standardized outcomes for more than one standardized population.
       ageStandard: a Pandas dataframe with information about the standardized population for several years
       ageGroups: a dictionary with information about how ages are distributed in age groups for each gender
                  key: gender, value: [ [0], [1,2,3,4], [5,6...], ... ]
       populationPercents: a dictionary with information about what percentage that age group represents of the entire population (all genders)
                  key: gender, value: [ 0.01, 0.01,     0.01,... ]'''

    #The standard population file is 388 MB of 14.5 million rows and parsing it takes about 4s, all
    #of it inside _parse_age_standard, while what a year needs out of it is 38 rows. Every
    #Population.calculate_mean_age_sex_standardized_incidence builds a StandardizedPopulation, and
    #get_cv_standardized_rates builds 18 of them (6 outcomes over 3 subgroups), so a single report
    #used to spend about 150s parsing the same year over and over. Each year is parsed once here.
    _ageStandardByYear = dict()

    def __init__(self, year=2016):
        self.year = year
        self.ageStandard = self.build_age_standard()
        self.ageGroups = self.get_age_groups()
        self.populationPercents = self.get_population_percents()
        self.populationWeightedStandard = self.get_population_weighted_standard()

    def get_population_weighted_standard(self):
        rows = []
        for age in range(1, 86): #NHANES top-codes age at 85, so age 85 stands in for the whole 85+ group
            for gender in range(1, 3):
                dfRow = self.ageStandard.loc[
                    (age >= self.ageStandard.lowerAgeBound)
                    & (age <= self.ageStandard.upperAgeBound)
                    & (self.ageStandard.gender == gender)
                ]
                #the 150 bound is a membership sentinel, not a real width: count only sampleable ages
                upperAge = min(dfRow["upperAgeBound"].values[0], 85)
                lowerAge = dfRow["lowerAgeBound"].values[0]
                totalPop = dfRow["standardPopulation"].values[0]
                rows.append(
                    {"age": age, "gender": gender, "pop": totalPop / (upperAge - lowerAge + 1)}
                )
        df = pd.DataFrame(rows)
        df["popWeight"] = df["pop"] / df["pop"].sum()
        return df

    def build_age_standard(self):
        '''Returns the standard population of self.year: standardPopulation by age group and
           gender. Parsed on the first call for that year (see _parse_age_standard) and cached for
           every later one, and handed out as a copy, so that a caller which adds or edits a column
           cannot corrupt the cache - the same convention as PopulationFactory.get_nhanesDf.'''
        if self.year not in StandardizedPopulation._ageStandardByYear:
            StandardizedPopulation._ageStandardByYear[self.year] = self._parse_age_standard()
        return StandardizedPopulation._ageStandardByYear[self.year].copy()

    def _parse_age_standard(self):
        '''Reads the standard population file and reduces it to the age group and gender rows of
           self.year. Slow, and so called once per year, through build_age_standard.'''
        datafile_path = get_absolute_datafile_path("us.1969_2017.19ages.adjusted.txt")
        ageStandard = pd.read_csv(datafile_path, header=0, names=["raw"])
        # https://seer.cancer.gov/popdata/popdic.html
        ageStandard["year"] = ageStandard["raw"].str[0:4]
        ageStandard["year"] = ageStandard.year.astype(int)
        # format changes in 1990...so, we'll go forward from there...
        # the year is selected here rather than further below so that every column read out of raw
        # below is read for the ~334,000 rows of one year instead of the 8.3 million rows of all of
        # them, which is half of the time this function takes
        ageStandard = ageStandard.loc[(ageStandard.year >= 1990) & (ageStandard.year == self.year)]
        ageStandard = ageStandard.copy()
        ageStandard["female"] = ageStandard["raw"].str[15:16]
        ageStandard["female"] = ageStandard["female"].astype(int)
        ageStandard["female"] = ageStandard["female"].replace({1: 0, 2: 1})
        ageStandard["ageGroup"] = ageStandard["raw"].str[16:18]
        ageStandard["ageGroup"] = ageStandard["ageGroup"].astype(int)
        ageStandard["standardPopulation"] = ageStandard["raw"].str[18:26]
        ageStandard["standardPopulation"] = ageStandard["standardPopulation"].astype(int)
        ageStandard["lowerAgeBound"] = (ageStandard.ageGroup - 1) * 5
        ageStandard["upperAgeBound"] = (ageStandard.ageGroup * 5) - 1
        ageStandard["lowerAgeBound"] = ageStandard["lowerAgeBound"].replace({-5: 0, 0: 1})
        ageStandard["upperAgeBound"] = ageStandard["upperAgeBound"].replace({-1: 0, 89: 150})
        #the rows are those of self.year already, they were selected before any of them was read
        ageStandardGroupby = ageStandard[
            ["female", "standardPopulation", "lowerAgeBound", "upperAgeBound", "ageGroup"]
        ].groupby(["ageGroup", "female"])
        ageStandardHeaders = ageStandardGroupby.first()[["lowerAgeBound", "upperAgeBound"]]
        ageStandardHeaders["female"] = ageStandardHeaders.index.get_level_values(1)
        ageStandardPopulation = ageStandard[["female", "standardPopulation", "ageGroup"]]
        ageStandardPopulation = ageStandardPopulation.groupby(["ageGroup", "female"]).sum()
        ageStandardPopulation = ageStandardHeaders.join(ageStandardPopulation, how="inner")

        #TO DO: probably I am undoing here what happend above, I could clean this up sometime....
        ageStandardPopulation = ageStandardPopulation.droplevel("female").reset_index(level="ageGroup")
        #instead of using the female flag, use the NHANESGender values
        genderDict = {0: NHANESGender.MALE.value, 1: NHANESGender.FEMALE.value}
        ageStandardPopulation[StaticRiskFactorsType.GENDER.value] = ageStandardPopulation["female"].replace(genderDict)
        ageStandardPopulation.drop("female", inplace=True, axis=1)

        return ageStandardPopulation

    def get_age_groups(self):
        ageGroups = dict()
        for gender in NHANESGender:
            ageStandardForGender = self.ageStandard.loc[self.ageStandard["gender"]==gender.value]
            ageGroups[gender.value] = (ageStandardForGender
                                       .apply(lambda x: [i for i in range(x["lowerAgeBound"],x["upperAgeBound"]+1)], axis=1)).to_list()
        return ageGroups

    def get_population_percents(self):
        standardPopulation = dict()
        standardPopulationPercent = dict()
        for gender in NHANESGender:
            ageStandardForGender = self.ageStandard.loc[self.ageStandard["gender"]==gender.value]
            standardPopulation[gender.value] = ageStandardForGender["standardPopulation"].to_list()

        #find the size of the standard population, which includes all genders
        standardPopulationSum = sum([sum(standardPopulation[gender.value]) for gender in NHANESGender])
        for gender in NHANESGender:
            standardPopulationPercent[gender.value] = [x/standardPopulationSum for x in standardPopulation[gender.value]]
        return standardPopulationPercent
