import pandas as pd

from microsim.data_loader import get_absolute_datafile_path
from microsim.gender import NHANESGender
from microsim.risk_factor import StaticRiskFactorsType

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
    def __init__(self, year=2016):
        self.year = year
        self.ageStandard = self.build_age_standard()
        self.ageGroups = self.get_age_groups()
        self.populationPercents = self.get_population_percents()

    def build_age_standard(self):

        datafile_path = get_absolute_datafile_path("us.1969_2017.19ages.adjusted.txt")
        ageStandard = pd.read_csv(datafile_path, header=0, names=["raw"])
        # https://seer.cancer.gov/popdata/popdic.html
        ageStandard["year"] = ageStandard["raw"].str[0:4]
        ageStandard["year"] = ageStandard.year.astype(int)
        # format changes in 1990...so, we'll go forward from there...
        ageStandard = ageStandard.loc[ageStandard.year >= 1990]
        ageStandard["state"] = ageStandard["raw"].str[4:6]
        ageStandard["state"] = ageStandard["raw"].str[4:6]
        # 1 = white, 2 = black, 3 = american indian/alaskan, 4 = asian/pacific islander
        ageStandard["race"] = ageStandard["raw"].str[13:14]
        ageStandard["hispanic"] = ageStandard["raw"].str[14:15]
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
        ageStandardYear = ageStandard.loc[ageStandard.year == self.year]
        ageStandardGroupby = ageStandardYear[
            ["female", "standardPopulation", "lowerAgeBound", "upperAgeBound", "ageGroup"]
        ].groupby(["ageGroup", "female"])
        ageStandardHeaders = ageStandardGroupby.first()[["lowerAgeBound", "upperAgeBound"]]
        ageStandardHeaders["female"] = ageStandardHeaders.index.get_level_values(1)
        ageStandardPopulation = ageStandardYear[["female", "standardPopulation", "ageGroup"]]
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
