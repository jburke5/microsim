import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
from enum import Enum

from microsim.person_factory import PersonFactory
from microsim.person_filter_factory import PersonFilterFactory
from microsim.population import Population
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.population_model_repository import PopulationModelRepository, PopulationRepositoryType
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository, 
                                                   CohortStaticRiskFactorModelRepository,
                                                   CohortDefaultTreatmentModelRepository)
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.race_ethnicity import RaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.treatment import DefaultTreatmentsType
from microsim.alcohol_category import AlcoholCategory
from microsim.standardized_population import StandardizedPopulation
from microsim.variable_type import VariableType
from microsim.outcome import OutcomeType
from microsim.population_type import PopulationType
from microsim.modality import Modality

class PopulationFactory:
    nhanes_pop_attributes = {PopulationRepositoryType.STATIC_RISK_FACTORS.value: 
                                                                    [StaticRiskFactorsType.GENDER.value,
                                                                     StaticRiskFactorsType.SMOKING_STATUS.value, 
                                                                     StaticRiskFactorsType.RACE_ETHNICITY.value,
                                                                     StaticRiskFactorsType.EDUCATION.value,
                                                                     StaticRiskFactorsType.MODALITY.value],
                             PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value: 
                                                                     [DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
                                                                      DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                                                                      DynamicRiskFactorsType.AGE.value, 
                                                                      DynamicRiskFactorsType.HDL.value, 
                                                                      DynamicRiskFactorsType.BMI.value, 
                                                                      DynamicRiskFactorsType.TOT_CHOL.value, 
                                                                      DynamicRiskFactorsType.TRIG.value, 
                                                                      DynamicRiskFactorsType.A1C.value, 
                                                                      DynamicRiskFactorsType.LDL.value, 
                                                                      DynamicRiskFactorsType.WAIST.value, 
                                                                      DynamicRiskFactorsType.CREATININE.value, 
                                                                      DynamicRiskFactorsType.SBP.value, 
                                                                      DynamicRiskFactorsType.DBP.value],
                             PopulationRepositoryType.DEFAULT_TREATMENTS.value: 
                                                                     [DefaultTreatmentsType.STATIN.value,
                                                                       DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value],
                             PopulationRepositoryType.OUTCOMES.value: 
                                                                      [OutcomeType.COGNITION.value,
                                                                       OutcomeType.CI.value,
                                                                       OutcomeType.CARDIOVASCULAR.value,
                                                                       OutcomeType.STROKE.value,
                                                                       OutcomeType.MI.value,
                                                                       OutcomeType.NONCARDIOVASCULAR.value,
                                                                       OutcomeType.DEMENTIA.value,
                                                                       OutcomeType.DEATH.value,
                                                                       OutcomeType.QUALITYADJUSTED_LIFE_YEARS.value]}
                                                  
    #these are used below to define groups ( = specific combinations of all NHANES categorical variables)
    # and to define which columns from the NHANES dataframe to model as Gaussians ( = all continuous variables present
    # in the original NHANES dataset).
    # The last point is important, the Gaussians model the continuous variables present in the original NHANES dataset
    # not all continuous variables present in the Microsim simulation (which includes more continuous variables not
    # present in the original NHANES dataset such as PVD)
    # The order of these two lists is important,as they define the column names of the final dataframe. The numpy arrays used in between do 
    # not keep track of which column is which attribute.
    nhanes_variable_types = {VariableType.CATEGORICAL.value:  [
                                                  StaticRiskFactorsType.MODALITY.value,
                                                  StaticRiskFactorsType.GENDER.value, 
                                                  StaticRiskFactorsType.SMOKING_STATUS.value, 
                                                  StaticRiskFactorsType.RACE_ETHNICITY.value, 
                                                  DefaultTreatmentsType.STATIN.value,
                                                  StaticRiskFactorsType.EDUCATION.value,
                                                  DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
                                                  DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                                                  DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value],
                             VariableType.CONTINUOUS.value:   [DynamicRiskFactorsType.AGE.value, 
                                                  DynamicRiskFactorsType.HDL.value, 
                                                  DynamicRiskFactorsType.BMI.value, 
                                                  DynamicRiskFactorsType.TOT_CHOL.value, 
                                                  DynamicRiskFactorsType.TRIG.value, 
                                                  DynamicRiskFactorsType.A1C.value, 
                                                  DynamicRiskFactorsType.LDL.value, 
                                                  DynamicRiskFactorsType.WAIST.value, 
                                                  DynamicRiskFactorsType.CREATININE.value, 
                                                  DynamicRiskFactorsType.SBP.value, 
                                                  DynamicRiskFactorsType.DBP.value]}
    #the order of the items in the two lists is critical because functions later on, eg draw from the distributions, depend on the order
    kaiser_variable_types = {VariableType.CATEGORICAL.value: [StaticRiskFactorsType.MODALITY.value,
                                                      StaticRiskFactorsType.GENDER.value, 
                                                      StaticRiskFactorsType.RACE_ETHNICITY.value, 
                                                      StaticRiskFactorsType.SMOKING_STATUS.value, 
                                                      DynamicRiskFactorsType.AFIB.value, 
                                                      DynamicRiskFactorsType.PVD.value, 
                                                      DefaultTreatmentsType.STATIN.value,
                                                      DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value],
                     VariableType.CONTINUOUS.value: [DynamicRiskFactorsType.AGE.value, 
                                                     DynamicRiskFactorsType.HDL.value, 
                                                     DynamicRiskFactorsType.A1C.value, 
                                                     DynamicRiskFactorsType.TOT_CHOL.value, 
                                                     DynamicRiskFactorsType.LDL.value, 
                                                     DynamicRiskFactorsType.TRIG.value, 
                                                     DynamicRiskFactorsType.CREATININE.value, 
                                                     DynamicRiskFactorsType.SBP.value, 
                                                     DynamicRiskFactorsType.DBP.value,
                                                     DynamicRiskFactorsType.BMI.value, 
                                                     DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]}

    @staticmethod
    def variable_types(varType=VariableType.CATEGORICAL.value, popType=PopulationType.NHANES.value):
        if popType==PopulationType.NHANES.value:
            return PopulationFactory.nhanes_variable_types[varType]
        elif popType==PopulationType.KAISER.value:
            return PopulationFactory.kaiser_variable_types[varType]
        else:
            raise RuntimeError("Unrecognized population type in PopulationFactory.variable_types.")       

    @staticmethod
    def get_pop_attributes(popType=PopulationType.NHANES.value):
        if popType == PopulationType.NHANES.value:
            return PopulationFactory.nhanes_pop_attributes 
        elif popType == PopulationType.KAISER.value:
            return PopulationFactory.kaiser_pop_attributes
        else:
            raise RuntimeError("Population type not a valid one in PopulationFactory.get_pop_attributes.")      

    @staticmethod
    def get_population(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population(**kwargs)
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_population(**kwargs)
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population function.")

    @staticmethod
    def get_people(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_people(**kwargs)
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_people(**kwargs)
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_people function.")

    @staticmethod
    def get_population_model_repo(popType):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population_model_repo()
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_nhanes_population_model_repo()
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population_model_repo function.")

    @staticmethod
    def set_index_in_people(people, start=0):
        """Once people are created, its Person-objects do not have a unique index.
           This function assigns a unique index to every Person-object in people."""
        list(map(lambda person, i: setattr(person, "_index", i+start), people, range(people.shape[0])))

    @staticmethod
    def get_nhanesDf():
        """Reads and modifies the NHANES dataframe so that it is ready to be used in the simulation.
           Returns a Pandas df with the NHANES information as exists in Microsim."""
        nhanesDf = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        #in Person-objects, the attribute name is used
        nhanesDf = nhanesDf.rename(columns={"level_0":"name"})
        #rename the columns that have different column names than the ones that appear in Microsim
        nhanesDf = PopulationFactory.rename_df_columns(nhanesDf, PersonFactory.microsimToNhanes)
        #convert the integers to booleans because in the simulation we always use bool for this rf
        nhanesDf[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value] = nhanesDf[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)
        #convert drinks per week to category
        nhanesDf[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value] = nhanesDf.apply(lambda x:
                                                                                 AlcoholCategory.get_category_for_consumption(x[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]), axis=1)
        return nhanesDf

    @staticmethod
    def get_kaiserDf(csvFile):
        """Reads and modifies the Kaiser file so that it is ready to be used in the simulation.
           Returns a Pandas df with the Kaiser information as named in Microsim."""
        df = pd.read_csv(csvFile).dropna()
        #TO DO: needs to be FIXED, or REMOVED
        #df = df.loc[ (df["AHL_nonStatin"]==0) ]
        #df = df.drop("AHL_nonStatin", axis=1)
        #if 'weight' in df.columns:
        #    df = df.drop('weight', axis=1)
        df = PopulationFactory.rename_df_columns(df, PersonFactory.microsimToKaiser)
        df = df.astype({StaticRiskFactorsType.SMOKING_STATUS.value: 'int',
                        DynamicRiskFactorsType.AFIB.value:'bool',
                        DynamicRiskFactorsType.PVD.value:'bool',
                        DefaultTreatmentsType.STATIN.value:'int', 
                        DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value:'bool',
                        #"age":'int'}).reset_index()
                       }).reset_index()
        df[StaticRiskFactorsType.GENDER.value] = df[StaticRiskFactorsType.GENDER.value].replace({'F': 2, 'M': 1}).astype('int') #.infer_objects(copy=False)  
        df[StaticRiskFactorsType.RACE_ETHNICITY.value] = df[StaticRiskFactorsType.RACE_ETHNICITY.value].replace(
                                        {'Black': RaceEthnicity.NON_HISPANIC_BLACK.value, 
                                        'Asian and Pacific Islander': RaceEthnicity.ASIAN.value,
                                        'White': RaceEthnicity.NON_HISPANIC_WHITE.value,
                                        'Multiple/Other/Unknown': RaceEthnicity.OTHER.value,
                                        'Hispanic': RaceEthnicity.OTHER_HISPANIC.value}).astype('int') 
        df[StaticRiskFactorsType.MODALITY.value] = df[StaticRiskFactorsType.MODALITY.value].replace({"CT": Modality.CT.value,
                                                                                                     "MR": Modality.MR.value})
        return df

    @staticmethod
    def rename_df_columns(df, microsimToDfDict):
        '''Dataframes that we typically use to import person data, eg NHANES, have column names that are different than microsim attributes.
        This function takes a dictionary that helps convert those column names to the exact names that microsim uses.'''
        for key, value in microsimToDfDict.items():
            if key!=value:
                df = df.rename(columns={value:key})
        return df

    @staticmethod
    def get_nhanes_people(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None):
        '''Returns a Pandas Series object with Person-Objects of all persons included in NHANES for year 
           with or without sampling. Filters are applied prior to sampling in order to maximize efficiency and minimize
           memory utilization. This does not affect the distribution of the relative percentages of groups 
           represented in people.
           The flag distributions controls if the Person-objects will come directly from the NHANES data or
           if Gaussian distributions will first be fit to the NHANES data and then draws are obtained from the distributions.'''
        nhanesDf = PopulationFactory.get_nhanesDf()        

        if year is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]
        nhanesDf = PopulationFactory.apply_person_filters_on_df(personFilters, nhanesDf)        
 
        #if we want to draw from the NHANES distributions, then we fit the NHANES data first, draw, convert the draws to 
        #a Pandas dataframe, bring in the NHANES weights (because I do not keep those when I do the fits)
        #and then have nhanesDf point to the df obtained from the draws
        if distributions:
            dfForGroups = PopulationFactory.get_partitioned_nhanes_people(year=year)
            distributions = PopulationFactory.get_distributions(dfForGroups)
            drawsForGroups, namesForGroups = PopulationFactory.draw_from_distributions(distributions)
            df = PopulationFactory.get_df_from_draws(drawsForGroups, namesForGroups, popType=PopulationType.NHANES.value)
            df = df.merge(nhanesDf[["name","WTINT2YR"]], on="name", how="inner").copy()
            nhanesDf = df

        if nhanesWeights & (customWeights is not None):
            raise RuntimeError("Cannot use both nhanesWeights (nhanesWeights=True) and custom weights (customWeights is not None).")

        if nhanesWeights:
            if (year is None) | (n is None):
                raise RuntimeError("""Cannot set nhanesWeights True without specifying a year and n.
                                    NHANES weights are defined for each year independently and for sampling 
                                    to occur the sampling size is needed.""")
            else:
                weights = nhanesDf.WTINT2YR
                nhanesDfForPeople = nhanesDf.sample(n, weights=weights, replace=True)
        elif customWeights is not None:
            nhanesDfForPeople = nhanesDf.sample(n, weights=customWeights, replace=True)
        else:
            nhanesDfForPeople = nhanesDf

        people = pd.DataFrame.apply(nhanesDfForPeople, PersonFactory.get_nhanes_person, axis="columns")

        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)

        if nhanesWeights:
            people = PopulationFactory.bring_people_to_target_n(n, people, nhanesDf, personFilters, popType=PopulationType.NHANES.value)
            
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_nhanes_population_model_repo():
        """Return the default, self-consistent set of models for advancing an NHANES Population."""
        return PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                         CohortDefaultTreatmentModelRepository(),
                                         OutcomeModelRepository(),
                                         CohortStaticRiskFactorModelRepository())

    @staticmethod    
    def get_nhanes_population(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.
           Person attributes can originate either from the NHANES dataset directly or from distributions fit to the NHANES dataset.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, personFilters=personFilters, nhanesWeights=nhanesWeights, distributions=distributions,customWeights=customWeights)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)

    @staticmethod
    def get_kaiser_population(n=1000, personFilters=None):
        people = PopulationFactory.get_kaiser_people(n=n, personFilters=personFilters)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)

    @staticmethod
    def get_partitioned_nhanes_people(year=None):
        """Partitions a NHANES df in all possible combinations of categorical variables that actually exist in NHANES.
           Group is defined as a specific combination of the categorical variables.
           Returns a dictionary: keys are the groups, values are dataframes with the NHANES rows for that particular group."""
        pop = PopulationFactory.get_nhanes_population(n=None, year=year, personFilters=None, nhanesWeights=False)
        pop.advance(1)
        df = pop.get_all_person_years_as_df()
        dfForGroups = dict()
        #this approach is running the risk of missing some categories present in the df, eg by the use of range for antiHypertensiveCount
        #gender, smoking, raceEthnicity, statin, education, alcoholPerWeek, anyPhysicalActivity, antiHypertensiveCount
        #for ge, sm, ra, st, ed, al, a, an in product(NHANESGender, SmokingStatus, NHANESRaceEthnicity, [True, False], 
        #                                               Education, AlcoholCategory, [True, False], range(7)):
        for ge, sm, ra, st, ed, al, a, an in product(set(df[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                     set(df[StaticRiskFactorsType.SMOKING_STATUS.value].tolist()),
                                                     set(df[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist()),
                                                     set(df[DefaultTreatmentsType.STATIN.value].tolist()),
                                                     set(df[StaticRiskFactorsType.EDUCATION.value].tolist()),
                                                     set(df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value].tolist()),
                                                     set(df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].tolist()),
                                                     set(df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value].tolist())):
            dfForGroup = df.loc[(df[StaticRiskFactorsType.GENDER.value]==ge) & 
                                (df[StaticRiskFactorsType.SMOKING_STATUS.value]==sm) &
                                (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==ra) &
                                (df[DefaultTreatmentsType.STATIN.value]==st) &
                                (df[StaticRiskFactorsType.EDUCATION.value]==ed) &
                                (df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]==al) &
                                (df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value]==a) &
                                (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]==an), :].copy()
            if dfForGroup.shape[0]>0:
                #dfForGroups[ge.value, sm.value, ra.value, st, ed.value, al.value, a, an] = dfForGroup
                dfForGroups[ge, sm, ra, st, ed, al, a, an] = dfForGroup
        return dfForGroups

    @staticmethod
    def is_singular(cov):
       """Checks if a covariance matrix is singular or not."""
       return True if not np.all(np.linalg.eig(cov)[0]>10**(-3)) else False

    @staticmethod
    def get_distributions(dfForGroups):
        """Fits a multivariate normal to the continuous variables of each specific combination of categorical variables (group).
           Returns a dictionary: keys are 'mean', 'cov', 'min', 'max', 'singular'.
           Values for 'singular' are boolean depending on whether the covariance matrix for that key is singular or not.
           Values for all other keys are np arrays.
           The min and max are useful because we need to impose bounds on the draws, Gaussians extend to infinity...."""
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value]
        meanForGroups = dict()
        covForGroups = dict()
        minForGroups = dict()
        maxForGroups = dict()
        sizeForGroups = dict()
        singularForGroups = dict()
        namesForGroups = dict()
        for key in dfForGroups.keys():
            meanForGroups[key], covForGroups[key] = multivariate_normal.fit(np.array(dfForGroups[key][nhanesContinuousVariables]))
            singularForGroups[key] = PopulationFactory.is_singular(covForGroups[key])
            minForGroups[key] = np.min(np.array(dfForGroups[key][nhanesContinuousVariables]), axis=0)
            maxForGroups[key] = np.max(np.array(dfForGroups[key][nhanesContinuousVariables]), axis=0)
            sizeForGroups[key] = dfForGroups[key].shape[0]
            namesForGroups[key] = dfForGroups[key]["name"].tolist()
        distributions = {"mean": meanForGroups, "cov": covForGroups, "min": minForGroups, "max": maxForGroups, "singular": singularForGroups,
                         "size": sizeForGroups, "names": namesForGroups}
        distributions = PopulationFactory.get_alt_groups(distributions)
        return distributions

    @staticmethod
    def get_kaiser_distributions():
        meanForGroups = dict()
        covForGroups = dict()
        minForGroups = dict()
        maxForGroups = dict()
        singularForGroups = dict()
        sizeForGroups = dict()
        namesForGroups = dict()
        
        #kaiser population size
        popSize = 315142

        fileDir = "microsim/data/kaiser"
        csvFiles = ['/kaiserMin.csv', '/kaiserMax.csv', '/kaiserMean.csv', '/kaiserCovariance.csv', '/kaiserWeight.csv']        
        (minDf, maxDf, meanDf, covDf, weightDf) = list(map(lambda x: PopulationFactory.get_kaiserDf(x), [fileDir+y for y in csvFiles]))
        
        catVariables = PopulationFactory.kaiser_variable_types[VariableType.CATEGORICAL.value]
        conVariables = PopulationFactory.kaiser_variable_types[VariableType.CONTINUOUS.value]
        
        for index, key in minDf[catVariables].iterrows():
            key = tuple(key.tolist())
            meanForGroups[key] = meanDf.loc[
                                (meanDf["modality"]==key[0]) &
                                (meanDf["gender"]==key[1]) &
                                (meanDf["raceEthnicity"]==key[2]) &
                                (meanDf["smokingStatus"]==key[3]) &
                                (meanDf["afib"]==key[4]) &
                                (meanDf["pvd"]==key[5]) &
                                (meanDf["statin"]==key[6]) &
                                (meanDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()[0]
            covForGroups[key] = covDf.loc[
                                (covDf["modality"]==key[0]) & 
                                (covDf["gender"]==key[1]) &
                                (covDf["raceEthnicity"]==key[2]) &
                                (covDf["smokingStatus"]==key[3]) &
                                (covDf["afib"]==key[4]) &
                                (covDf["pvd"]==key[5]) &
                                (covDf["statin"]==key[6]) &
                                (covDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()
            singularForGroups[key] = PopulationFactory.is_singular(covForGroups[key])
            minForGroups[key] = minDf.loc[
                                (minDf["modality"]==key[0]) &
                                (minDf["gender"]==key[1]) &
                                (minDf["raceEthnicity"]==key[2]) &
                                (minDf["smokingStatus"]==key[3]) &
                                (minDf["afib"]==key[4]) &
                                (minDf["pvd"]==key[5]) &
                                (minDf["statin"]==key[6]) &
                                (minDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()[0]
            maxForGroups[key] = maxDf.loc[
                                (maxDf["modality"]==key[0]) &
                                (maxDf["gender"]==key[1]) &
                                (maxDf["raceEthnicity"]==key[2]) &
                                (maxDf["smokingStatus"]==key[3]) &
                                (maxDf["afib"]==key[4]) &
                                (maxDf["pvd"]==key[5]) &
                                (maxDf["statin"]==key[6]) &
                                (maxDf["anyPhysicalActivity"]==key[7]), conVariables].to_numpy()[0]
            sizeForGroups[key] = int(
                                 popSize * 
                                 weightDf.loc[
                                (weightDf["modality"]==key[0]) &
                                (weightDf["gender"]==key[1]) &
                                (weightDf["raceEthnicity"]==key[2]) &
                                (weightDf["smokingStatus"]==key[3]) &
                                (weightDf["afib"]==key[4]) &
                                (weightDf["pvd"]==key[5]) &
                                (weightDf["statin"]==key[6]) &
                                (weightDf["anyPhysicalActivity"]==key[7]), "weight"].to_numpy()[0])
            namesForGroups[key] = [f"{index}kaiserPerson{i}" for i in range(sizeForGroups[key])]
        distributions = {"mean": meanForGroups, "cov": covForGroups, "min": minForGroups, "max": maxForGroups, 
                         "singular": singularForGroups, "size": sizeForGroups, "names": namesForGroups}
        distributions = PopulationFactory.get_alt_groups(distributions)
        return distributions

    @staticmethod
    def get_alt_groups(distributions):
        """For every singular covariance matrix in the distributions dict, finds an alternative distribution, a similar one,
        with a non-singular covariance matrix.
        The term 'similar' can be defined in many different ways..."""
        altForSingular = dict()
        for key in distributions["singular"].keys():
            if distributions["singular"][key]:
                altKeys = list()
                altProbs = list()
                meanOfSingular = distributions["mean"][key]
                for altKey in distributions["singular"].keys():
                    if not distributions["singular"][altKey]:
                       altProbability = multivariate_normal(distributions["mean"][altKey],
                                                            distributions["cov"][altKey], allow_singular=False).pdf(meanOfSingular)
                       altKeys += [altKey]
                       altProbs += [altProbability]
                #using the max probability means we are using both the mean and the sd of the alternative distribution
                altForSingular[key] = altKeys[altProbs.index(max(altProbs))]
        distributions["alt"] = altForSingular
        return distributions

    @staticmethod
    def draw_from_distributions(distributions):
        """Draws from the multivariate normal distributions for each combination of categorical variables (group).
        If a draw includes a continuous variable value outside the bounds, it re-draws.
        For each group, the number of draws from the distribution is equal to the number of people in that group in 
        the original NHANES dataframe (as contained in dfForGroups).""" 
        drawsForGroups = dict()
        namesForGroups = dict()
        #just use the "mean" for the keys
        for key in distributions["mean"].keys():
            size = distributions["size"][key]
            namesForGroups[key] = distributions["names"][key]
            #use either the original distribution or the alternative if the cov matrix is singular
            distKey = key if not distributions["singular"][key] else distributions["alt"][key]
            distMean = distributions["mean"][distKey]
            distCov = distributions["cov"][distKey]
            dist = multivariate_normal(distMean, distCov, allow_singular=False)
            #this determines which bounds we use if the cov matrix is singular...the original ones or the ones from the alternative distribution
            if distributions["singular"][key] & (size>4):
                distMin = distributions["min"][key]
                distMax = distributions["max"][key]
            else:
                distMin = distributions["min"][distKey]
                distMax = distributions["max"][distKey]

            nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value]
            drawsNeeded = size
            draws = None
            #the logic about when to reshape can be improved probably...
            while drawsNeeded>0:
                if draws is None:
                    draws = dist.rvs(size=drawsNeeded)
                else:
                    if len(draws.shape)==1:
                        draws = draws.reshape((1, len(nhanesContinuousVariables)))
                    if (drawsNeeded==1):
                        draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded).reshape((1,distMean.shape[0]))), axis=0 )
                    else:
                        draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded)), axis=0 )
                if size==1:
                    draws = draws.reshape((1, distMean.shape[0]))
 
                #find which draws contain one or more continuous variables that is outside of the bounds
                rowsOutOfBounds = np.array([False]*size)
                for i, bound in enumerate(distMin):
                    rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]<0.9*bound)
                for i, bound in enumerate(distMax):
                    rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]>1.1*bound)
                #how many more draws we need in the next iteration
                drawsNeeded = size - np.sum(~rowsOutOfBounds)
                #keep the draws that have all continuous variables within the bounds
                draws = draws[~rowsOutOfBounds,:] 
            drawsForGroups[key] = draws
        return drawsForGroups, namesForGroups

    @staticmethod
    def get_df_from_draws(drawsForGroups, namesForGroups, popType=PopulationType.NHANES.value):
        """Converts the draws from the distributions to a Pandas df."""
        catVariables = PopulationFactory.variable_types(VariableType.CATEGORICAL.value, popType=popType)
        conVariables = PopulationFactory.variable_types(VariableType.CONTINUOUS.value, popType=popType)
        df = pd.DataFrame(data=None, columns= ["name"]+catVariables+conVariables)
        for key in drawsForGroups.keys():
            #names = dfForGroups[key]["name"].tolist()
            names = namesForGroups[key]
            size = drawsForGroups[key].shape[0]
            dfCont = pd.DataFrame(drawsForGroups[key])
            dfCont.columns = conVariables
            dfCat = pd.concat([pd.DataFrame(key).T]*size, ignore_index=True)
            dfCat.columns = catVariables
            dfForGroup = pd.concat( [pd.Series(names), dfCat, dfCont], axis=1).rename(columns={0:"name"})
            df = pd.concat([df,dfForGroup]) if not df.empty else dfForGroup
        df[DynamicRiskFactorsType.AGE.value] = round(df[DynamicRiskFactorsType.AGE.value]).astype('int')
        return df

    @staticmethod
    def get_nhanes_age_standardized_population(n, year):
        #nhanesDf is needed just for the index
        #nhanesDf = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        nhanesDf = PopulationFactory.get_nhanesDf() 
        standardizedPop = StandardizedPopulation(year=year)
        weights = standardizedPop.populationWeightedStandard
        #it is ok weights are merged with the entire nhanesDf, because pandas sampling takes into account the index of the series
        weights = pd.merge(nhanesDf, weights, how="left", on=["age", "gender"]).popWeight
        pop = PopulationFactory.get_nhanes_population(n=n, year=year, personFilters=None, nhanesWeights=False, distributions=False, customWeights=weights)
        return pop

    @staticmethod
    def get_cloned_people(person, n):
        return pd.Series([person.__deepcopy__() for i in range(n)])

    @staticmethod
    def apply_person_filters_on_df(personFilters, df):
        if personFilters is not None:
            for personFilterFunction in personFilters.filters["df"].values():
                df = df.loc[df.apply(personFilterFunction, axis=1)] 
        return df

    @staticmethod
    def apply_person_filters_on_people(personFilters, people):
        if personFilters is not None:
            for filterFunction in personFilters.filters["person"].values():
                people = pd.Series(list(filter(filterFunction, people)), dtype=object)
        return people

    @staticmethod
    def bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.NHANES.value):
        nRemaining = n - people.shape[0]
        while nRemaining>0:
            dfForPeople = df.sample(nRemaining, replace=True)
            peopleRemaining = pd.DataFrame.apply(dfForPeople, PersonFactory.get_person, popType=popType, axis="columns")
            peopleRemaining = PopulationFactory.apply_person_filters_on_people(personFilters, peopleRemaining)
            people = pd.concat([people, peopleRemaining])
            nRemaining = n - people.shape[0]
        return people

    @staticmethod
    def get_kaiser_people(n=1000, personFilters=None):
        distributions = PopulationFactory.get_kaiser_distributions()
        drawsForGroups, namesForGroups = PopulationFactory.draw_from_distributions(distributions)
        df = PopulationFactory.get_df_from_draws(drawsForGroups, namesForGroups, popType=PopulationType.KAISER.value)
        df = PopulationFactory.apply_person_filters_on_df(personFilters, df)
        dfForPeople = df.sample(n, weights=None, replace=True)
        people = pd.DataFrame.apply(dfForPeople, PersonFactory.get_kaiser_person, axis="columns")
        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
        people = PopulationFactory.bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.KAISER.value)   

        PopulationFactory.set_index_in_people(people)
        return people











            
















                       












