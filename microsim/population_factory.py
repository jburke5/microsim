import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
from enum import Enum

from microsim.person_factory import PersonFactory, microsimToNhanes
from microsim.population import Population
from microsim.afib_model import AFibPrevalenceModel
from microsim.pvd_model import PVDPrevalenceModel
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.population_model_repository import PopulationModelRepository
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository, 
                                                   CohortStaticRiskFactorModelRepository,
                                                   CohortDefaultTreatmentModelRepository)
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.treatment import DefaultTreatmentsType
from microsim.alcohol_category import AlcoholCategory

#these are used below to define groups ( = specific combinations of all NHANES categorical variables)
# and to define which columns from the NHANES dataframe to model as Gaussians ( = all continuous variables present
# in the original NHANES dataset).
# The last point is important, the Gaussians model the continuous variables present in the original NHANES dataset
# not all continuous variables present in the Microsim simulation (which includes more continuous variables not
# present in the original NHANES dataset such as PVD)
# The order of these two lists is important,as they define the column names of the final dataframe. The numpy arrays used in between do 
# not keep track of which column is which attribute.
nhanesCategoricalVariables = [StaticRiskFactorsType.GENDER.value, 
                              StaticRiskFactorsType.SMOKING_STATUS.value, 
                              StaticRiskFactorsType.RACE_ETHNICITY.value, 
                              DefaultTreatmentsType.STATIN.value,
                              StaticRiskFactorsType.EDUCATION.value,
                              DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
                              DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                              DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]
nhanesContinuousVariables = [DynamicRiskFactorsType.AGE.value, 
                             DynamicRiskFactorsType.HDL.value, 
                             DynamicRiskFactorsType.BMI.value, 
                             DynamicRiskFactorsType.TOT_CHOL.value, 
                             DynamicRiskFactorsType.TRIG.value, 
                             DynamicRiskFactorsType.A1C.value, 
                             DynamicRiskFactorsType.LDL.value, 
                             DynamicRiskFactorsType.WAIST.value, 
                             DynamicRiskFactorsType.CREATININE.value, 
                             DynamicRiskFactorsType.SBP.value, 
                             DynamicRiskFactorsType.DBP.value]

class PopulationType(Enum):
    NHANES = "nhanes"
    KAISER = "kaiser"

class PopulationFactory:

    @staticmethod
    def get_population(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population(**kwargs)
        elif popType == PopulationType.KAISER:
            raise RuntimeError("Kaiser popType not implemented yet in PopulationFactory.get_population function.")
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population function.")

    @staticmethod
    def get_people(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_people(**kwargs)
        elif popType == PopulationType.KAISER:
            raise RuntimeError("Kaiser popType not implemented yet in PopulationFactory.get_people function.")
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_people function.")

    @staticmethod
    def get_population_model_repo(popType):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population_model_repo()
        elif popType == PopulationType.KAISER:
            raise RuntimeError("Kaiser popType not implemented yet in PopulationFactory.get_population_model_repo function.")
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population_model_repo function.")


    @staticmethod
    def get_nhanes_person_initialization_model_repo():
        """Returns the repository needed in order to initialize a Person object using NHANES data.
           This is due to the fact that some risk factors that are needed in Microsim simulations
           are not included in the NHANES data but we have models for these risk factors. """
        return {DynamicRiskFactorsType.AFIB: AFibPrevalenceModel(),
                DynamicRiskFactorsType.PVD: PVDPrevalenceModel()}

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
        for key, value in microsimToNhanes.items():
            if key!=value:
                nhanesDf = nhanesDf.rename(columns={value:key})
        #convert the integers to booleans because in the simulation we always use bool for this rf
        nhanesDf[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value] = nhanesDf[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)
        #convert drinks per week to category
        nhanesDf[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value] = nhanesDf.apply(lambda x:
                                                                                 AlcoholCategory.get_category_for_consumption(x[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]), axis=1)
        return nhanesDf

    @staticmethod
    def get_nhanes_people(n=None, year=None, dfFilter=None, nhanesWeights=False, distributions=False):
        '''Returns a Pandas Series object with Person-Objects of all persons included in NHANES for year 
           with or without sampling. Filters are applied prior to sampling in order to maximize efficiency and minimize
           memory utilization. This does not affect the distribution of the relative percentages of groups 
           represented in people.
           The flag distributions controls if the Person-objects will come directly from the NHANES data or
           if Gaussian distributions will first be fit to the NHANES data and then draws are obtained from the distributions.'''
        nhanesDf = PopulationFactory.get_nhanesDf()        

        if year is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]
        if dfFilter is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.apply(dfFilter, axis=1)] 
 
        #if we want to draw from the NHANES distributions, then we fit the NHANES data first, draw, convert the draws to 
        #a Pandas dataframe, bring in the NHANES weights (because I do not keep those when I do the fits)
        #and then have nhanesDf point to the df obtained from the draws
        if distributions:
            dfForGroups = PopulationFactory.get_partitioned_nhanes_people(year=year)
            distributions = PopulationFactory.get_distributions(dfForGroups)
            drawsForGroups = PopulationFactory.draw_from_distributions(dfForGroups, distributions)
            df = PopulationFactory.get_df_from_draws(drawsForGroups, dfForGroups)
            df = df.merge(nhanesDf[["name","WTINT2YR"]], on="name", how="inner").copy()
            nhanesDf = df

        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        if nhanesWeights:
            if (year is None) | (n is None):
                raise RuntimeError("""Cannot set nhanesWeights True without specifying a year and n.
                                    NHANES weights are defined for each year independently and for sampling 
                                    to occur the sampling size is needed.""")
            else:
                weights = nhanesDf.WTINT2YR
                nhanesDf = nhanesDf.sample(n, weights=weights, replace=True)
           
        people = pd.DataFrame.apply(nhanesDf,
                                    PersonFactory.get_nhanes_person, initializationModelRepository=initializationModelRepository, axis="columns")
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
    def get_nhanes_population(n=None, year=None, dfFilter=None, nhanesWeights=False, distributions=False):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.
           Person attributes can originate either from the NHANES dataset directly or from distributions fit to the NHANES dataset.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, dfFilter=dfFilter, nhanesWeights=nhanesWeights, distributions=distributions)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)

    @staticmethod
    def get_partitioned_nhanes_people(year=None):
        """Partitions a NHANES df in all possible combinations of categorical variables that actually exist in NHANES.
           Group is defined as a specific combination of the categorical variables.
           Returns a dictionary: keys are the groups, values are dataframes with the NHANES rows for that particular group."""
        pop = PopulationFactory.get_nhanes_population(n=None, year=year, dfFilter=None, nhanesWeights=False)
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
        meanForGroups = dict()
        covForGroups = dict()
        minForGroups = dict()
        maxForGroups = dict()
        singularForGroups = dict()
        for key in dfForGroups.keys():
            meanForGroups[key], covForGroups[key] = multivariate_normal.fit(np.array(dfForGroups[key][nhanesContinuousVariables]))
            singularForGroups[key] = PopulationFactory.is_singular(covForGroups[key])
            minForGroups[key] = np.min(np.array(dfForGroups[key][nhanesContinuousVariables]), axis=0)
            maxForGroups[key] = np.max(np.array(dfForGroups[key][nhanesContinuousVariables]), axis=0)
        distributions = {"mean": meanForGroups, "cov": covForGroups, "min": minForGroups, "max": maxForGroups, "singular": singularForGroups}
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
    def draw_from_distributions(dfForGroups, distributions):
        """Draws from the multivariate normal distributions for each combination of categorical variables (group).
        If a draw includes a continuous variable value outside the bounds, it re-draws.
        For each group, the number of draws from the distribution is equal to the number of people in that group in 
        the original NHANES dataframe (as contained in dfForGroups).""" 
        drawsForGroups = dict()
        for key in dfForGroups.keys():
            size = len(dfForGroups[key]["name"].tolist())
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
        return drawsForGroups

    @staticmethod
    def get_df_from_draws(drawsForGroups, dfForGroups):
        """Converts the draws from the distributions to a Pandas df."""
        df = pd.DataFrame(data=None, columns= ["name"]+nhanesCategoricalVariables+nhanesContinuousVariables)
        for key in drawsForGroups.keys():
            names = dfForGroups[key]["name"].tolist()
            size = drawsForGroups[key].shape[0]
            dfCont = pd.DataFrame(drawsForGroups[key])
            dfCont.columns = nhanesContinuousVariables
            dfCat = pd.concat([pd.DataFrame(key).T]*size, ignore_index=True)
            dfCat.columns = nhanesCategoricalVariables
            dfForGroup = pd.concat( [pd.Series(names), dfCat, dfCont], axis=1).rename(columns={0:"name"})
            df = pd.concat([df,dfForGroup])
        df[DynamicRiskFactorsType.AGE.value] = round(df[DynamicRiskFactorsType.AGE.value]).astype('int')
        return df













            
















                       












