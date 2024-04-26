import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal

from microsim.person_factory import PersonFactory
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

class PopulationFactory:

    @staticmethod
    def get_nhanes_person_initialization_model_repo():
        """Returns the repository needed in order to initialize a Person object using NHANES data.
           This is due to the fact that some risk factors that are needed in Microsim simulations
           are not included in the NHANES data but we have models for these risk factors. """
        return {DynamicRiskFactorsType.AFIB: AFibPrevalenceModel(),
                DynamicRiskFactorsType.PVD: PVDPrevalenceModel()}

    @staticmethod
    def set_index_in_people(people):
        """Once people are created, its Person-objects do not have a unique index.
           This function assigns a unique index to every Person-object in people."""
        list(map(lambda person, i: setattr(person, "_index", i), people, range(people.shape[0])))

    @staticmethod
    def get_nhanes_people(n=None, year=None, dfFilter=None, nhanesWeights=False):
        '''Returns a Pandas Series object with Person-Objects of all persons included in NHANES for year 
           with or without sampling. Filters are applied prior to sampling in order to maximize efficiency and minimize
           memory utilization. This does not affect the distribution of the relative percentages of groups 
           represented in people.'''
        nhanesDf = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        
        if year is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]

        if dfFilter is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.apply(dfFilter, axis=1)] 

        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        #convert the integers to booleans because in the simulation we always use bool for this rf
        nhanesDf[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value] = nhanesDf[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)
        #convert drinks per week to category
        nhanesDf[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value] = nhanesDf.apply(lambda x:
                                                                                 AlcoholCategory.get_category_for_consumption(x[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]), axis=1)

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
    def get_nhanes_population(n=None, year=None, dfFilter=None, nhanesWeights=False):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, dfFilter=dfFilter, nhanesWeights=nhanesWeights)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)

    @staticmethod
    def get_partitioned_nhanes_people(year=None):
        """Partitions a NHANES df in all possible combinations of categorical variables that actually exist in NHANES."""
        pop = PopulationFactory.get_nhanes_population(n=None, year=year, dfFilter=None, nhanesWeights=False)
        pop.advance(1)
        df = pop.get_all_person_years_as_df()
        dfForGroups = dict()
        #namesForGroups = dict()
        #this approach is running the risk of missing some categories present in the df, eg by the use of range for antiHypertensiveCount
        #gender, smoking, raceEthnicity, statin, education, alcoholPerWeek, anyPhysicalActivity, antiHypertensiveCount
        for ge, sm, ra, st, ed, al, a, an in product(NHANESGender, SmokingStatus, NHANESRaceEthnicity, [True, False], 
                                                       Education, AlcoholCategory, [True, False], range(7)):
            dfForGroup = df.loc[(df[StaticRiskFactorsType.GENDER.value]==ge) & 
                                (df[StaticRiskFactorsType.SMOKING_STATUS.value]==sm) &
                                (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==ra) &
                                (df[DefaultTreatmentsType.STATIN.value]==st) &
                                (df[StaticRiskFactorsType.EDUCATION.value]==ed) &
                                (df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]==al) &
                                (df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value]==a) &
                                (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]==an), :].copy()
            if dfForGroup.shape[0]>0:
                dfForGroups[ge.value, sm.value, ra.value, st, ed.value, al.value, a, an] = dfForGroup
        return dfForGroups

    @staticmethod
    def is_singular(cov):
       return True if not np.all(np.linalg.eig(cov)[0]>10**(-3)) else False

    @staticmethod
    def get_distributions(dfForGroups):
        meanForGroups = dict()
        covForGroups = dict()
        minForGroups = dict()
        maxForGroups = dict()
        singularForGroups = dict()
        continuousVars = ['age', 'hdl', 'bmi', 'totChol', 'trig', 'a1c', 'ldl', 'waist', 'creatinine', 'sbp', 'dbp']
        for key in dfForGroups.keys():
            meanForGroups[key], covForGroups[key] = multivariate_normal.fit(np.array(dfForGroups[key][continuousVars]))
            singularForGroups[key] = PopulationFactory.is_singular(covForGroups[key])
            minForGroups[key] = np.min(np.array(dfForGroups[key][continuousVars]))
            maxForGroups[key] = np.max(np.array(dfForGroups[key][continuousVars]))
        return {"mean": meanForGroups, "cov": covForGroups, "min": minForGroups, "max": maxForGroups, "singular": singularForGroups}

    @staticmethod
    def get_alt_groups(distributions):
        altForSingular = dict()
        altKeys = list()
        altProbs = list()
        for key in distributions["singular"].keys():
            if distributions["singular"][key]:
                meanOfSingular = distributions["mean"][key]
                for altKey in distributions["singular"].keys():
                    if not distributions["singular"][altKey]:
                       altProbability = multivariate_normal(distributions["mean"][altKey],
                                                            distributions["cov"][altKey], allow_singular=False).pdf(meanOfSingular)
                       altKeys += [altKey]
                       altProbs += [altProbability]
                altForSingular[key] = altKey[altProbs.index(max(altProbs))]
        distributions["alt"] = altForSingular
        return distributions

    @staticmethod
    def draw_from_distributions(dfForGroups, distributions):
        drawsForGroups = dict()
        for key in dfForGroups:
            size = len(dfForGroups[key]["name"].tolist())
            distKey = key if not distributions["singular"][key] else distributions["alt"][key]
            distMean = distributions["mean"][distKey]
            distCov = distributions["cov"][distKey]
            dist = multivariate_normal(distMean, distCov, allow_singular=False)
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
                        draws = draws.reshape((1, len(continuousVars)))
                    if (drawsNeeded==1):
                        draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded).reshape((1,distMean.shape[0]))), axis=0 )
                    else:
                        draws = np.concatenate( (draws, dist.rvs(size=drawsNeeded)), axis=0 )
                if size==1:
                    draws = draws.reshape((1, distMean.shape[0]))
 
                rowsOutOfBounds = np.array([False]*size)
                for i, bound in enumerate(np.array(distMin)[0]):
                    rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]<0.9*bound)
                for i, bound in enumerate(np.array(distMax)[0]):
                    rowsOutOfBounds = rowsOutOfBounds | (draws[:,i]>1.1*bound)
                drawsNeeded = size - np.sum(~rowsOutOfBounds)
                draws = draws[~rowsOutOfBounds,:] 
            drawsForGroups[key] = draws
        return drawsForGroups

    @staticmethod
    def get_df_from_draws(drawsForGroups):
        df = pd.DataFrame(data=None, columns= ["name"]+categoricalVars+continuousVars)
        for key in drawsForGroups.keys():
            size = drawsForGroups[key].shape[0]
            dfCont = pd.DataFrame(drawsForGroups[key])
            dfCont.columns = continuousVars
            dfCat = pd.concat([pd.DataFrame(key).T]*size, ignore_index=True)
            dfCat.columns = categoricalVars
            dfForGroup = pd.concat( [pd.Series(names), dfCat, dfCont], axis=1).rename(columns={0:"name"})
            df = pd.concat([df,dfForGroup])
        df[DynamicRiskFactorsType.AGE.value] = round(df[DynamicRiskFactorsType.AGE.value]).astype('int')
        return df













            
















                       












