import copy
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
from scipy.optimize import brentq
import math

from microsim.person.person_factory import PersonFactory
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.population.population import Population
from microsim.common.age_scope import AgeScope
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.population.population_model_repository import PopulationModelRepository, PopulationRepositoryType
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.risk_factors. cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository,
                                                                 CohortStaticRiskFactorModelRepository)
from microsim.default_treatments.default_treatment_model_repository import DefaultTreatmentModelRepository
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.population.standardized_population import StandardizedPopulation
from microsim.common.variable_type import VariableType
from microsim.outcomes.outcome import OutcomeType
from microsim.common.population_type import PopulationType
from microsim.common.data_loader import get_absolute_datafile_path
from microsim.risk_factors.modality import Modality

class PopulationFactory:
    #the NHANES df as get_nhanesDf builds it, cached because every population build needs it
    _nhanesDf = None

    #the distributions the continuous variables are drawn from when distributions=True, cached because
    #neither the partition nor the fit depends on any argument: they are the same for the life of the
    #process and cost about 5 seconds to build
    _crudeDistributions = None

    #the mean of every continuous variable for every group of NHANES people, the mean a draw is shifted
    #to. Cached for the same reason: it takes no argument and every population build needs it.
    _groupMeans = None

    #a survey-weight bootstrap of the NHANES df, what the crude fits and the group means are computed on
    #so that they reflect the WTINT2YR weights the person-row sampling uses. Fixed seed: the cached
    #distributions and means must be the same in every process.
    _nhanesDfResampled = None
    RESAMPLE_SEED = 0

    #per-(year, gender, ageGroup) mean minus the same cell's pooled mean, for every drawn continuous
    #variable: the Gaussians are fit on all years pooled, this moves each row to its year's level
    _yearCorrections = None

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
                                                  DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value,
                                                  DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value],
                             VariableType.CONTINUOUS.value:   [DynamicRiskFactorsType.AGE.value,
                                                  DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value,
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
    #all antiHypertensiveCount values NHANES holds; the state pipeline enumerates these, both for the
    #cross product in get_stateDf and for the proportions in get_proportionForDefaultTreatments, so the
    #two cannot drift apart and no observed count is left out of the denominators
    antiHypertensiveCounts = [0., 1., 2., 3., 4., 5., 6., 7.]

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
        elif popType == PopulationType.STATE:
            return PopulationFactory.get_state_population(**kwargs)
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_population function.")

    @staticmethod
    def get_people(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_people(**kwargs)
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_people(**kwargs)
        elif popType == PopulationType.STATE:
            return PopulationFactory.get_state_people(**kwargs)
        else:
            raise RuntimeError("Unknown popType in PopulationFactory.get_people function.")

    @staticmethod
    def get_population_model_repo(popType, **kwargs):
        if popType == PopulationType.NHANES:
            return PopulationFactory.get_nhanes_population_model_repo(**kwargs)
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_population_model_repo(**kwargs)
        elif popType == PopulationType.STATE:
            return PopulationFactory.get_nhanes_population_model_repo(**kwargs)
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
           Returns a Pandas df with the NHANES information as exists in Microsim.
           The df is built once and kept in _nhanesDf: reading the file and converting the columns
           costs about 14 seconds, and every population build needs the df. Callers get a copy of
           the cached df, some of them add columns to what they get back (see get_treatment_weights
           and get_partitioned_nhanes_people_crude), which would corrupt the cache for everyone else."""
        if PopulationFactory._nhanesDf is None:
            nhanesDf = pd.read_stata(get_absolute_datafile_path("fullyImputedDataset.dta"))
            #in Person-objects, the attribute name is used
            #the column holding the row number is called index in the data file, renaming a level_0
            #column (which pandas produces only when reset_index runs on an index already named index)
            #silently did nothing, so no name column existed and the distributions path could not merge on it
            nhanesDf = nhanesDf.rename(columns={"index":"name"})
            #rename the columns that have different column names than the ones that appear in Microsim
            nhanesDf = PopulationFactory.rename_df_columns(nhanesDf, PersonFactory.microsimToNhanes)
            #convert the integers to booleans because in the simulation we always use bool for these
            for col in [DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value, DefaultTreatmentsType.STATIN.value]:
                nhanesDf[col] = nhanesDf[col].astype(bool)
            #convert these columns to int type
            for col in [StaticRiskFactorsType.RACE_ETHNICITY.value,
                        StaticRiskFactorsType.EDUCATION.value,
                        StaticRiskFactorsType.GENDER.value,
                        StaticRiskFactorsType.SMOKING_STATUS.value]:
                nhanesDf[col] = nhanesDf[col].astype(int)
            PopulationFactory._nhanesDf = nhanesDf
        return PopulationFactory._nhanesDf.copy()

    @staticmethod
    def get_nhanesDf_resampled():
        """A survey-weight bootstrap of the NHANES df: rows drawn with replacement, probability
           proportional to WTINT2YR, same total size. The crude Gaussian fits and the group means are
           computed on this df so they reflect the same weights the person-row sampling uses.
           The seed is fixed because the fits and means built from this df are cached and must be
           identical in every process. Trade-offs of a bootstrap: duplicated rows add Monte Carlo
           noise, and an extreme low-weight row can drop out and tighten a group's observed min/max,
           which draw_within_bounds mitigates by widening the bounds to 0.9*min-1.1*max."""
        if PopulationFactory._nhanesDfResampled is None:
            df = PopulationFactory.get_nhanesDf()
            PopulationFactory._nhanesDfResampled = df.sample(n=df.shape[0], replace=True,
                                                             weights=df.WTINT2YR,
                                                             random_state=PopulationFactory.RESAMPLE_SEED)
        return PopulationFactory._nhanesDfResampled.copy()

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
        #map, not replace: replacing strings with integers makes pandas downcast the object column
        #silently, which is deprecated and warns. map does the same substitution without the downcast.
        #The astype below still catches a value missing from the dict, as map turns it into NaN.
        df[StaticRiskFactorsType.GENDER.value] = df[StaticRiskFactorsType.GENDER.value].map({'F': 2, 'M': 1}).astype('int')
        df[StaticRiskFactorsType.RACE_ETHNICITY.value] = df[StaticRiskFactorsType.RACE_ETHNICITY.value].map(
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
    def check_nhanes_people_arguments(n=None, year=None, nhanesWeights=False, distributions=False, customWeights=None, maxDraws=None):
        '''Raises RuntimeError for any argument of get_nhanes_people that cannot be honored, see its
           docstring for what each one means. Called before any of the work, so a call that cannot be
           served fails in 0.000s rather than after the distributions have been fit and every row redrawn.
           The order of the checks is what decides which message a call gets, see the two comments below.'''

        if (year is not None) & (year not in [2011, 2015, 2007, 2003, 2009, 2001, 2005, 1999, 2013, 2017]):
            raise RuntimeError(f"NHANES data for year {year} is not available")

        #n is a count of Person-objects: a float is not rounded here, since which way to round is the
        #caller's decision, and n=0 is refused because an empty population advances without error and
        #reports a prevalence of 0 for every outcome, so it reads as a result rather than as a mistake.
        #bool is excluded explicitly, n=True is far more likely a flag typed into the wrong argument.
        #These two use 'and', not the '&' below, because '&' evaluates both sides and both right-hand
        #sides raise on n=None
        if (n is not None) and (isinstance(n, bool) or not isinstance(n, (int, np.integer))):
            raise RuntimeError(f"""Cannot build a population of n={n!r} people: n is a count of
                                   Person-objects and must be an integer, not a {type(n).__name__}.""")

        if (n is not None) and (n < 1):
            raise RuntimeError(f"""Cannot build a population of n={n} people: n is the number of
                                   Person-objects to return and must be at least 1. Pass n=None to get
                                   every NHANES person of that year instead.""")

        #same rationale as the nhanesWeights check below: the check right after this one combines
        #distributions with '&', where a non-bool either raises an opaque TypeError or slips through
        if not isinstance(distributions, (bool, np.bool_)):
            raise RuntimeError(f"""distributions must be True or False, not {distributions!r}
                                   ({type(distributions).__name__}). It says whether the continuous
                                   variables are redrawn from the Gaussians fit to NHANES.""")

        #this one comes before the nhanesWeights checks, which would otherwise report this combination
        #as the both-weight-kinds one and give the less useful of the two messages
        if distributions & (customWeights is not None):
            raise RuntimeError("""Cannot use customWeights with distributions=True. With distributions the
                                  categorical variables and the age of each person still come from the
                                  NHANES rows and only the continuous variables are drawn, so it is the
                                  sampling of those rows that decides who the population is made of, and
                                  nhanesWeights is what makes that sampling representative.""")

        #the type is checked because the checks below combine nhanesWeights with '&': a non-bool would
        #either raise a TypeError out of the operator or, worse, pass through it silently. numpy bools
        #are accepted for the same reason numpy integers are, they arrive from dataframe comparisons
        if not isinstance(nhanesWeights, (bool, np.bool_)):
            raise RuntimeError(f"""nhanesWeights must be True or False, not {nhanesWeights!r}
                                   ({type(nhanesWeights).__name__}). It says whether the NHANES rows are
                                   sampled with the survey weights; it is not inferred from any other
                                   argument, and None is not a way of leaving it open.""")

        if nhanesWeights & (customWeights is not None):
            raise RuntimeError("Cannot use both nhanesWeights (nhanesWeights=True) and custom weights (customWeights is not None).")

        if nhanesWeights & (n is None):
            raise RuntimeError("""Cannot set nhanesWeights True without specifying n.
                                  NHANES weights are defined for each year independently and for sampling
                                  to occur the sampling size is needed.""")

        #WTINT2YR weighs a person only against the others of their own two-year cycle, so sampling every
        #year at once with these weights builds a population nobody can interpret
        if nhanesWeights & (year is None):
            raise RuntimeError("""Cannot use nhanesWeights True with year=None. The NHANES weights are
                                  defined for each survey year independently, so they cannot weigh people
                                  of different years against each other. Pass a year, or nhanesWeights
                                  False, or supply customWeights built for a pooled draw.""")

        if (customWeights is not None) & (n is None):
            raise RuntimeError("Cannot use customWeights without specifying n, for sampling to occur the sampling size is needed.")

        if (maxDraws is not None) & (n is None):
            raise RuntimeError("""Cannot use maxDraws without specifying n. maxDraws budgets the sampling
                                  of NHANES rows and without an n no sampling takes place at all.""")

    @staticmethod
    def get_nhanes_people(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None, outcomePrevalenceModelRepository=None, maxDraws=None):
        '''Returns a Pandas Series of Person-objects built from NHANES, with or without sampling.
           year: NHANES survey year, 1999 to 2017 in steps of 2. year=None uses every year at once, and is
              refused with nhanesWeights=True since those weights cannot weigh a 1999 person against a 2017
              one; a weighted pooled draw needs customWeights.
           n: number of Person-objects to return, honored in every sampling mode. n=None means no sampling:
              every NHANES person of that year that passes the filters is returned. nhanesWeights and
              customWeights both require an n.
           nhanesWeights: sample the rows with the survey weights (WTINT2YR), which is what makes a sample
              representative of the US population of that year. Must be True or False, defaults to False and
              is inferred from nothing, so the call itself says whether a population is weighted. Requires
              an n; refused with customWeights and with year=None.
           distributions: keep the categorical variables and the age of each row and redraw only the
              continuous ones from Gaussians fit to NHANES, so which rows are sampled still decides the
              make-up of the population. The redraw happens per sampled row, so a row drawn twice yields
              two people with different continuous variables. The df-level filters move after the sampling
              with it, since they have to hold for the drawn values. Refused with customWeights.
           maxDraws: budget on the NHANES rows sampled to reach n people, default max(100*n, 500) in
              bring_people_to_target_n. Filters reject a row only after it is drawn, so a selective set
              needs many more than n draws (0.3% acceptance is ~330 draws per person against a budget of
              100) and raises asking for this argument. Requires an n.
           Sampling is always with replacement and returns exactly n people: filters run after a row is
           drawn and whatever they drop is drawn again with the same weights (see bring_people_to_target_n).
           Without distributions the df-level filters run before the sampling, for speed and memory, which
           does not affect the relative make-up of the people returned.
           Which combinations of the arguments are refused, and why, is in check_nhanes_people_arguments.'''

        PopulationFactory.check_nhanes_people_arguments(n=n, year=year, nhanesWeights=nhanesWeights, distributions=distributions, customWeights=customWeights, maxDraws=maxDraws)

        nhanesDf = PopulationFactory.get_nhanesDf()

        if year is not None: #if year is None, then use the entire dataframe
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]

        if personFilters is None: #since we started including children in the NHANES df, by default use an adult filter on the df
            personFilters = PersonFilterFactory.get_person_filter(["adult"])
        else:
            print("Warning: NHANES populations now include children. If you need an adult population you need to add an age filter.")

        if distributions:
            #with distributions the continuous variables of a row are replaced by a draw from a Gaussian fit on gender, race ethnicity, education and age, shifted
            #to the mean of the group the row belongs to (see group_key_frame). Grouping on those four only, over all years and with overlapping age windows, 
            #is what leaves enough people per group to fit a covariance matrix that is not singular
            crudeDistributions = PopulationFactory.get_crude_distributions()
            if n is None: #no sampling: every row becomes a person, so every row is redrawn exactly once
                nhanesDf = PopulationFactory.redraw_continuous_variables(nhanesDf, crudeDistributions)
                nhanesDf = PopulationFactory.apply_person_filters_on_df(personFilters, nhanesDf)
        else:
            crudeDistributions = None
            nhanesDf = PopulationFactory.apply_person_filters_on_df(personFilters, nhanesDf) #without distributions the df-level filters run here, before the sampling

        if nhanesDf.shape[0] == 0: #if filters are too restrictive stop here (distributions=True + n not None bypass the filters and that is ok)
            raise RuntimeError("""The df-level filters of personFilters rejected every row, so there is nobody left to build Person-objects from.""")

        #the weights are picked here because the NHANES weights have to come from the df the sampling is done on, ie after the filters have run
        if nhanesWeights:
            weights = nhanesDf.WTINT2YR
        elif customWeights is not None:
            weights = customWeights
        else:
            weights = None

        imr = InitializationModelRepository()
        if n is None: #no sampling: every row that passed the df-level filters above becomes a person
            people = pd.DataFrame.apply(nhanesDf, PersonFactory.get_person, popType=PopulationType.NHANES.value, initializationModelRepository=imr, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, axis="columns")
            people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
            if people.shape[0] == 0: #if person-filters are too restrictive stop here
                raise RuntimeError(f"""The person-level filters of personFilters ({sorted(personFilters.filters["person"].keys())}) rejected all
                                       {nhanesDf.shape[0]} of the Person-objects built, so the population would be empty.""")
        else: #draw, redraw and filtering all happen in the one loop, which keeps drawing until n people have passed
            people = PopulationFactory.bring_people_to_target_n(n, pd.Series([], dtype=object), nhanesDf, personFilters, popType=PopulationType.NHANES.value, initializationModelRepository=imr, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, weights=weights, maxDraws=maxDraws, distributions=crudeDistributions)

        #what identifies a person is _index, set right below, and _name is the NHANES row they came from
        people = people.reset_index(drop=True)
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_nhanes_population_model_repo(riskScaling=None):
        """Return the default, self-consistent set of models for advancing an NHANES Population."""
        return PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                         DefaultTreatmentModelRepository(),
                                         OutcomeModelRepository(riskScaling=riskScaling),
                                         CohortStaticRiskFactorModelRepository())

    @staticmethod
    def get_kaiser_population_model_repo(wmhSpecific=True, riskScaling=None):
        """Return the default, self-consistent set of models for advancing a Kaiser Population."""
        return PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                         DefaultTreatmentModelRepository(),
                                         OutcomeModelRepository(wmhSpecific=wmhSpecific, riskScaling=riskScaling),
                                         CohortStaticRiskFactorModelRepository())

    @staticmethod
    def get_nhanes_population(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None, riskScaling=None, prevalenceRiskScaling=None, maxDraws=None):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.
           Person attributes can originate either from the NHANES dataset directly or from distributions fit to the NHANES dataset.
           nhanesWeights: True or False, defaulting to False and inferred from nothing, see get_nhanes_people.
           riskScaling: optional dict[OutcomeType, float] applied to per-outcome risk inside the OutcomeModelRepository.
           prevalenceRiskScaling: optional dict[OutcomeType, float] applied to per-outcome priorToSim risk inside the OutcomePrevalenceModelRepository.
           maxDraws: budget on the NHANES rows sampled to reach n people, raise it for restrictive
              personFilters, see get_nhanes_people.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, personFilters=personFilters, nhanesWeights=nhanesWeights, distributions=distributions, customWeights=customWeights, outcomePrevalenceModelRepository=OutcomePrevalenceModelRepository(riskScaling=prevalenceRiskScaling), maxDraws=maxDraws)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo(riskScaling=riskScaling)
        return Population(people, popModelRepository)

    @staticmethod
    def get_kaiser_population(n=1000, personFilters=None, wmhSpecific=True, riskScaling=None):
        people = PopulationFactory.get_kaiser_people(n=n, personFilters=personFilters)
        popModelRepository = PopulationFactory.get_kaiser_population_model_repo(wmhSpecific=wmhSpecific, riskScaling=riskScaling)
        return Population(people, popModelRepository)

    @staticmethod
    def calibrate_prevalence(scaleOutcomeType, targetOutcomeType, target, scope,
                             popType, peopleArgs, baselineRiskScaling=None):
        '''Empirically find the OutcomePrevalenceModelRepository riskScaling on
           `scaleOutcomeType` such that the realized priorToSim prevalence of
           `targetOutcomeType` in `scope` equals `target`.

           When `scaleOutcomeType == targetOutcomeType` this calibrates a single outcome
           directly. When they differ, the scale outcome must precede the target in
           OutcomeType seeding order so its priorToSim status can cascade into the
           target's linear predictor — the canonical case is stroke/MI, whose prevalence
           models short-circuit unless CV was seeded as priorToSim.

           scaleOutcomeType: outcome whose prevalence riskScaling is the unknown. Its
                             prevalence model must honor riskScaling; MI and COGNITION
                             do not and are refused.
           targetOutcomeType: outcome whose realized priorToSim prevalence the search
                              drives to `target`. May equal scaleOutcomeType.
           target:           desired realized prevalence in (0, 1).
           scope:            AgeScope restricting which persons contribute to the
                             measured prevalence, e.g. AgeScope(65, None) for age>=65.
           popType:          PopulationType.NHANES is supported. Kaiser inlines its
                             prevalence calls in person_factory.get_kaiser_person and
                             does not honor riskScaling; refused.
           peopleArgs:       dict forwarded to get_nhanes_people(**peopleArgs). An
                             outcomePrevalenceModelRepository key, if present, is dropped
                             (the calibrator manages seeding itself).
           baselineRiskScaling: optional dict[OutcomeType, float] applied to other
                                outcomes for the duration of the calibration. Lets
                                callers chain calibrations (e.g., pin epilepsy at a
                                previously calibrated value while solving CV-for-stroke).
                                The entry for scaleOutcomeType, if present, is overridden.

           Implementation: constructs people once with no priorToSim seeding, snapshots
           each in-scope person's RNG state, and per brentq iteration restores the RNG,
           clears _outcomes, and re-runs Person.seed_prevalent_outcomes with the trial
           scaling. The inner loop is deterministic in s irrespective of nhanesWeights.

           Returns: float scaling to feed into prevalenceRiskScaling={scaleOutcomeType: s}.

           Note the search runs against a pristine baseline (useDefaults=False), while a
           population built with prevalenceRiskScaling={scaleOutcomeType: s} merges in
           DEFAULT_PREVALENCE_RISK_SCALING, whose entries for other outcomes then stack on top
           of s and move the realized prevalence off the target. Either neutralize those
           defaults when building (e.g. add STROKE: 1.0) and pass the same dict here as
           baselineRiskScaling, or pass the defaults you intend to keep as baselineRiskScaling
           so the returned s already accounts for them.'''

        if popType is not PopulationType.NHANES:
            raise NotImplementedError(
                f"calibrate_prevalence only supports PopulationType.NHANES; "
                f"got {popType}."
            )
        if scaleOutcomeType in {OutcomeType.MI, OutcomeType.COGNITION}:
            raise ValueError(
                f"{scaleOutcomeType} prevalence model ignores riskScaling; cannot scale."
            )
        defaultOpmr = OutcomePrevalenceModelRepository(useDefaults=False)
        if not defaultOpmr.has_prevalence_model(scaleOutcomeType):
            raise ValueError(
                f"{scaleOutcomeType} has no prevalence model registered; nothing to scale."
            )
        if not defaultOpmr.has_prevalence_model(targetOutcomeType):
            raise ValueError(
                f"{targetOutcomeType} has no prevalence model registered; nothing to measure."
            )

        order = list(OutcomeType)
        if order.index(targetOutcomeType) < order.index(scaleOutcomeType):
            raise ValueError(
                f"targetOutcomeType {targetOutcomeType} is seeded before scaleOutcomeType "
                f"{scaleOutcomeType} in OutcomeType order; cannot cascade."
            )

        if not (0. < target < 1.):
            raise ValueError(
                f"target must be in (0, 1) for an empirical prevalence; got {target}."
            )

        peopleArgs = {k: v for k, v in peopleArgs.items() if k != "outcomePrevalenceModelRepository"}

        unseededPeople = PopulationFactory.get_nhanes_people(
            **peopleArgs, outcomePrevalenceModelRepository=None
        )
        scopePeople = [p for p in unseededPeople if scope.contains(p._current_age)]
        if len(scopePeople) == 0:
            raise ValueError(
                f"scope {scope} matched zero constructed persons; cannot calibrate against "
                f"an empty subset (check peopleArgs personFilters and n)."
            )

        rngStates = [copy.deepcopy(p._rng.bit_generator.state) for p in scopePeople]
        baseScaling = dict(baselineRiskScaling or {})

        def empiricalGap(logS):
            scaling = {**baseScaling, scaleOutcomeType: math.exp(logS)}
            opmr = OutcomePrevalenceModelRepository(riskScaling=scaling, useDefaults=False)
            hits = 0
            for person, state in zip(scopePeople, rngStates):
                person._rng.bit_generator.state = copy.deepcopy(state)
                for ot in person._outcomes:
                    person._outcomes[ot] = []
                person.seed_prevalent_outcomes(opmr)
                if person.has_outcome_prior_to_simulation(targetOutcomeType):
                    hits += 1
            return hits / len(scopePeople) - target

        # Bracket in log-space (scaling = exp(logS)): keeps scaling > 0 and is well-conditioned
        # for log-linear prevalence models. ±15 maps to ~3e-7..3e6, wide enough to span the
        # achievable prevalence floor/ceiling so brentq's sign-change requirement is met.
        lo, hi = -15.0, 15.0
        gLo, gHi = empiricalGap(lo), empiricalGap(hi)
        # Both endpoints same-sign => target is outside the achievable range, not a bracketing
        # failure; surface a specific error instead of letting brentq raise.
        if gLo > 0 and gHi > 0:
            raise ValueError(
                f"target {target} for {targetOutcomeType} in scope {scope} is below the "
                f"minimum achievable prevalence ({gLo + target:.4f}) by scaling "
                f"{scaleOutcomeType}."
            )
        if gLo < 0 and gHi < 0:
            raise ValueError(
                f"target {target} for {targetOutcomeType} in scope {scope} is above the "
                f"maximum achievable prevalence ({gHi + target:.4f}) by scaling "
                f"{scaleOutcomeType}. This often means scaling {scaleOutcomeType} does not "
                f"cascade into {targetOutcomeType}."
            )
        if gLo == 0:
            scaling = math.exp(lo)
        elif gHi == 0:
            scaling = math.exp(hi)
        else:
            # Brent's method: bracketed root finder combining bisection (guaranteed convergence)
            # with secant / inverse-quadratic steps (superlinear when well-behaved). xtol is on
            # logS, so 1e-4 ≈ 0.01% resolution on scaling — well below epidemiological precision.
            logScaling = brentq(empiricalGap, lo, hi, xtol=1e-4)
            scaling = math.exp(logScaling)

        print(f"calibrate_prevalence: scale={scaleOutcomeType.value} "
              f"target_outcome={targetOutcomeType.value} scope={scope.label} "
              f"target={target:.4f} scaling={scaling:.4f} scopeN={len(scopePeople)}")
        return scaling

    @staticmethod
    def get_state_population(year=2030, personFilters=None, state="OH", samplingRate=0.025):
        people = PopulationFactory.get_state_people(year=year, personFilters=personFilters, state=state, samplingRate=samplingRate)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)

    @staticmethod
    def is_singular(cov):
       """Checks if a covariance matrix is singular or not."""
       #eigvalsh, not eig: cov is symmetric, and eig can return complex eigenvalues, which '>' refuses to order
       return not np.all(np.linalg.eigvalsh(cov) > 10**(-3))

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

        fileDir = get_absolute_datafile_path("kaiser")
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
    def draw_from_distributions(distributions, maxDraws=None):
        """Draws from the multivariate normal distributions for each combination of categorical variables (group).
        If a draw includes a continuous variable value outside the bounds, it re-draws.
        For each group, the number of draws from the distribution is equal to the number of people in that group in
        the original NHANES dataframe (as contained in dfForGroups).

        maxDraws: budget on the number of draws made for a single group, defaults to max(1000*size, 50000).
                  A group whose bounds its distribution never satisfies would otherwise loop forever, so
                  exhausting the budget raises a RuntimeError that reports the observed acceptance rate.
                  The budget is deliberately generous because low acceptance rates are normal here: a group
                  with a singular covariance matrix draws from an alternative group's distribution while
                  keeping its own bounds, and in NHANES 1999 that leaves at least one group accepting about
                  0.12% of its draws (its own ages run 52-93 while the distribution it draws from is centered
                  at 21). Such a group needs a few thousand draws to fill, and the draws it does keep are far
                  out in the tail of the distribution they came from."""
        drawsForGroups = dict()
        namesForGroups = dict()
        #just use the "mean" for the keys
        for key in distributions["mean"].keys():
            size = distributions["size"][key]
            if size == 0: #a group too small to hold a person contributes no rows, and the loop below would leave draws None
                continue
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

            drawsNeeded = size
            draws = None
            #the batch is always exactly the shortfall, so it never exceeds size and does not need to be
            #capped against the budget -- note that rowsOutOfBounds below relies on draws holding size rows
            groupMaxDraws = max(1000*size, 50000) if maxDraws is None else maxDraws
            drawn = 0
            #the logic about when to reshape can be improved probably...
            while drawsNeeded>0:
                if drawn>=groupMaxDraws:
                    accepted = size - drawsNeeded
                    raise RuntimeError(f"""Cannot draw {size} in-bounds draws for group {key}: kept {accepted} of
                                           {drawn} draws (acceptance rate: {accepted/drawn:.4f}). The bounds of this
                                           group are almost never satisfied by the distribution being drawn from
                                           (singular covariance: {distributions["singular"][key]}, drawing from
                                           group: {distKey}). Increase maxDraws if the group is this narrow by design.""")
                drawn += drawsNeeded
                if draws is None:
                    draws = dist.rvs(size=drawsNeeded)
                else:
                    #draws is always 2-D here: the only 1-D producer is the initial rvs of a size-1
                    #group, which the size==1 reshape below fixes within the same iteration
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
        #the df supplies the age and gender of every row, so the merge below can hand each row its weight
        nhanesDf = PopulationFactory.get_nhanesDf()
        standardizedPop = StandardizedPopulation(year=year)
        weights = standardizedPop.populationWeightedStandard
        #NHANES top-codes age (85 through 2005, 80 after), so the standard population at and above the
        #top-coded age is collapsed onto it, where the people it stands for actually sit
        maxAge = nhanesDf.loc[nhanesDf.year == year, "age"].max()
        weights = weights.assign(age=weights["age"].clip(upper=maxAge))
        weights = weights.groupby(["age", "gender"], as_index=False)["popWeight"].sum()
        #it is ok weights are merged with the entire nhanesDf, because pandas sampling takes into account the index of the series
        weights = pd.merge(nhanesDf, weights, how="left", on=["age", "gender"]).popWeight
        #sampling picks rows, so each age-gender group's standard share is split over that year's rows of
        #the group: without the division the sampled shares come out as standard share times NHANES row count
        nRowsInYear = nhanesDf.loc[nhanesDf.year == year].groupby(["age", "gender"])["age"].transform("size")
        weights = weights / nRowsInYear
        pop = PopulationFactory.get_nhanes_population(n=n, year=year, personFilters=None, nhanesWeights=False, distributions=False, customWeights=weights)
        return pop

    @staticmethod
    def get_cloned_people(person, n):
        return pd.Series([person.__deepcopy__() for i in range(n)])

    @staticmethod
    def apply_person_filters_on_df(personFilters, df):
        """Keeps the rows of df that every df-level filter accepts.

           The df can be the NHANES rows or, when drawing from the distributions, the rows drawn from
           them. The second carries only the variables in PopulationFactory.variable_types plus name and
           WTINT2YR, so a filter written against a column that exists only in NHANES works on one and not
           on the other; that is reported rather than left as a KeyError out of pandas."""
        if personFilters is not None:
            for filterName, personFilterFunction in personFilters.filters["df"].items():
                if df.shape[0] == 0: #an earlier filter left nothing, and apply on an empty df has no rows to align to
                    return df
                try:
                    df = df.loc[df.apply(personFilterFunction, axis=1)]
                except KeyError as e:
                    raise RuntimeError(f"""The df filter '{filterName}' asked for the variable {e}, which
                                           this dataframe does not have. Its columns are:
                                           {sorted(df.columns.tolist())}.""")
        return df

    @staticmethod
    def apply_person_filters_on_people(personFilters, people):
        if personFilters is not None:
            for filterFunction in personFilters.filters["person"].values():
                people = pd.Series(list(filter(filterFunction, people)), dtype=object)
        return people

    @staticmethod
    def bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.NHANES.value, initializationModelRepository=None, outcomePrevalenceModelRepository=None, weights=None, maxDraws=None, distributions=None):
        """Samples rows from df until people holds exactly n Person-objects that pass personFilters.

           Filters drop a row only after it has been drawn -- the person-level ones only after a whole
           Person-object has been built from it -- so a draw of n rows usually yields fewer than n people
           and the shortfall has to be drawn again. Called with an empty people this is the whole draw,
           called with the people of an earlier draw it is the top-up of that draw.

           weights: the sampling weights (a Pandas Series aligned on the index of df, or None for uniform
                    sampling). Every pass here uses them, which is what keeps the people added by a later
                    pass drawn from the same distribution as the people of the first -- with weights=None
                    on a weighted population the top-up would silently bias the sample.
           distributions: when given, the continuous variables of every sampled row are redrawn from these
                    before the row becomes a Person, so that a row sampled more than once yields people
                    whose continuous variables differ. The df-level filters are then applied here too,
                    because the values they have to hold for are the drawn ones and those do not exist
                    until the row has been sampled.
           maxDraws: budget on the total number of rows sampled here, defaults to max(100*n, 500). Filters
                     that accept (almost) nothing would otherwise loop forever, so exhausting the budget
                     raises a RuntimeError: with the observed acceptance rate when some rows did pass and
                     the budget is what ran out, and without it when none did, since then no budget is
                     large enough and the rate carries nothing but zero."""
        if df.shape[0]==0:
            raise RuntimeError(f"""Cannot bring people to the target n={n}: the dataframe to sample from is empty.
                                   The df-level filters of personFilters rejected every row.""")
        maxDraws = max(100*n, 500) if maxDraws is None else maxDraws
        drawn = 0
        accepted = 0
        nRemaining = n - people.shape[0]
        batch = nRemaining
        while nRemaining>0:
            if drawn>=maxDraws:
                #nothing passed at all, so no budget is large enough and asking for a larger one misleads
                if accepted==0:
                    raise RuntimeError(f"None of the {drawn} rows sampled passed personFilters.")
                raise RuntimeError(f"""Reached {people.shape[0]} of n={n} people in {drawn} draws
                                       (acceptance rate {accepted/drawn:.4f}). Raise maxDraws.""")
            #the first pass has no information about how many draws survive the filters and so draws exactly
            #the shortfall, later passes size the draw with the acceptance rate observed so far (with a 20%
            #margin) so that restrictive filters converge in a few passes instead of one pass per person.
            #While nothing has been accepted there is still no rate to size with, and repeating the shortfall
            #would spend the budget one shortfall at a time, so the batch doubles instead and the filters
            #that accept nobody are found out in ~log2 passes rather than maxDraws/n of them
            if accepted>0:
                batch = int(np.ceil(nRemaining/(accepted/drawn)*1.2))
            elif drawn>0:
                batch = 2*batch
            batch = min(batch, maxDraws-drawn)
            dfForPeople = df.sample(batch, replace=True, weights=weights)
            drawn += batch
            if distributions is not None:
                #each sampled row is redrawn on its own, so the people that came from one NHANES row differ
                dfForPeople = PopulationFactory.redraw_continuous_variables(dfForPeople, distributions)
                dfForPeople = PopulationFactory.apply_person_filters_on_df(personFilters, dfForPeople)
                if dfForPeople.shape[0]==0: #every drawn row was rejected, and apply on an empty df has no rows
                    continue
            peopleRemaining = pd.DataFrame.apply(dfForPeople, PersonFactory.get_person, popType=popType, initializationModelRepository=initializationModelRepository, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, axis="columns")
            peopleRemaining = PopulationFactory.apply_person_filters_on_people(personFilters, peopleRemaining)
            accepted += peopleRemaining.shape[0]
            people = pd.concat([people, peopleRemaining])
            nRemaining = n - people.shape[0]
        #the draws are iid so keeping the first n of an overshooting batch does not bias the sample
        return people.iloc[:n]

    @staticmethod
    def get_kaiser_people(n=1000, personFilters=None, wmhSpecific=None):
        '''The wmhSpecific variable is not needed in the function but it is passed on to the function from the trial.py
        because the NHANES get_nhanes_people function needs to get arguments from the trial.py.
        Creating Kaiser people is a time consuming process, in part due to the time needed to get the distributions.
        That is why we do that step only once and create a pandas dataframe only once.
        Since we need to plan for the possibility of using filters, and sometimes filters can be fairly restrictive,
        we need to use sampling with replacement from the dataframe. 
        It is unclear what memory needs we would have in order to create always a much larger sample than the one we need in
        simulations in order to avoid sampling with replacement.'''
        distributions = PopulationFactory.get_kaiser_distributions()
        drawsForGroups, namesForGroups = PopulationFactory.draw_from_distributions(distributions)
        df = PopulationFactory.get_df_from_draws(drawsForGroups, namesForGroups, popType=PopulationType.KAISER.value)
        df = PopulationFactory.apply_person_filters_on_df(personFilters, df)
        if df.shape[0] == 0: #stop with a clear error, df.sample on an empty df raises an opaque one
            raise RuntimeError("""The df-level filters of personFilters rejected every row drawn from
                                  the Kaiser distributions, so there is nobody left to build Person-objects from.""")
        dfForPeople = df.sample(n, weights=None, replace=True)
        imr = InitializationModelRepository()
        people = pd.DataFrame.apply(dfForPeople, PersonFactory.get_kaiser_person, args=(imr,), axis="columns")
        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
        #weights=None because the initial Kaiser draw above is unweighted as well
        people = PopulationFactory.bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.KAISER.value, initializationModelRepository=imr, weights=None)
        #sampling with replacement leaves duplicate index labels, what identifies a person is _index
        people = people.reset_index(drop=True)
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_state_people(year=2030, personFilters=None, state="OH", samplingRate=0.025):
        '''Creates people as a representative part of a state's population a given year.
        The argument samplingRate indicates what proportion of the state's population we will simulate.
        Note that due to rounding by using sampling the increase in the size of the people creates is not proportional to the increase
        in the samplingRate.'''
        #df with only categorical variables completed
        dfWithCategoricals = PopulationFactory.get_dataframe_with_categoricals(year=year, state=state, samplingRate=samplingRate) 
        #get Gaussian distributions of continuous variables stratified...
        distributions = PopulationFactory.get_crude_distributions()
        #each row of dfWithCategoricals gets values for continuous variables based on the distributions
        df = PopulationFactory.append_dataframe_with_continuous(dfWithCategoricals, distributions)
        #df-level filters run after the draw because the values they have to hold for are the drawn ones.
        #No redraw of rejected rows: each row is a fixed slice of the state population, so a filter
        #shrinks the population to the filtered subpopulation at the same sampling rate
        df = PopulationFactory.apply_person_filters_on_df(personFilters, df)
        if df.shape[0] == 0:
            raise RuntimeError("""The df-level filters of personFilters rejected every row of the state
                                  population, so there is nobody left to build Person-objects from.""")
        imr = InitializationModelRepository()
        opmr = OutcomePrevalenceModelRepository()
        people = pd.DataFrame.apply(df, PersonFactory.get_nhanes_person, args=(imr,), outcomePrevalenceModelRepository=opmr, axis="columns")
        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
        if people.shape[0] == 0:
            raise RuntimeError("""The person-level filters of personFilters rejected all the Person-objects
                                  built from the state population, so the population would be empty.""")
        #the two explodes leave duplicate index labels, what identifies a person is _index
        people = people.reset_index(drop=True)
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_crude_distributions():
        '''Returns the Gaussians the continuous variables are drawn from, fit on the partition of NHANES
        by gender, race ethnicity, education and a 5-year age window.

        Built once and kept in _crudeDistributions. Neither the partition nor the fit takes an argument,
        so both give the same answer every time they are called, and together they cost about 5 seconds:
        partitioning scans the whole NHANES df once per combination of the four variables. Unlike
        get_nhanesDf this hands back the cached object itself rather than a copy, because everything
        downstream only reads from it.'''
        if PopulationFactory._crudeDistributions is None:
            partitionedNhanesDf = PopulationFactory.get_partitioned_nhanes_people_crude()
            PopulationFactory._crudeDistributions = PopulationFactory.get_distributions_crude(partitionedNhanesDf)
        return PopulationFactory._crudeDistributions

    @staticmethod
    def get_partitioned_nhanes_people_crude():
        '''Partitions the NHANES data, all rows, according to 4 categorical variables, the ones that are the most important overall for the prediction
        of continuous variables by using Gaussian distributions.
        Because the continuous variable distributions do not differ much for ages that are off by 1 or 2 years, use a range of ages and not just an exact age match.'''
        df = PopulationFactory.get_nhanesDf_resampled() #weighted bootstrap so the fits reflect the survey weights
        dictForCategoricals = dict()
        for gender, raceEthnicity, education, age in product(
                                                       set(df[StaticRiskFactorsType.GENDER.value].tolist()),
                                                       set(df[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist()),
                                                       set(df[StaticRiskFactorsType.EDUCATION.value].tolist()),
                                                       set(range(0,82,1))): #for age
            #symmetric window of ages age-2 to age+2, clamped at 0; range excludes ageMax.
            #The top key's window runs to 85 instead: the file holds ages up to 85 (NHANES top-codes
            #age there), and the 988 people aged 84-85 are too few for keys of their own — extending
            #the keys past 81 leaves groups singular even after pooling education, while widening the
            #top window to 79-85 keeps every fit non-singular and lets their data feed the fit that
            #serves them (see get_dist_keys_for_dataframe, which clips the draw key to 81)
            ageMin = max(age-2, 0)
            ageMax = 86 if age==81 else age+3
            dfForCategoricals = df.loc[(df[StaticRiskFactorsType.GENDER.value]==gender) & 
                                       (df[DynamicRiskFactorsType.AGE.value].isin(list(range(ageMin,ageMax,1)))) & 
                                       (df[StaticRiskFactorsType.EDUCATION.value]==education) &
                                       (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==raceEthnicity), :]#.copy()
            if dfForCategoricals.shape[0]>0:
                dictForCategoricals[gender,raceEthnicity,education,age] = dfForCategoricals
        return dictForCategoricals

    @staticmethod
    def get_distributions_crude(dfForCategoricals):
        '''dfForCategoricals: a dictionary with keys gender, raceEthnicity, education, age and values a dataframe based on NHANES dataframe
        This function will attempt to fit Gaussian distributions for the continuous variables using this key and value 
        But if a singular Gaussian is created then the education level is removed from the key and the Gaussian is created by
        combining the values with all education levels
        If removing the education from the key does not provide a non-singular Gaussian we will need to fix it...'''
        meanForCategoricals = dict()
        covForCategoricals = dict()
        singularForCategoricals = dict()
        minForCategoricals = dict()
        maxForCategoricals = dict()
        for key in dfForCategoricals.keys():
            nhanesContinuousVariables = PopulationFactory.continuous_variables_for_key_age(key[3])
            meanForCategoricals[key], covForCategoricals[key] = multivariate_normal.fit(np.array(dfForCategoricals[key][nhanesContinuousVariables]))
            singularForCategoricals[key] = PopulationFactory.is_singular(covForCategoricals[key]) #some distributions might be singular
            minForCategoricals[key] = np.min(np.array(dfForCategoricals[key][nhanesContinuousVariables]), axis=0)
            maxForCategoricals[key] = np.max(np.array(dfForCategoricals[key][nhanesContinuousVariables]), axis=0)
        #keysToRemove = list() #these are the distributions that are singular
        keysSingular = list(filter(lambda x: singularForCategoricals[x], singularForCategoricals.keys()))
        for key in keysSingular:
            keyMinusEducation = tuple(list(key[0:2]) + [key[3]]) #key includes gender, race ethnicity, education, age
            if keyMinusEducation not in meanForCategoricals.keys(): #I might have done the fit on a prior pass
                allEducationKeys = [list(key[0:2]) + [ed.value, key[3]] for ed in Education]
                allEducationKeys = list(filter(lambda x: tuple(x) in list(dfForCategoricals.keys()), allEducationKeys))
                dfForAllEducationKeys = pd.concat([dfForCategoricals[tuple(edKey)] for edKey in allEducationKeys], ignore_index=True)
                nhanesContinuousVariables = PopulationFactory.continuous_variables_for_key_age(key[3])
                meanForCategoricals[keyMinusEducation], covForCategoricals[keyMinusEducation] = multivariate_normal.fit(
                    np.array(dfForAllEducationKeys[nhanesContinuousVariables]))
                singularForCategoricals[keyMinusEducation] = PopulationFactory.is_singular(covForCategoricals[keyMinusEducation])
                minForCategoricals[keyMinusEducation] = np.min(np.array(dfForAllEducationKeys[nhanesContinuousVariables]), axis=0)
                maxForCategoricals[keyMinusEducation] = np.max(np.array(dfForAllEducationKeys[nhanesContinuousVariables]), axis=0)
                if singularForCategoricals[keyMinusEducation]: #if removing education does not create a non-singular Gaussian the process failed
                    raise RuntimeError("Process of creating non-singular Gaussian distributions has failed.")
            #keysToRemove.append(key)    
        #for key in keysToRemove:
        for key in keysSingular: 
            del singularForCategoricals[key]
            del meanForCategoricals[key]
            del covForCategoricals[key]
        distributions = {"mean": meanForCategoricals, "cov": covForCategoricals, "singular": singularForCategoricals,
                         "min": minForCategoricals, "max": maxForCategoricals}
        return distributions


    @staticmethod
    def get_dataframe_with_categoricals(year=2030, state="OH", samplingRate=0.01):
        '''Returns dataframe with complete categorical variables but no continuous variables, with each row
        corresponding to a single person.
        Because state population projections do not include information on default treatments, we will use NHANES data to partition each group to
        a meaningful default treatment group.'''
        df = PopulationFactory.get_stateDf(year=year, state=state)
        #partition the people to default treatments in a similar way as found in the nhanes data
        proportionForDefaultTreatments = PopulationFactory.get_proportionForDefaultTreatments()
        df['nForAgeAndDefaultTreatments'] = df.apply(lambda x: PopulationFactory.get_nForDefaultTreatments(
                                                                 x["ageGroup"], x["gender"], x["raceEthnicity"], x["statin"], 
                                                                 x["antiHypertensiveCount"], proportionForDefaultTreatments, x["nForAge"]), axis=1)
        df = df.loc[ (df["nForAgeAndDefaultTreatments"]>0) ] #keep only the rows that have 1 or more people
        df["name"] = np.arange(len(df)) #people with the same categorical variables will have the same name
        df["nForSampling"] = df["nForAgeAndDefaultTreatments"].apply(lambda x: range(math.floor(x*samplingRate + 0.5))) #this is how samplingRate influences the number of people created
        df = df.explode("nForSampling")
        #explode turns an empty range into a NaN row, which would create a person from every group whose sampled count rounded to 0
        df = df.dropna(subset=["nForSampling"])
        df["modality"] = Modality.NO.value #all NHANES people will have the same modality
        return df 

    @staticmethod
    def get_stateDf(year=2030, state='OH'):
        '''Reads the CSV file that includes some categorical variables for each state and year and performs a bit of initial processing.
        Returns a dataframe that includes a portion of the microsim categorical variables and the number of people in that state by age.'''
        dataDir = get_absolute_datafile_path("state")
        data = pd.read_csv(dataDir+f"/pop_projection_{state.lower()}_{year}.csv")
        #the whole state pipeline is built on NHANES, so a race NHANES holds nobody of (eg ASIAN) has
        #neither treatment proportions nor risk-factor distributions; fail here with the codes rather
        #than with a KeyError deep in an apply
        nhanesRaces = set(int(r) for r in PopulationFactory.get_nhanesDf()[StaticRiskFactorsType.RACE_ETHNICITY.value].unique())
        unsupportedRaces = set(int(r) for r in data[StaticRiskFactorsType.RACE_ETHNICITY.value].unique()) - nhanesRaces
        if unsupportedRaces:
            raise RuntimeError(f"""The {state} {year} projection contains raceEthnicity codes {sorted(unsupportedRaces)}
                                   that NHANES holds no people of, so no treatment proportions or risk-factor
                                   distributions exist for them. Map those rows to one of {sorted(nhanesRaces)}
                                   in the projection file.""")
        data[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value] = data[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)
        #the projection carries alcohol as a 0-3 category; persons store drinks/week and the distributions
        #draw it as a continuous variable, so the projection's alcohol composition is no longer used
        data = data.drop(columns=[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value])
        ageList5Years = [x for x in range(0,5)]
        ageList10Years = [x for x in range(0,10)] #this is for the last age group, the oldest
        data['age'] = data['ageGroup'].apply(lambda i: [x+(i-1)*len(ageList5Years) for x in ageList5Years] if i!=17 else 
                                                       [x+(i-1)*len(ageList5Years) for x in ageList10Years]  ) #not a typo
        data = data.explode('age')
        data["nForAge"] = data.apply(lambda x: PopulationFactory.get_nForAge_from_nForAgeGroup(x["age"],x["n"]), axis=1)
        data["statinAntihypertensiveCount"] = [[[st, an] for st in [True, False] for an in PopulationFactory.antiHypertensiveCounts]] * len(data)
        data = data.explode("statinAntihypertensiveCount")
        data[["statin", "antiHypertensiveCount"]] = pd.DataFrame(data["statinAntihypertensiveCount"].tolist(), index=data.index)
        return data

    @staticmethod
    def get_nForAge_from_nForAgeGroup(age, nForAgeGroup):
        '''Returns the number of people we would reasonably expect to find with given age and number of people in age group.
        Uniform distribution over ages for ages less than 80 years old and a decreasing distribution after that.'''
        if age<80:
            return round(nForAgeGroup/5) #divides the number of people to equal parts for all ages in ageGroup
        elif age < 90:
            #the coefficients are np.linspace(10, 1, 10) normalized: the taper ends above zero so that
            #age 89 exists at baseline (linspace(9, 0, 10) gave it exactly 0 people). Ages 90+ are not
            #represented at baseline, their share of the 80+ group is spread over 80-89.
            proportionsForAgeDict = {80: 10/55, 81: 9/55, 82: 8/55, 83: 7/55, 84: 6/55,
                                     85: 5/55, 86: 4/55, 87: 3/55, 88: 2/55, 89: 1/55}
            return round(nForAgeGroup * proportionsForAgeDict[age]) #divides the number of people to decreasing parts as age increases
        else:
            raise RuntimeError(f"Age {age} is not covered: the state pipeline generates baseline ages up to 89.")
 
    @staticmethod
    def get_ageGroup_from_age(age):
        '''Returns age group given age.
        Age groups include 5 ages until age 79, and anyone older than 79 belongs to age group 17.'''
        if age>=80:
            return 17
        else:
            return age//5 + 1

    @staticmethod
    def get_nForDefaultTreatments(ageGroup, gender, raceEthnicity, statin, antiHypertensiveCount, proportionForDefaultTreatments, nForAge):
        '''Returns the number of people we expect to find with given statin  and antiHypertensiveCount from the number of 
        people with that age, the ageGroup, gender and raceEthnicity'''
        return int(round(proportionForDefaultTreatments[ageGroup, gender, raceEthnicity][statin, antiHypertensiveCount] * nForAge))

    @staticmethod
    def get_proportionForDefaultTreatments():
        '''For a given age group, gender, race ethnicity, statin, anti hypertensive count returns the proportion of NHANES people
        that have given statin and anti hypertensive count from all NHANES people with given age group, gender, race ethnicity.'''
        weightForTreatments = dict()
        proportionForTreatments = dict()
        df = PopulationFactory.get_nhanesDf() 
        df["ageGroup"] = df["age"].apply(lambda x: PopulationFactory.get_ageGroup_from_age(x))
        df["age"] = df["age"].astype(int)
        for ageGroup, gender, raceEthnicity in product(
                                                list(range(1,18,1)),
                                                #set(data[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                [ge.value for ge in NHANESGender],
                                                #set(data[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist())):
                                                [ra.value for ra in RaceEthnicity if ra.value!=6]): #NHANES does not include any asian...
            proportionForTreatments[ageGroup, gender, raceEthnicity] = dict()
            weightForTreatments = dict()
            sumForKey = 0
            for statin in [True, False]:
                for antiHypertensiveCount in PopulationFactory.antiHypertensiveCounts:
                    dfForGroup = df.loc[
                                    (df["ageGroup"]==ageGroup) &
                                    (df[StaticRiskFactorsType.GENDER.value]==gender) & 
                                    (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==raceEthnicity) &
                                    (df[DefaultTreatmentsType.STATIN.value]==statin) &
                                    (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]==antiHypertensiveCount), :].copy()
                    if dfForGroup.shape[0]>0:
                        weightForTreatments[statin, antiHypertensiveCount] = sum(dfForGroup.loc[:,"WTINT2YR"].tolist())
                        sumForKey += weightForTreatments[statin, antiHypertensiveCount]
                    else:
                        weightForTreatments[statin, antiHypertensiveCount] = 0
            if sumForKey==0.:
                raise RuntimeError(f"Did not find NHANES people-data with gender {gender}, raceEthnicity {raceEthnicity}, age group {ageGroup}") 
            for statin in [True, False]:
                for antiHypertensiveCount in PopulationFactory.antiHypertensiveCounts:
                    proportion =  weightForTreatments[statin, antiHypertensiveCount]/sumForKey if sumForKey>0. else 0.
                    proportionForTreatments[ageGroup, gender, raceEthnicity][statin, antiHypertensiveCount] = proportion
        return proportionForTreatments

    @staticmethod
    def redraw_continuous_variables(df, distributions):
        '''Replaces the continuous variables of every row of df with a draw from the distributions.

        The categorical variables and the age of each row are kept as they are, which is what makes this
        the same construction the state populations use: there the categorical variables and the age come
        from a state projection, here they come from the NHANES rows themselves, and in both cases only
        the continuous variables are drawn. The draw is made from a Gaussian fit on the few categorical
        variables that matter most for the continuous ones, and is then shifted to the mean of the group
        the row belongs to (see group_key_frame).'''
        df = df.copy()
        #age is not redrawn, it is one of the keys the distributions are stored under
        continuousToRedraw = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        continuousToRedraw.remove(DynamicRiskFactorsType.AGE.value)
        df = df.drop(columns=continuousToRedraw)
        return PopulationFactory.append_dataframe_with_continuous(df, distributions)

    @staticmethod
    def append_dataframe_with_continuous(dfWithCategoricals, distributions):
        '''Takes a dataframe where all categorical variables exist for each row, and uses the distributions to append columns
        with all continuous variables.
        The complete dataframe is returned.'''
        nhanesContinuousVariables = PopulationFactory.continuous_variables_drawn()
        #a draw is a point of its own distribution, and what is wanted is a point of the group the person
        #belongs to, so the draw is kept as its distance from the mean of the distribution it came from and
        #that distance is measured out from the mean of the group instead
        distKeysForRows = PopulationFactory.get_dist_keys_for_dataframe(dfWithCategoricals, distributions)
        alcIdx = nhanesContinuousVariables.index(DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value)
        drawMeans = np.empty((dfWithCategoricals.shape[0], len(nhanesContinuousVariables)))
        for distKey, rowsOfKey in distKeysForRows:
            if distKey[-1] < PopulationFactory.ALCOHOL_MIN_KEY_AGE: #the key's distribution has no alcohol dimension
                drawMeans[rowsOfKey] = np.insert(distributions["mean"][distKey], alcIdx, 0.)
            else:
                drawMeans[rowsOfKey] = distributions["mean"][distKey]
        groupMeans = PopulationFactory.get_group_means_for_dataframe(dfWithCategoricals, drawMeans)
        #the shift is handed to the draw rather than added to the draw afterwards because the value that is
        #kept is the shifted one, so the shifted one is what the bounds have to hold for: a draw inside the
        #bounds of its distribution can be shifted outside of them, and outside of them meant a negative
        #trig, ldl, hdl or creatinine for 151 of the 5448 people of NHANES 1999 when the shift was applied
        #after the bounds had been checked
        shift = groupMeans - drawMeans
        if "year" in dfWithCategoricals.columns: #state projections carry no NHANES year and keep pooled levels
            yearKeyFrame = PopulationFactory.year_correction_key_frame(dfWithCategoricals)
            yearCorrection = PopulationFactory.get_year_corrections().reindex(
                pd.MultiIndex.from_frame(yearKeyFrame)).to_numpy()
            shift = shift + np.nan_to_num(yearCorrection) #a cell NHANES does not hold stays uncorrected
        draws = PopulationFactory.get_draws_for_dataframe(distKeysForRows, distributions, shift)
        dfWithContinuous = pd.DataFrame(draws,
                                        columns=nhanesContinuousVariables,
                                        index=dfWithCategoricals.index)
        return pd.concat([dfWithCategoricals, dfWithContinuous], axis=1)

    @staticmethod
    def continuous_variables_drawn():
        '''The continuous variables that are drawn: all of them except age, which is not drawn but is one
        of the keys the distributions are stored under.'''
        continuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        continuousVariables.remove(DynamicRiskFactorsType.AGE.value)
        return continuousVariables

    #all NHANES rows under 18 have alcoholPerWeek 0, so the child/teen age windows have no alcohol
    #variance to fit; fits at key ages >=18 were verified non-singular after education pooling
    ALCOHOL_MIN_KEY_AGE = 18

    @staticmethod
    def continuous_variables_for_key_age(age):
        '''The continuous variables the distribution of a key holds: alcohol is left out of keys below
        ALCOHOL_MIN_KEY_AGE, and everyone drawn from such a key gets exactly 0 drinks. Both the fit and
        the draw read the column set from here so the two sides cannot drift apart.'''
        continuousVariables = PopulationFactory.continuous_variables_drawn()
        if age < PopulationFactory.ALCOHOL_MIN_KEY_AGE:
            continuousVariables.remove(DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value)
        return continuousVariables

    @staticmethod
    def get_dist_keys_for_dataframe(dfWithCategoricals, distributions):
        '''Returns, for every distribution that has rows drawing from it, that distribution's key and the
        positions of those rows.

        The rows are handled one distribution at a time rather than one row at a time: every row with the
        same gender, race ethnicity, education and age draws from the same Gaussian, so that Gaussian is
        built once and asked for as many points as there are such rows.'''
        #the keys stop at 81, rows older than that (state projections carry ages to 89) draw from the
        #81 key, whose window runs to 85 so its fit includes the NHANES people it serves
        keyFrame = pd.DataFrame({
            StaticRiskFactorsType.GENDER.value: dfWithCategoricals[StaticRiskFactorsType.GENDER.value],
            StaticRiskFactorsType.RACE_ETHNICITY.value: dfWithCategoricals[StaticRiskFactorsType.RACE_ETHNICITY.value],
            StaticRiskFactorsType.EDUCATION.value: dfWithCategoricals[StaticRiskFactorsType.EDUCATION.value],
            DynamicRiskFactorsType.AGE.value: dfWithCategoricals[DynamicRiskFactorsType.AGE.value].clip(upper=81)})
        distKeysForRows = list()
        for key, rowsOfKey in keyFrame.groupby(list(keyFrame.columns), observed=True).indices.items():
            #a group whose own distribution was singular was fit without education, see get_distributions_crude
            distKey = key if key in distributions["mean"].keys() else (key[0], key[1], key[3])
            distKeysForRows.append((distKey, rowsOfKey))
        return distKeysForRows

    @staticmethod
    def get_draws_for_dataframe(distKeysForRows, distributions, shift):
        '''Draws the continuous variables of every row, one distribution at a time.

        distKeysForRows: the (distribution key, row positions) pairs of get_dist_keys_for_dataframe.
        shift: what is added to the draw of each row in order to move it from the mean of the distribution
               it was drawn from to the mean of the group the row belongs to. It is applied before the
               bounds are checked, see draw_within_bounds.

        Returns the drawn, shifted values.'''
        draws = np.empty_like(shift)
        alcIdx = PopulationFactory.continuous_variables_drawn().index(DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value)
        clipped = 0
        for distKey, rowsOfKey in distKeysForRows:
            dist = multivariate_normal(distributions["mean"][distKey], distributions["cov"][distKey],
                                       allow_singular=False)
            hasAlcohol = distKey[-1] >= PopulationFactory.ALCOHOL_MIN_KEY_AGE
            shiftForKey = shift[rowsOfKey] if hasAlcohol else np.delete(shift[rowsOfKey], alcIdx, axis=1)
            drawsForKey, clippedForKey = PopulationFactory.draw_within_bounds(dist,
                                                                    distributions["min"][distKey],
                                                                    distributions["max"][distKey],
                                                                    len(rowsOfKey), shift=shiftForKey)
            if not hasAlcohol: #alcohol is exactly 0 below the cutoff, never drawn and shifted
                drawsForKey = np.insert(drawsForKey, alcIdx, 0., axis=1)
            draws[rowsOfKey] = drawsForKey
            clipped += clippedForKey
        if clipped > 0:
            print(f"""Warning: {clipped} of {draws.shape[0]} people had a group mean too far from the distribution
                      they were drawn from for the bounds of that distribution to be met, and were clipped to
                      those bounds. Their group holds too few people for its mean to be a reliable one.""")
        return draws

    @staticmethod
    def draw_within_bounds(dist, distMin, distMax, size, shift=None, maxAttempts=100):
        '''Returns size draws from dist, shifted by shift, redrawing the ones that fall outside the bounds.

        Gaussians extend to infinity while the people they were fit on do not, so a draw that no NHANES
        person comes close to is thrown away and made again. The bounds are widened by 10% first, so that
        the edges of the observed range stay reachable.

        shift, one row of it per draw, is added to a draw before its bounds are checked, because it is the
        shifted value that is kept and therefore the shifted value the bounds have to hold for. Rows of the
        same distribution do not share a shift, so each row is redrawn on its own until its own shifted
        draw is in bounds.

        maxAttempts bounds the number of redraws of a single row, because a shift can be large enough that
        almost no draw meets the bounds under it. A row that exhausts it keeps its last draw clipped into
        the bounds, and is counted in the second return value: such a shift comes from the mean of a group
        of very few people, and leaving that row at the edge of what NHANES holds is better than both
        raising on data that is otherwise fine and keeping a value that no person could have.

        Returns the draws and how many of them had to be clipped.'''
        lowerBound, upperBound = 0.9*distMin, 1.1*distMax
        shift = np.zeros((size, len(distMin))) if shift is None else shift
        draws = np.empty((size, len(distMin)))
        pending = np.arange(size)
        for attempt in range(maxAttempts):
            newDraws = np.atleast_2d(dist.rvs(size=pending.shape[0])) + shift[pending]
            inBounds = ((newDraws >= lowerBound) & (newDraws <= upperBound)).all(axis=1)
            draws[pending[inBounds]] = newDraws[inBounds]
            pending = pending[~inBounds]
            if pending.shape[0] == 0:
                return draws, 0
            #the rows that are still pending keep this draw if the attempts run out, hence the bookkeeping
            lastDraws = newDraws[~inBounds]
        draws[pending] = np.clip(lastDraws, lowerBound, upperBound)
        return draws, pending.shape[0]

    @staticmethod
    def get_group_means_for_dataframe(dfWithCategoricals, drawMeans):
        '''Returns, for every row of dfWithCategoricals, the mean of its group, and for the rows whose
        group NHANES does not hold at all the mean of the distribution the row was drawn from, which
        leaves that row where its draw put it.'''
        groupMeans = PopulationFactory.look_up_group_means(dfWithCategoricals)
        return np.where(np.isnan(groupMeans), drawMeans, groupMeans)

    @staticmethod
    def look_up_group_means(dfWithCategoricals):
        '''Looks up the mean of the group of every row, np.nan for the rows whose group is not there.'''
        keyFrame = PopulationFactory.group_key_frame(dfWithCategoricals)
        return PopulationFactory.get_group_means().reindex(pd.MultiIndex.from_frame(keyFrame)).to_numpy()

    @staticmethod
    def group_key_frame(df):
        '''The variables a group is defined by, read off any df that carries the categorical variables.

        Both the means and the look-up of those means are built from this one function, so the key that
        the means are stored under and the key they are asked for cannot drift apart. The columns are
        cast because the two sides do not always arrive with the same dtype -- the age of a state
        projection, for one, comes out of an explode as object -- and a MultiIndex whose level dtype
        differs matches nothing, which would silently leave every row unshifted.

        Which variables these are is a compromise. A group has to be fine enough to say something the
        crude distribution does not already say, and large enough for its mean to be worth using: the
        draw already conditions on gender, race ethnicity, education and age, so what a group adds is
        whether the person is on antihypertensives (which decides sbp and dbp) and whether they are
        physically active (bmi and waist). Both are taken as yes/no rather than as the count and any
        finer grouping, which keeps the median group at 54 people. Grouping on all nine categorical
        variables, as this used to, left the median group holding ONE person, so the "mean" was that
        person and the shift carried their noise into everyone drawn for that group.'''
        return pd.DataFrame({
            StaticRiskFactorsType.GENDER.value: df[StaticRiskFactorsType.GENDER.value].astype(int),
            StaticRiskFactorsType.RACE_ETHNICITY.value: df[StaticRiskFactorsType.RACE_ETHNICITY.value].astype(int),
            StaticRiskFactorsType.EDUCATION.value: df[StaticRiskFactorsType.EDUCATION.value].astype(int),
            "ageGroup": df[DynamicRiskFactorsType.AGE.value].apply(PopulationFactory.get_ageGroup_from_age).astype(int),
            "anyAntiHypertensive": (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value] > 0).astype(bool),
            DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)})

    @staticmethod
    def get_group_means():
        '''The mean of every continuous variable, for every group of NHANES people.

        This is all the draws need from those groups: a draw is re-centered on the mean of the group of
        the person it is for. A group is far too small to fit a covariance matrix on, which is why the
        distributions themselves are fit on a much coarser partition, but it is big enough for a mean.'''
        if PopulationFactory._groupMeans is None:
            df = PopulationFactory.get_nhanesDf_resampled() #same bootstrap as the fits, so shift and draw agree on the weights
            keyFrame = PopulationFactory.group_key_frame(df)
            PopulationFactory._groupMeans = df[PopulationFactory.continuous_variables_drawn()].groupby(
                                                [keyFrame[column] for column in keyFrame.columns],
                                                observed=True).mean()
        return PopulationFactory._groupMeans

    @staticmethod
    def year_correction_key_frame(df):
        '''The cells the per-year correction is defined by. Coarse on purpose: these cells hold a median
        of ~226 people per year, while the fine group_key_frame cells hold ~2 per year, far too few for
        year-specific means. Both the correction table and its look-up are built from this one function.'''
        return pd.DataFrame({
            "year": df["year"].astype(int),
            StaticRiskFactorsType.GENDER.value: df[StaticRiskFactorsType.GENDER.value].astype(int),
            "ageGroup": df[DynamicRiskFactorsType.AGE.value].apply(PopulationFactory.get_ageGroup_from_age).astype(int)})

    @staticmethod
    def get_year_corrections():
        '''yearCellMean - pooledCellMean of every drawn continuous variable, per (year, gender, ageGroup).

        The Gaussians and the group means are fit on all NHANES years pooled, so a drawn population
        inherits the pooled levels (eg the later years' bmi in a 1999 population). Adding this difference
        to the shift moves each row to its own year's level while keeping the pooled covariance.
        Computed on the same weighted bootstrap as the fits and the group means.'''
        if PopulationFactory._yearCorrections is None:
            df = PopulationFactory.get_nhanesDf_resampled()
            keyFrame = PopulationFactory.year_correction_key_frame(df)
            contVars = PopulationFactory.continuous_variables_drawn()
            yearMeans = df[contVars].groupby([keyFrame[column] for column in keyFrame.columns],
                                             observed=True).mean()
            pooledMeans = df[contVars].groupby([keyFrame[column] for column in keyFrame.columns[1:]],
                                               observed=True).mean()
            pooledAligned = pooledMeans.reindex(yearMeans.index.droplevel("year"))
            PopulationFactory._yearCorrections = pd.DataFrame(
                yearMeans.to_numpy() - pooledAligned.to_numpy(), index=yearMeans.index, columns=contVars)
        return PopulationFactory._yearCorrections
