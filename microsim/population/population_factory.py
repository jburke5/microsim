import copy
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import multivariate_normal
from scipy.optimize import brentq
from enum import Enum
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
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.population.standardized_population import StandardizedPopulation
from microsim.common.variable_type import VariableType
from microsim.outcomes.outcome import OutcomeType
from microsim.common.population_type import PopulationType
from microsim.risk_factors.modality import Modality

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
            return PopulationFactory.get_nhanes_population_model_repo()
        elif popType == PopulationType.KAISER:
            return PopulationFactory.get_kaiser_population_model_repo(**kwargs)
        elif popType == PopulationType.STATE:
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
        #convert the integers to booleans because in the simulation we always use bool for these
        for col in [DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value, DefaultTreatmentsType.STATIN.value]:
            nhanesDf[col] = nhanesDf[col].astype(bool)
        #convert drinks per week to category
        nhanesDf[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value] = nhanesDf.apply(lambda x:
                                                                                 AlcoholCategory.get_category_for_consumption(x[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]), axis=1)
        #convert these columns to int type
        for col in [StaticRiskFactorsType.RACE_ETHNICITY.value, 
                    StaticRiskFactorsType.EDUCATION.value,
                    StaticRiskFactorsType.GENDER.value,
                    StaticRiskFactorsType.SMOKING_STATUS.value]:
            nhanesDf[col] = nhanesDf[col].astype(int)
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
    def get_nhanes_people(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None, outcomePrevalenceModelRepository=None):
        '''Returns a Pandas Series object with Person-Objects of all persons included in NHANES for year 
           with or without sampling. Filters are applied prior to sampling in order to maximize efficiency and minimize
           memory utilization. This does not affect the distribution of the relative percentages of groups 
           represented in people.
           The flag distributions controls if the Person-objects will come directly from the NHANES data or
           if Gaussian distributions will first be fit to the NHANES data and then draws are obtained from the distributions.'''

        if (year is not None) and (year not in [2011, 2015, 2007, 2003, 2009, 2001, 2005, 1999, 2013, 2017]):
            raise RuntimeError(f"NHANES data for year {year} is not available")

        nhanesDf = PopulationFactory.get_nhanesDf()        

        if year is not None: #if year is None, then use the entire dataframe
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]

        if personFilters is None: #since we started including children in the NHANES df, by default use an adult filter on the df
            personFilters = PersonFilterFactory.get_person_filter()
        else:
            print("Warning: NHANES populations now include children by default. Add an age filter for adults only.")

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

        imr = InitializationModelRepository()
        people = pd.DataFrame.apply(nhanesDfForPeople, PersonFactory.get_nhanes_person, args=(imr,), outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, axis="columns")

        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)

        if nhanesWeights:
            people = PopulationFactory.bring_people_to_target_n(n, people, nhanesDf, personFilters, popType=PopulationType.NHANES.value, initializationModelRepository=imr, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository)
            
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
    def get_nhanes_population(n=None, year=None, personFilters=None, nhanesWeights=False, distributions=False, customWeights=None, riskScaling=None, prevalenceRiskScaling=None):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.
           Person attributes can originate either from the NHANES dataset directly or from distributions fit to the NHANES dataset.
           riskScaling: optional dict[OutcomeType, float] applied to per-outcome risk inside the OutcomeModelRepository.
           prevalenceRiskScaling: optional dict[OutcomeType, float] applied to per-outcome priorToSim risk inside the OutcomePrevalenceModelRepository.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, personFilters=personFilters, nhanesWeights=nhanesWeights, distributions=distributions, customWeights=customWeights, outcomePrevalenceModelRepository=OutcomePrevalenceModelRepository(riskScaling=prevalenceRiskScaling))
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

           Returns: float scaling to feed into prevalenceRiskScaling={scaleOutcomeType: s}.'''

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
    def get_state_population(proportion=0.01, year=2030, personFilters=None):
        people = PopulationFactory.get_state_people(proportion=proportion, year=year, personFilters=personFilters)
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
        for mo, ge, sm, ra, st, ed, al, a, an in product(set(df[StaticRiskFactorsType.MODALITY.value].tolist()),
                                                     set(df[StaticRiskFactorsType.GENDER.value].tolist()), 
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
                dfForGroups[mo, ge, sm, ra, st, ed, al, a, an] = dfForGroup
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
    def bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.NHANES.value, initializationModelRepository=None, outcomePrevalenceModelRepository=None):
        nRemaining = n - people.shape[0]
        while nRemaining>0:
            dfForPeople = df.sample(nRemaining, replace=True)
            peopleRemaining = pd.DataFrame.apply(dfForPeople, PersonFactory.get_person, popType=popType, initializationModelRepository=initializationModelRepository, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository, axis="columns")
            peopleRemaining = PopulationFactory.apply_person_filters_on_people(personFilters, peopleRemaining)
            people = pd.concat([people, peopleRemaining])
            nRemaining = n - people.shape[0]
        return people

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
        dfForPeople = df.sample(n, weights=None, replace=True)
        people = pd.DataFrame.apply(dfForPeople, PersonFactory.get_kaiser_person, axis="columns")
        people = PopulationFactory.apply_person_filters_on_people(personFilters, people)
        people = PopulationFactory.bring_people_to_target_n(n, people, df, personFilters, popType=PopulationType.KAISER.value)   
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
        partitionedNhanesDf = PopulationFactory.get_partitioned_nhanes_people_crude()
        distributions = PopulationFactory.get_distributions_crude(partitionedNhanesDf)
        #each row of dfWithCategoricals gets values for continuous variables based on the distributions
        df = PopulationFactory.append_dataframe_with_continuous(dfWithCategoricals, distributions)
        imr = InitializationModelRepository()
        opmr = OutcomePrevalenceModelRepository()
        people = pd.DataFrame.apply(df, PersonFactory.get_nhanes_person, args=(imr,), outcomePrevalenceModelRepository=opmr, axis="columns")
        PopulationFactory.set_index_in_people(people)
        return people

    @staticmethod
    def get_partitioned_nhanes_people_crude():
        '''Partitions the NHANES data, all rows, according to 4 categorical variables, the ones that are the most important overall for the prediction
        of continuous variables by using Gaussian distributions.
        Because the continuous variable distributions do not differ much for ages that are off by 1 or 2 years, use a range of ages and not just an exact age match.'''
        df = PopulationFactory.get_nhanesDf() 
        dictForCategoricals = dict()
        for gender, raceEthnicity, education, age in product(
                                                       set(df[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                       set(df[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist()),
                                                       set(df[StaticRiskFactorsType.EDUCATION.value].tolist()),
                                                       set(range(0,82,1))): #for age
            if age>1:
                ageMin = age-2
                ageMax = age+2
            elif age==1:
                ageMin = age-1
                ageMax = age+2
            elif age==0: #there is NOBODY in NHANES with age 0
                ageMin = age
                ageMax = age+2
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
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        #during the simulation, age is treated as a continuous variable, but when we are given a population projection that includes ageGroup
        #ageGroup and age are treated as a categorical variable that we know
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)
        meanForCategoricals = dict()
        covForCategoricals = dict()
        singularForCategoricals = dict()
        minForCategoricals = dict()
        maxForCategoricals = dict()
        for key in dfForCategoricals.keys():
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
        df["modality"] = Modality.NO.value #all NHANES people will have the same modality
        return df 

    @staticmethod
    def get_stateDf(year=2030, state='OH'):
        '''Reads the CSV file that includes some categorical variables for each state and year and performs a bit of initial processing.
        Returns a dataframe that includes a portion of the microsim categorical variables and the number of people in that state by age.'''
        dataDir = "microsim/data/state"
        data = pd.read_csv(dataDir+f"/pop_projection_{state.lower()}_{year}.csv")
        data[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value] = data[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].astype(bool)
        ageList5Years = [x for x in range(0,5)]
        ageList10Years = [x for x in range(0,10)] #this is for the last age group, the oldest
        data['age'] = data['ageGroup'].apply(lambda i: [x+(i-1)*len(ageList5Years) for x in ageList5Years] if i!=17 else 
                                                       [x+(i-1)*len(ageList5Years) for x in ageList10Years]  ) #not a typo
        data = data.explode('age')
        data["nForAge"] = data.apply(lambda x: PopulationFactory.get_nForAge_from_nForAgeGroup(x["age"],x["n"]), axis=1)
        data["statinAntihypertensiveCount"] = [[[st, an] for st in [True, False] for an in [0., 1., 2.]]] * len(data)
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
            #the coefficients were obtained with arr = np.linspace(9, 0, 10) and normalized_arr = arr / arr.sum()
            proportionsForAgeDict = {80: 0.2, 81:0.17777778, 82: 0.15555556, 83: 0.13333333, 84: 0.11111111,
                                     85: 0.08888889, 86: 0.06666667, 87: 0.04444444, 88: 0.02222222, 89: 0.}
            return round(nForAgeGroup * proportionsForAgeDict[age]) #divides the number of people to decreasing parts as age increases
 
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
                for antiHypertensiveCount in [0., 1., 2.]:
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
                for antiHypertensiveCount in [0., 1., 2.]:
                    proportion =  weightForTreatments[statin, antiHypertensiveCount]/sumForKey if sumForKey>0. else 0.
                    proportionForTreatments[ageGroup, gender, raceEthnicity][statin, antiHypertensiveCount] = proportion
        return proportionForTreatments

    def append_dataframe_with_continuous(dfWithCategoricals, distributions):
        '''Takes a dataframe where all categorical variables exist for each row, and uses the distributions to append columns
        with all continuous variables.
        The complete dataframe is returned.'''
        nhanesDfPartitioned = PopulationFactory.get_partitioned_nhanes_df_with_age_group()
        dfWithContinuous = dfWithCategoricals.apply(PopulationFactory.get_draws_from_distributions_adjusted, args=(distributions, nhanesDfPartitioned), axis=1)
        dfWithContinuous = pd.DataFrame(dfWithContinuous.tolist(), index=dfWithCategoricals.index)
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)         
        dfWithContinuous.columns = nhanesContinuousVariables 
        dfComplete = pd.concat([dfWithCategoricals, dfWithContinuous], axis=1)
        return dfComplete 

    @staticmethod
    def get_draws_from_distributions_adjusted(row, distributions, nhanesDfPartitioned):
        '''For each group of categorical variables, first obtains values for the continuous variables by using distributions only from the
        main 3 or 4 categoricals and then shifts the draw for the continuous variables to an amount equal to the difference of the means
        between the NHANES people that match all categorical variables and the distribution of the NHANES people using only the 3 or 4 categoricals.
        This is the method for adjusting the fact that we used a crude distribution to do the draw, using only the most important 3 or 4 categoricals.'''
        draws = PopulationFactory.get_draws_from_distributions_crude(row, distributions)

        ageGroup = row["ageGroup"]
        ge = row["gender"]
        sm = row["smokingStatus"]
        ra = row["raceEthnicity"]
        ed = row["education"]
        al = row["alcoholPerWeek"]
        a = row["anyPhysicalActivity"]
        st = row["statin"]
        an = row["antiHypertensiveCount"]
        age = row["age"]
        
        if age>80:
            age=80
            ageGroup=16

        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)

        distKey = (ge, ra, ed, age)
        distKey = distKey if distKey in distributions["mean"].keys() else (ge, ra, age)
        distMean = distributions["mean"][distKey]

        if (ge, sm, ra, st, ed, al, a, an, ageGroup) in nhanesDfPartitioned.keys():
            dfForGroup = nhanesDfPartitioned[ge, sm, ra, st, ed, al, a, an, ageGroup]
            meanOfGroup = np.mean(np.array(dfForGroup[nhanesContinuousVariables]), axis=0)
            meandiff = meanOfGroup - distMean
        else:
            meandiff= np.zeros(len(draws[0])) #if the specific group is not found at all in the Nhanes dataframe then just use the more crude estimate
        drawsShifted = np.array(draws[0]) + meandiff
        return drawsShifted

    def get_partitioned_nhanes_df_with_age_group():
        '''Uses all NHANES data, from all years, and then partitions the dataframe to a dictionary where keys are the set of categorical variables and values
        are dataframes with all NHANES rows that correspond to the specific values of the categorical variables.'''
        dfForGroups = dict()
        df = PopulationFactory.get_nhanesDf()
        df["ageGroup"] = df["age"].apply(lambda x: PopulationFactory.get_ageGroup_from_age(x))
        for ge, sm, ra, st, ed, al, a, an, ageGroup in product(
                                                     set(df[StaticRiskFactorsType.GENDER.value].tolist()), 
                                                     set(df[StaticRiskFactorsType.SMOKING_STATUS.value].tolist()),
                                                     set(df[StaticRiskFactorsType.RACE_ETHNICITY.value].tolist()),
                                                     set(df[DefaultTreatmentsType.STATIN.value].tolist()),
                                                     set(df[StaticRiskFactorsType.EDUCATION.value].tolist()),
                                                     set(df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value].tolist()),
                                                     set(df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value].tolist()),
                                                     set(df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value].tolist()),
                                                     set(df["ageGroup"].tolist())):

            dfForGroup = df.loc[
                                (df["ageGroup"]==ageGroup) &
                                (df[StaticRiskFactorsType.GENDER.value]==ge) & 
                                (df[StaticRiskFactorsType.SMOKING_STATUS.value]==sm) &
                                (df[StaticRiskFactorsType.RACE_ETHNICITY.value]==ra) &
                                (df[DefaultTreatmentsType.STATIN.value]==st) &
                                (df[StaticRiskFactorsType.EDUCATION.value]==ed) &
                                (df[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value]==al) &
                                (df[DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value]==a) &
                                (df[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]==an), :].copy()
            if dfForGroup.shape[0]>0:
                dfForGroups[ge, sm, ra, st, ed, al, a, an, ageGroup] = dfForGroup 
        return dfForGroups

    def get_draws_from_distributions_crude(row, distributions):
        '''Uses categorical variable information to access the distribution of the continuous variables for that particular set of categoricals
        and then makes a draw from that distribution.
        If the draw ends up corresponding to a point in space where no NHANES data exist, that is no person has that extreme value(s),
        then the draw happens again.
        This function is only using a few basic categorical variables, hence the label "crude".
        For most groups, gender, race ethnicity, education and age we can get a non-singular gaussian distribution but in the few cases
        where we have a singular distribution then we use just gender, race ethnicity and age.'''
        gender = row["gender"]
        raceEthnicity = row["raceEthnicity"]
        education = row["education"]
        age = row["age"]

        if age>80:
            age=80
    
        nhanesContinuousVariables = PopulationFactory.nhanes_variable_types[VariableType.CONTINUOUS.value].copy()
        nhanesContinuousVariables.remove(DynamicRiskFactorsType.AGE.value)

        distKey = (gender, raceEthnicity, education, age)
        distKey = distKey if distKey in distributions["mean"].keys() else (gender, raceEthnicity, age)
    
        distMean = distributions["mean"][distKey]
        distCov = distributions["cov"][distKey]
        dist = multivariate_normal(distMean, distCov, allow_singular=False)
        distMin = distributions["min"][distKey]
        distMax = distributions["max"][distKey]
    
        drawsNeeded = 1 #1 draw per row, since each row represents one person
        size=drawsNeeded
        draws = None
        drawsForGroups = dict()
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
            if drawsNeeded==1:
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
        return draws







            
















                       












