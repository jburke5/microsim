import pandas as pd

from microsim.person.person_filter import PersonFilter
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.outcomes.cv_model_repository import CVModelRepository

#shared by the highCVLimit filter so the CV models are not rebuilt for every person tested
_cvModelRepository = CVModelRepository()

class PersonFilterFactory:
    '''Factory that builds a PersonFilter, the object used to include/exclude individuals from a Population.

    A PersonFilter holds two collections of named, boolean-returning functions (see PersonFilter):
        filters["df"]     functions applied to a dataframe ROW before any Person object is built
        filters["person"] functions applied to a Person OBJECT after it has been built
    In both cases a person/row is KEPT when the function returns True and DROPPED when it returns False.
    The two levels exist for efficiency: building Person objects costs memory and time, so cheap
    criteria that can be read straight off the source data ("df" filters) are applied first, and only
    criteria that need a fully constructed Person ("person" filters, e.g. anything that runs a risk
    model) are applied afterward.

    How a PersonFilter is consumed
    ------------------------------
    A PersonFilter is not applied by itself; it is passed to a PopulationFactory method as the
    personFilters= argument, which then applies the two levels in order (see
    PopulationFactory.apply_person_filters_on_df / apply_person_filters_on_people):

        pf = PersonFilterFactory.get_person_filter()
        pop = PopulationFactory.get_nhanes_population(n=1000, year=2007, personFilters=pf,
                                                      nhanesWeights=True, distributions=False)

    The same pf can be handed to a trial via the personFilters= argument of the trial description /
    trial factory, where it defines trial eligibility (inclusion/exclusion criteria).

    The named-filter registry
    --------------------------
    filterMap is a class-level registry that maps a short string key to a (filterType, filterFunction)
    pair, so that pre-defined filters can be requested by name without writing any lambdas. The
    available keys are:

        "adult"                      df     age >= 18
        "lowSBPLimit"                df     SBP > 126
        "lowDBPLimit"                df     DBP > 85
        "highAntiHypertensivesLimit" df     antiHypertensiveCount <= 3
        "highCVLimit"                person one-year CV risk < 0.00477
        "noMCI"                      person no MCI at baseline
        "hasEpilepsy"                person has epilepsy at baseline
        "usBornOrInUs15PlusYears"    df     US-born, or foreign-born in the US 15+ years

    Pass any subset of these keys to get_person_filter to obtain a PersonFilter holding
    exactly those filters (each is added under its own key as its filter name):

        pf = PersonFilterFactory.get_person_filter(["lowSBPLimit", "highCVLimit"])

    get_person_filter() defaults filterNames to ["adult"], so calling it with no arguments returns
    the default PersonFilter holding only the "adult" filter.

    Setting up your own PersonFilter
    --------------------------------
    Start from the default adult filter and add your own:

        pf = PersonFilterFactory.get_person_filter()                  # adult (age >= 18) filter only
        pf.add_filter("df", "under80", lambda x: x[DynamicRiskFactorsType.AGE.value] < 80)

    or start empty (pass an empty list) and register every filter yourself:

        pf = PersonFilterFactory.get_person_filter([])
        pf.add_filter("df", "men", lambda x: x[StaticRiskFactorsType.GENDER.value] == NHANESGender.MALE.value)
        pf.add_filter("person", "noPriorStroke", lambda x: not x.has_stroke_prior_to_simulation())
        pf.rm_filter("df", "men")                                     # remove a filter by name if needed

    Guidance for writing filters
    ----------------------------
    - filterType is "df" or "person"; filterName is any string unique within that level (re-using a
      name overwrites the earlier filter).
    - A "df" filter receives a single dataframe row (a pandas Series) and must return True/False.
      Index it with the source-column key, most conveniently the *.value of a risk-factor or treatment
      enum (e.g. DynamicRiskFactorsType.SBP.value, DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value).
    - A "person" filter receives a Person object and must return True/False; use it when the criterion
      needs a method or model that only a built Person exposes (e.g. person.has_mci(inSim=False) or a
      risk model from an outcome repository). Pass inSim=False on Person query methods so the filter
      reflects baseline state rather than simulated events.
    - Prefer "df" filters whenever the criterion can be read from the raw data; reserve "person"
      filters for model-based criteria so Person objects are not built for rows that will be rejected.
    '''

    # Registry of pre-defined filters: string key -> (filterType, filterFunction).
    # Add an entry here to make a new filter requestable by name via get_person_filter.
    filterMap = {
        "adult": ("df", lambda x: x[DynamicRiskFactorsType.AGE.value]>=18),
        "lowSBPLimit": ("df", lambda x: x[DynamicRiskFactorsType.SBP.value]>126),
        "lowDBPLimit": ("df", lambda x: x[DynamicRiskFactorsType.DBP.value]>85),
        "highAntiHypertensivesLimit": ("df", lambda x: x[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]<=3),
        "highCVLimit": ("person",
            lambda x: (_cvModelRepository.select_outcome_model_for_person(x).get_risk_for_person(x)< (0.00477) )),
        "noMCI": ("person",
            lambda x: not x.has_mci(inSim=False)),
        "hasEpilepsy": ("person",
            lambda x: x.has_epilepsy()),
        #timeInUS is NHANES DMDYRSUS: duration bands asked only of the foreign-born (5 = 15-<20
        #years, the band holding an 18-year lookback; 77/99 = refused/don't know), NaN for the
        #US-born - who must be kept, NaN >= threshold silently drops them all
        "usBornOrInUs15PlusYears": ("df",
            lambda x: pd.isna(x["timeInUS"]) or (5 <= x["timeInUS"] <= 9)),
    }

    @staticmethod
    def get_person_filter(filterNames=None):
        '''Return a PersonFilter holding the pre-defined filters named in filterNames.

        filterNames is a list of keys into filterMap; each named filter is added to the returned
        PersonFilter at its registered level ("df" or "person") under its key as the filter name.
        An unknown key raises ValueError listing the available keys.

        filterNames defaults to ["adult"] (the age >= 18 df filter), so get_person_filter() with no
        arguments returns the default adult-only PersonFilter. Pass an explicit list to choose a
        different set, or [] for a PersonFilter with no filters:

            pf = PersonFilterFactory.get_person_filter()                          # adult (age >= 18) only
            pf = PersonFilterFactory.get_person_filter(["lowSBPLimit", "highCVLimit"])
            pf = PersonFilterFactory.get_person_filter([])                         # no filters
        '''
        if filterNames is None:
            filterNames = ["adult"]
        pf = PersonFilter()
        for filterName in filterNames:
            if filterName not in PersonFilterFactory.filterMap:
                raise ValueError(f"Unknown person filter '{filterName}'. "
                                 f"Available filters: {sorted(PersonFilterFactory.filterMap.keys())}")
            filterType, filterFunction = PersonFilterFactory.filterMap[filterName]
            pf.add_filter(filterType, filterName, filterFunction)
        return pf
