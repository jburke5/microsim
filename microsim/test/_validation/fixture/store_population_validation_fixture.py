from types import MappingProxyType
from unittest import TestCase
import numpy as np
import pandas as pd
from microsim.bp_treatment_recalibration import BPTreatmentRecalibration
from microsim.bp_treatment_strategies import AddASingleBPMedTreatmentStrategy
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.data_loader import load_regression_model
from microsim.nhanes_person_record_loader import (
    NHANESPersonRecordLoader,
    BPCOGCohortPersonRecordFactory,
)
from microsim.outcome import OutcomeType
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome_model_type import OutcomeModelType
from microsim.person.bpcog_person_records import (
    BPCOGPersonStaticRecordProtocol,
    BPCOGPersonDynamicRecordProtocol,
)
from microsim.population.population import Population
from microsim.population.store_population import StorePopulation
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel
from microsim.store.bpcog_numpy_person_proxy import new_bpcog_person_proxy_class
from microsim.store.numpy_record_mapping import (
    NumpyRecordMapping,
    NumpyEventRecordMapping,
)
from microsim.store.numpy_person_store import NumpyPersonStore
from microsim.test._validation.helper import person_obj_from_person_record


def get_init_afib():
    model = load_regression_model("BaselineAFibModel")
    afib_model = StatsModelLogisticRiskFactorModel(model)

    def init_afib(person_record):
        return afib_model.estimate_next_risk_vectorized(person_record)

    return init_afib


def get_init_gcp(outcome_model_repository):
    gcp_model = outcome_model_repository.select_model_for_gender(
        None, OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE
    )

    def init_gcp(person_record):
        return gcp_model.calc_linear_predictor_for_patient_characteristics(
            years_in_simulation=0,
            raceEthnicity=person_record.raceEthnicity,
            gender=person_record.gender,
            baseAge=person_record.age,
            education=person_record.education,
            smokingStatus=person_record.smokingStatus,
            bmi=person_record.bmi,
            waist=person_record.waist,
            totChol=person_record.totChol,
            meanSbp=person_record.sbp,
            afib=person_record.afib,
            anyPhysicalActivity=person_record.anyPhysicalActivity,
            alc=person_record.alcoholPerWeek,
            antiHypertensiveCount=person_record.antiHypertensiveCount,
            a1c=person_record.a1c,
        )

    return init_gcp


def get_init_qalys(qaly_assignment_strategy):
    def init_qalys(person_record):
        current_age = person_record.age
        conditions = {
            OutcomeType.DEMENTIA: (person_record.dementia, current_age),
            OutcomeType.STROKE: (person_record.stroke, current_age),
            OutcomeType.MI: (person_record.mi, current_age),
        }
        has_died = person_record.alive
        return qaly_assignment_strategy.get_qalys_for_age_and_conditions(
            current_age, conditions, has_died
        )

    return init_qalys


class StorePopulationValidationFixture(TestCase):
    _person_records = None

    @classmethod
    def get_or_init_person_records(
        cls,
        num_persons=1_000,
        nhanes_year=2013,
        random_seed=3334670448,
    ):
        """Returns existing or creates, sets, & returns person record list."""
        if cls._person_records is not None:
            return cls._person_records

        factory_seed = np.random.RandomState(random_seed).randint(2 ** 32 - 1)
        factory = BPCOGCohortPersonRecordFactory(seed=factory_seed)
        loader = NHANESPersonRecordLoader(num_persons, nhanes_year, factory, seed=random_seed)
        cls._person_records = list(loader)
        return cls._person_records

    def _new_store_pop(self, person_records, num_years, combined_record_mapping):
        """Returns a new store population"""
        person_proxy_class = new_bpcog_person_proxy_class(
            {c: m.property_mappings for c, m in combined_record_mapping.items()}
        )
        person_store = NumpyPersonStore(
            len(person_records),
            num_years,
            combined_record_mapping["static"],
            combined_record_mapping["dynamic"],
            combined_record_mapping["event"],
            person_records,
            person_proxy_class=person_proxy_class,
        )

        rf_prop_names = [
            "sbp",
            "dbp",
            "a1c",
            "hdl",
            "ldl",
            "trig",
            "totChol",
            "bmi",
            "anyPhysicalActivity",
            "afib",
            "waist",
            "alcoholPerWeek",
        ]
        treatment_prop_names = ["antiHypertensiveCount", "statin"]
        outcome_prop_names = ["stroke", "mi", "dementia", "gcp"]
        store_pop = StorePopulation(
            person_store,
            self.risk_model_repository,
            self.outcome_model_repository,
            self.bp_treatment_strategy,
            self.bp_treatment_recalibration,
            self.qaly_assignment_strategy,
            rf_prop_names,
            treatment_prop_names,
            outcome_prop_names,
        )
        return store_pop

    def _new_vec_pop(self, person_records):
        people = pd.Series(
            [person_obj_from_person_record(i, r) for i, r in enumerate(person_records)]
        )
        population = Population(people)
        population._risk_model_repository = self.risk_model_repository
        population._outcome_model_repository = self.outcome_model_repository
        population._qaly_assignment_strategy = self.qaly_assignment_strategy
        population._bpTreatmentStrategy = self.bp_treatment_strategy
        return population

    def setUp(self):
        # call class method directly to set person records on (& to return them from) one place
        self.initial_person_records = StorePopulationValidationFixture.get_or_init_person_records()

        # setup population dependencies. Use overrides if present; else, use defaults
        if not hasattr(self, "risk_model_repository"):
            self.risk_model_repository = CohortRiskModelRepository()
        if not hasattr(self, "outcome_model_repository"):
            self.outcome_model_repository = OutcomeModelRepository()
        if not hasattr(self, "bp_treatment_strategy"):
            self.bp_treatment_strategy = AddASingleBPMedTreatmentStrategy()
        if not hasattr(self, "bp_treatment_recalibration"):
            self.bp_treatment_recalibration = BPTreatmentRecalibration(
                self.bp_treatment_strategy, self.outcome_model_repository
            )
        if not hasattr(self, "qaly_assignment_strategy"):
            self.qaly_assignment_strategy = QALYAssignmentStrategy()
        self.num_years = getattr(self, "num_years", 10)

        # finally, actually create the populations
        self.combined_record_mapping = MappingProxyType(
            {
                "static": NumpyRecordMapping(BPCOGPersonStaticRecordProtocol),
                "dynamic": NumpyRecordMapping(BPCOGPersonDynamicRecordProtocol),
                "event": NumpyEventRecordMapping(),
            }
        )
        self.store_pop = self._new_store_pop(
            self.initial_person_records, self.num_years, self.combined_record_mapping
        )
        self.vec_pop = self._new_vec_pop(self.initial_person_records)
