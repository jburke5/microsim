from bdb import BdbQuit
import dataclasses
import math
from types import MappingProxyType
import numpy as np
import pandas as pd
from microsim.bp_treatment_recalibration import BPTreatmentRecalibration
from microsim.bp_treatment_strategies import AddASingleBPMedTreatmentStrategy
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.person.bpcog_person_records import (
    BPCOGPersonRecord,
    BPCOGPersonStaticRecordProtocol,
    BPCOGPersonDynamicRecordProtocol,
)
from microsim.person.person import Person
from microsim.population.population import Population
from microsim.population.store_population import StorePopulation
from microsim.store.numpy_record_mapping import (
    NumpyRecordMapping,
    NumpyEventRecordMapping,
)
from microsim.test._validation.helper import BPCOGCohortPersonRecordLoader
from microsim.store.numpy_person_store import NumpyPersonStore
from microsim.store.bpcog_numpy_person_proxy import new_bpcog_person_proxy_class
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy


def is_alive(person):
    return person.current.alive


def assert_data_integrity(num_persons, person_records, store, initial_pop):
    assert num_persons == store.get_num_persons()

    # everyone should be alive
    assert num_persons == initial_pop.num_persons
    expected_member_indices = np.arange(0, num_persons)
    assert np.all(expected_member_indices == initial_pop.member_indices)

    # values should be roughly equivalent
    prop_names = [f.name for f in dataclasses.fields(BPCOGPersonRecord)]
    for i, (original_record, person) in enumerate(zip(person_records, initial_pop)):
        for name in prop_names:
            proxied_record = person.current
            original_value = getattr(original_record, name)
            proxied_value = getattr(proxied_record, name)
            assert original_value == proxied_value or math.isclose(
                original_value, proxied_value, abs_tol=0.99
            )

    # `.next` should be unset/empty...
    expected_prop_values = {
        "alive": False,
        "age": 0,
        "sbp": 0,
        "dbp": 0,
        "a1c": 0,
        "hdl": 0,
        "ldl": 0,
        "trig": 0,
        "totChol": 0,
        "bmi": 0,
        "waist": 0,
        "anyPhysicalActivity": False,
        "alcoholPerWeek": 0,
        "antiHypertensiveCount": 0,
        "statin": 0,
        "bpMedsAdded": 0,
        "afib": False,
        "qalys": 0,
        "gcp": 0,
        "mi": None,
        "stroke": None,
        "dementia": None,
    }
    for i, person in enumerate(initial_pop):
        # ...except for static props, which should be equal initial values
        proxied_record = person.current
        next_proxied_record = person.next
        assert proxied_record.gender == next_proxied_record.gender
        assert proxied_record.raceEthnicity == next_proxied_record.raceEthnicity
        assert proxied_record.education == next_proxied_record.education
        assert proxied_record.smokingStatus == next_proxied_record.smokingStatus
        assert proxied_record.gcpRandomEffect == next_proxied_record.gcpRandomEffect
        assert (
            proxied_record.otherLipidLowerMedication
            == next_proxied_record.otherLipidLowerMedication
        )

        for name, expected_value in expected_prop_values.items():
            proxied_value = getattr(next_proxied_record, name)
            assert expected_value == proxied_value


def new_bpcog_person_store(num_persons, num_years, loader):
    static_mapping = NumpyRecordMapping(BPCOGPersonStaticRecordProtocol)
    dynamic_mapping = NumpyRecordMapping(BPCOGPersonDynamicRecordProtocol)
    event_mapping = NumpyEventRecordMapping()
    combined_prop_mappings = MappingProxyType(
        {
            "static": static_mapping.property_mappings,
            "dynamic": dynamic_mapping.property_mappings,
            "event": event_mapping.property_mappings,
        }
    )
    person_proxy_class = new_bpcog_person_proxy_class(combined_prop_mappings)

    person_store = NumpyPersonStore(
        num_persons,
        num_years,
        static_mapping,
        dynamic_mapping,
        event_mapping,
        loader,
        person_proxy_class=person_proxy_class,
    )

    return person_store


def person_record_to_person(population_index, person_record):
    person = Person(
        age=person_record.age,
        gender=person_record.gender,
        raceEthnicity=person_record.raceEthnicity,
        sbp=person_record.sbp,
        dbp=person_record.dbp,
        a1c=person_record.a1c,
        hdl=person_record.hdl,
        totChol=person_record.totChol,
        bmi=person_record.bmi,
        ldl=person_record.ldl,
        trig=person_record.trig,
        waist=person_record.waist,
        anyPhysicalActivity=person_record.anyPhysicalActivity,
        education=person_record.education,
        smokingStatus=person_record.smokingStatus,
        alcohol=person_record.alcoholPerWeek,
        antiHypertensiveCount=person_record.antiHypertensiveCount,
        statin=person_record.statin,
        otherLipidLoweringMedicationCount=person_record.otherLipidLowerMedication,
        initializeAfib=lambda _: person_record.afib,
        randomEffects={"gcp": person_record.gcpRandomEffect},
        _populationIndex=population_index,
    )
    person._qalys.append(person_record.qalys)
    person._gcp.append(person_record.gcp)
    return person


def new_vectorized_population(
    person_records,
    risk_model_repository,
    outcome_model_repository,
    qaly_assignment_strategy,
    bp_treatment_strategy,
):
    people = pd.Series([person_record_to_person(i, r) for i, r in enumerate(person_records)])
    population = Population(people)
    population._risk_model_repository = risk_model_repository
    population._outcome_model_repository = outcome_model_repository
    population._qaly_assignment_strategy = qaly_assignment_strategy
    population._bpTreatmentStrategy = bp_treatment_strategy
    return population


def init_populations(num_persons, num_years, nhanes_year, seed=None):
    loader = BPCOGCohortPersonRecordLoader(num_persons, nhanes_year, seed=seed)
    person_records = list(loader)
    person_store = new_bpcog_person_store(num_persons, num_years, person_records)
    initial_pop = person_store.get_population_at(0).where(is_alive)
    assert_data_integrity(num_persons, person_records, person_store, initial_pop)

    risk_model_repository = CohortRiskModelRepository()
    outcome_model_repository = OutcomeModelRepository()
    bp_treatment_strategy = AddASingleBPMedTreatmentStrategy()
    bp_treatment_recalibration = BPTreatmentRecalibration(
        bp_treatment_strategy, outcome_model_repository
    )
    qaly_assignment_strategy = QALYAssignmentStrategy()
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
        risk_model_repository,
        outcome_model_repository,
        bp_treatment_strategy,
        bp_treatment_recalibration,
        qaly_assignment_strategy,
        rf_prop_names,
        treatment_prop_names,
        outcome_prop_names,
    )

    vec_pop = new_vectorized_population(
        person_records,
        risk_model_repository,
        outcome_model_repository,
        qaly_assignment_strategy,
        bp_treatment_strategy,
    )

    return store_pop, vec_pop


def main():
    num_persons = 1_000
    num_years = 10
    year = 2013
    seed = None

    store_pop, vec_pop = init_populations(num_persons, num_years, year, seed)

    store_pop.advance()
    vec_pop.advance_vectorized(1)


if __name__ == "__main__":
    try:
        main()
    except BdbQuit:
        pass
    except Exception:
        import pdb, sys, traceback

        err_type, err, tb = sys.exc_info()
        traceback.print_exception(err_type, err, tb)
        pdb.post_mortem(tb)
