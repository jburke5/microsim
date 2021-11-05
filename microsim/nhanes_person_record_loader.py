import dataclasses
import numpy as np
import pandas as pd
from microsim.alcohol_category import AlcoholCategory
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome_model_type import OutcomeModelType
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.smoking_status import SmokingStatus
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.data_loader import get_absolute_datafile_path, load_regression_model
from microsim.outcome import Outcome, OutcomeType
from microsim.person.bpcog_person_records import (
    BPCOGPersonRecord,
    BPCOGPersonDynamicRecord,
    BPCOGPersonEventRecord,
    BPCOGPersonStaticRecord,
)
from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel


def get_nhanes_imputed_dataset(year):
    dataset_path = get_absolute_datafile_path("fullyImputedDataset.dta")
    dataset = pd.read_stata(dataset_path)
    dataset = dataset.loc[dataset.year == year]
    return dataset


def build_prior_mi_event(prior_mi_age, current_age):
    if np.isnan(prior_mi_age) or prior_mi_age is None or prior_mi_age <= 1:
        return None
    if prior_mi_age == 99999:
        prior_mi_age = np.random.randint(18, current_age)
    prior_mi_age = prior_mi_age if prior_mi_age <= current_age else current_age
    return Outcome(OutcomeType.MI, False, age=prior_mi_age)


def build_prior_stroke_event(prior_stroke_age, current_age):
    if np.isnan(prior_stroke_age) or prior_stroke_age is None or prior_stroke_age <= 1:
        return None
    prior_stroke_age = prior_stroke_age if prior_stroke_age <= current_age else current_age
    return Outcome(OutcomeType.STROKE, False, age=prior_stroke_age)


class NHANESPersonRecordFactory:
    def __init__(self, init_random_effects, init_afib, init_gcp, init_qalys):
        self._init_random_effects = init_random_effects
        self._init_afib = init_afib
        self._init_gcp = init_gcp
        self._init_qalys = init_qalys

    @property
    def required_nhanes_column_names(self):
        return [
            "gender",
            "raceEthnicity",
            "education",
            "smokingStatus",
            "selfReportMIAge",
            "selfReportStrokeAge",
            "age",
            "meanSBP",
            "meanDBP",
            "a1c",
            "hdl",
            "ldl",
            "trig",
            "tot_chol",
            "bmi",
            "waist",
            "anyPhysicalActivity",
            "alcoholPerWeek",
            "antiHypertensive",
            "statin",
            "otherLipidLowering",
        ]

    def from_nhanes_dataset_row(
        self,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        selfReportMIAge,
        selfReportStrokeAge,
        age,
        meanSBP,
        meanDBP,
        a1c,
        hdl,
        ldl,
        trig,
        tot_chol,
        bmi,
        waist,
        anyPhysicalActivity,
        alcoholPerWeek,
        antiHypertensive,
        statin,
        otherLipidLowering,
    ):
        random_effects = self._init_random_effects()
        prior_mi = build_prior_mi_event(selfReportMIAge, age)
        prior_stroke = build_prior_stroke_event(selfReportStrokeAge, age)
        person_record = BPCOGPersonRecord(
            gender=NHANESGender(int(gender)),
            raceEthnicity=NHANESRaceEthnicity(int(raceEthnicity)),
            education=Education(int(education)),
            smokingStatus=SmokingStatus(int(smokingStatus)),
            gcpRandomEffect=random_effects["gcp"],
            alive=True,
            age=age,
            sbp=meanSBP,
            dbp=meanDBP,
            a1c=a1c,
            hdl=hdl,
            ldl=ldl,
            trig=trig,
            totChol=tot_chol,
            bmi=bmi,
            waist=waist,
            anyPhysicalActivity=anyPhysicalActivity,
            alcoholPerWeek=AlcoholCategory.get_category_for_consumption(alcoholPerWeek),
            antiHypertensiveCount=antiHypertensive,
            statin=bool(statin),
            otherLipidLowerMedication=otherLipidLowering,
            bpMedsAdded=0,
            afib=False,
            qalys=0,
            gcp=0,
            mi=prior_mi,
            stroke=prior_stroke,
            dementia=None,
        )

        afib = self._init_afib(person_record)
        gcp = self._init_gcp(person_record)
        person_record = dataclasses.replace(person_record, afib=afib, gcp=gcp)

        qalys = self._init_qalys(person_record)
        person_record = dataclasses.replace(person_record, qalys=qalys)

        return person_record


class NHANESPersonRecordLoader:
    """Loads PersonRecords from a sample of a given NHANES year."""

    def __init__(self, n, year, nhanes_person_record_factory, weights=None, seed=None):
        self._nhanes_dataset = get_nhanes_imputed_dataset(year)
        self._n = n
        self._weights = weights
        self._seed = seed if seed is not None else np.random.randint(2 ** 32 - 1)
        self._factory = nhanes_person_record_factory

    def __len__(self):
        return self._n

    def __iter__(self):
        sample = self._nhanes_dataset.sample(
            self._n,
            weights=self._weights,
            random_state=np.random.RandomState(self._seed),
            replace=True,
        )
        column_names = self._factory.required_nhanes_column_names
        for row_data in zip(*[sample[k] for k in column_names]):
            person_record = self._factory.from_nhanes_dataset_row(*row_data)
            yield person_record


class BPCOGCohortPersonRecordFactory(NHANESPersonRecordFactory):
    """Creates `BPCOGPersonRecord`s from NHANES data & BPCOG cohort models."""

    def __init__(
        self,
        risk_model_repository=None,
        outcome_model_repository=None,
        qaly_assignment_strategy=None,
        seed=None,
    ):
        super().__init__(
            self._init_random_effects, self._init_afib, self._init_gcp, self._init_qalys
        )

        self._risk_model_repository = risk_model_repository
        if self._risk_model_repository is None:
            self._risk_model_repository = CohortRiskModelRepository()

        self._outcome_model_repository = outcome_model_repository
        if self._outcome_model_repository is None:
            self._outcome_model_repository = OutcomeModelRepository()

        self._qaly_assignment_strategy = qaly_assignment_strategy
        if self._qaly_assignment_strategy is None:
            self._qaly_assignment_strategy = QALYAssignmentStrategy()

        self._afib_model = StatsModelLogisticRiskFactorModel(
            load_regression_model("BaselineAFibModel")
        )
        self._gcp_model = self._outcome_model_repository.select_model_for_gender(
            None, OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE
        )

        self._seed = seed
        self.reset_randstate()

    def reset_randstate(self):
        self._randstate = np.random.RandomState(self._seed).get_state()

    def _init_afib(self, person_record):
        prev_randstate = np.random.get_state()
        np.random.set_state(self._randstate)
        try:
            return self._afib_model.estimate_next_risk_vectorized(person_record)
        finally:
            np.random.set_state(prev_randstate)

    def _init_gcp(self, person_record):
        return self._gcp_model.calc_linear_predictor_for_patient_characteristics(
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

    def _init_qalys(self, person_record):
        current_age = person_record.age
        conditions = {
            OutcomeType.DEMENTIA: (person_record.dementia, current_age),
            OutcomeType.STROKE: (person_record.stroke, current_age),
            OutcomeType.MI: (person_record.mi, current_age),
        }
        has_died = person_record.alive
        return self._qaly_assignment_strategy.get_qalys_for_age_and_conditions(
            current_age, conditions, has_died
        )

    def _init_random_effects(self):
        prev_randstate = np.random.get_state()
        np.random.set_state(self._randstate)
        try:
            return self._outcome_model_repository.get_random_effects()
        finally:
            np.random.set_state(prev_randstate)
