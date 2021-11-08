import numpy as np
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.data_loader import load_regression_model
from microsim.nhanes_person_record_loader import (
    NHANESPersonRecordFactory,
    NHANESPersonRecordLoader,
)
from microsim.outcome import OutcomeType
from microsim.outcome_model_repository import OutcomeModelRepository, OutcomeModelType
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy
from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel


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
        prev_randstate = np.random.get_state()
        np.random.set_state(self._randstate)
        try:
            return super().from_nhanes_dataset_row(
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
            )
        finally:
            np.random.set_state(prev_randstate)

    def _init_afib(self, person_record):
        return self._afib_model.estimate_next_risk_vectorized(person_record)

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
        return self._outcome_model_repository.get_random_effects()


class BPCOGCohortPersonRecordLoader(NHANESPersonRecordLoader):
    def __init__(
        self,
        n,
        year,
        weights=None,
        seed=None,
        factory_seed=None,
        risk_model_repository=None,
        outcome_model_repository=None,
        qaly_assignment_strategy=None,
    ):
        seed = seed if seed is not None else np.random.randint(2 ** 32 - 1)
        if factory_seed is None:
            factory_seed = np.random.RandomState(seed).randint(2 ** 32 - 1)
        factory = BPCOGCohortPersonRecordFactory(
            risk_model_repository,
            outcome_model_repository,
            qaly_assignment_strategy,
            seed=factory_seed,
        )
        super().__init__(n, year, factory, weights, seed)

    def __iter__(self):
        self._factory.reset_randstate()
        yield from super().__iter__()
