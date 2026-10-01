from microsim.risk_factors.risk_model_repository import RiskModelRepository
from microsim.default_treatments.default_treatments import DefaultTreatmentsType


class DefaultTreatmentModelRepository(RiskModelRepository):
    def __init__(self):
        super().__init__()
        self._initialize_linear_probability_risk_model(
            DefaultTreatmentsType.STATIN.value, "statinCohortModel"
        )
        self._initialize_int_rounded_linear_risk_model(
            DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value, "antiHypertensiveCountCohortModel"
        )
