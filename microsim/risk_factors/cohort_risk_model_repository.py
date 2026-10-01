from microsim.risk_factors.risk_model_repository import RiskModelRepository
from microsim.regression_models.rounded_linear_risk_factor_model import RoundedLinearRiskFactorModel
from microsim.common.data_loader import load_regression_model
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.pvd_model import PVDIncidenceModel
from microsim.risk_factors.age_model import AgeModel
from microsim.risk_factors.afib_model import AFibIncidenceModel
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType

class CohortStaticRiskFactorModelRepository:
    def __init__(self):
        self._repository = {StaticRiskFactorsType.RACE_ETHNICITY.value: None,
                            StaticRiskFactorsType.EDUCATION.value: None,
                            StaticRiskFactorsType.GENDER.value: None,
                            StaticRiskFactorsType.SMOKING_STATUS.value: None,
                            StaticRiskFactorsType.MODALITY.value: None}

class CohortDynamicRiskFactorModelRepository(RiskModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[DynamicRiskFactorsType.AFIB.value] = AFibIncidenceModel()
        self._repository[DynamicRiskFactorsType.PVD.value] = PVDIncidenceModel()
        self._repository[DynamicRiskFactorsType.AGE.value] = AgeModel()
        self._repository[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value] = AlcoholModel(
            load_regression_model("alcoholPerWeekCohortModel")
        )

        self._initialize_linear_risk_model(DynamicRiskFactorsType.HDL.value, "hdlCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.BMI.value, "bmiCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.TOT_CHOL.value, "totCholCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.TRIG.value, "trigCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.A1C.value, "a1cCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.LDL.value, "ldlCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.WAIST.value, "waistCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.CREATININE.value, "creatinineCohortModel")
        self._initialize_linear_risk_model(DynamicRiskFactorsType.SBP.value, "logSbpCohortModel", log=True)
        self._initialize_linear_risk_model(DynamicRiskFactorsType.DBP.value, "logDbpCohortModel", log=True)

        self._initialize_linear_probability_risk_model(DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value, "anyPhysicalActivityCohortModel")

class AlcoholModel(RoundedLinearRiskFactorModel):
    #returns drinks/week; deterministic like the other linear risk factors (residual not drawn)
    def estimate_next_risk(self, person):
        drinks = super(RoundedLinearRiskFactorModel, self).estimate_next_risk(person)
        return round(drinks) if drinks > 0 else 0


