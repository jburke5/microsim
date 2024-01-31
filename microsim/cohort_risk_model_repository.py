from microsim.risk_model_repository import RiskModelRepository
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.stats_model_linear_probability_risk_factor_model import (
    StatsModelLinearProbabilityRiskFactorModel,
)
from microsim.stats_model_rounded_linear_risk_factor_model import (
    StatsModelRoundedLinearRiskFactorModel,
)
from microsim.data_loader import load_regression_model
from microsim.alcohol_category import AlcoholCategory
from microsim.pvd_model import PVDIncidenceModel
from microsim.age_model import AgeModel
from microsim.afib_model import AFibIncidenceModel

class CohortRiskModelRepository(RiskModelRepository):
    def __init__(self):
        super(CohortRiskModelRepository, self).__init__()
        self._initialize_linear_risk_model("hdl", "hdlCohortModel")
        self._initialize_linear_risk_model("bmi", "bmiCohortModel")
        self._initialize_linear_risk_model("totChol", "totCholCohortModel")
        self._initialize_linear_risk_model("trig", "trigCohortModel")
        self._initialize_linear_risk_model("a1c", "a1cCohortModel")
        self._initialize_linear_risk_model("ldl", "ldlCohortModel")
        self._initialize_linear_risk_model("waist", "waistCohortModel")
        self._initialize_linear_probability_risk_model(
            "anyPhysicalActivity", "anyPhysicalActivityCohortModel"
        )
        self._repository["afib"] = AFibIncidenceModel()
        self._repository["pvd"] = PVDIncidenceModel()
        self._repository["age"] = AgeModel()
        self._initialize_linear_probability_risk_model("statin", "statinCohortModel")
        self._initialize_linear_risk_model("creatinine", "creatinineCohortModel")
        self._initialize_int_rounded_linear_risk_model(
            "antiHypertensiveCount", "antiHypertensiveCountCohortModel"
        )
        self._repository["alcoholPerWeek"] = AlcoholCategoryModel(
            load_regression_model("alcoholPerWeekCohortModel")
        )
        self._initialize_linear_risk_model("sbp", "logSbpCohortModel", log=True)
        self._initialize_linear_risk_model("dbp", "logDbpCohortModel", log=True)

    def _initialize_linear_risk_model(self, referenceName, modelName, log=False):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelLinearRiskFactorModel(model, log)

    def _initialize_linear_probability_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelLinearProbabilityRiskFactorModel(model)

    def _initialize_int_rounded_linear_risk_model(self, referenceName, modelName):
        model = load_regression_model(modelName)
        self._repository[referenceName] = StatsModelRoundedLinearRiskFactorModel(model)


class AlcoholCategoryModel(StatsModelRoundedLinearRiskFactorModel):
    def estimate_next_risk(self, person, rng=None):
        drinks = super(StatsModelRoundedLinearRiskFactorModel, self).estimate_next_risk(person)
        return AlcoholCategory.get_category_for_consumption(drinks if drinks > 0 else 0)

    def estimate_next_risk_vectorized(self, x, rng=None):
        drinks = super(StatsModelRoundedLinearRiskFactorModel, self).estimate_next_risk_vectorized(
            x
        )
        return AlcoholCategory.get_category_for_consumption(drinks if drinks > 0 else 0)


