from microsim.risk_model_repository import RiskModelRepository
#from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.treatment import DefaultTreatmentsType
from microsim.age_model import AgeModel

# we'll let anti hypertensives get updated as per normal...
class StaticRiskFactorOverTimeRepository(RiskModelRepository):
    def __init__(self):
        super().__init__()
        #self._repository["antiHypertensiveCount"] = CohortRiskModelRepository()._repository["antiHypertensiveCount"]
        self._repository["age"] = AgeModel()

    def get_model(self, name):
        if name in self._repository.keys():
            return self._repository[name]
        else:
            return DoNothingModel(name)

class StaticDefaultTreatmentModelRepository(RiskModelRepository):
    def __init__(self):
        super().__init__()
        self._initialize_int_rounded_linear_risk_model(DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value, "antiHypertensiveCountCohortModel")

    def get_model(self, name):
        if name in self._repository.keys():
            return self._repository[name]
        else:
            return DoNothingModel(name)

class DoNothingModel:
    def __init__(self, name):
        self.name = name

    def estimate_next_risk(self, person):
        return getattr(person, f"_{self.name}")[-1]
    
