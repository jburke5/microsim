from microsim.risk_model_repository import RiskModelRepository
from microsim.cohort_risk_model_repository import CohortRiskModelRepository

# we'll let anti hypertensives get updated as per normal...
class StaticRiskFactorOverTimeRepository(RiskModelRepository):
    def __init__(self):
        super(StaticRiskFactorOverTimeRepository, self).__init__()
        self._repository["antiHypertensiveCount"] = CohortRiskModelRepository()._repository["antiHypertensiveCount"]


    def get_model(self, name):
        if name in self._repository.keys():
            return self._repository[name]
        else:
            return DoNothingModel(name)


class DoNothingModel:
    def __init__(self, name):
        self.name = name

    def estimate_next_risk(self, person, rng=None):
        return getattr(person, f"_{self.name}")[-1]
    
    def estimate_next_risk_vectorized(self, x, rng=None):
        return x[self.name]

