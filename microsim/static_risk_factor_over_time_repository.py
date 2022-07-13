from microsim.risk_model_repository import RiskModelRepository


class StaticRiskFactorOverTimeRepository(RiskModelRepository):
    def get_model(self, name):
        return DoNothingModel(name)


class DoNothingModel:
    def __init__(self, name):
        self.name = name

    def estimate_next_risk(self, person):
        return getattr(person, f"_{self.name}")[-1]
    
    def estimate_next_risk_vectorized(self, x):
        return x[self.name]

