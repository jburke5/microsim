from microsim.risk_model_repository import RiskModelRepository


class DoNotChangeRiskFactorModel:
    def __init__(self, varName):
        self.varName = varName

    def estimate_next_risk(self, person, rng=None):
        return getattr(person, "_" + self.varName)[-1]

    def estimate_next_risk_vectorized(self, person, rng=None):
        return getattr(person, self.varName)


class DoNotChangeRiskFactorsModelRepository(RiskModelRepository):
    def __init__(self):
        super(DoNotChangeRiskFactorsModelRepository, self).__init__()

    def get_model(self, name):
        return DoNotChangeRiskFactorModel(name)
