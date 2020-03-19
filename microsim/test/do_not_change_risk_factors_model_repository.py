from microsim.risk_model_repository import RiskModelRepository


class DoNotChangeRiskFactorModel:
    def __init__(self, varName):
        self.varName = varName

    def estimate_next_risk(self, person):
        return getattr(person, "_"+self.varName)[-1]


class DoNotChangeRiskFactorsModelRepository(RiskModelRepository):
    def __init__(self):
        super(DoNotChangeRiskFactorsModelRepository, self).__init__()

    def get_model(self, name):
        return DoNotChangeRiskFactorModel(name)
