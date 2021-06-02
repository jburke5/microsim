class RiskModelRepository:
    def __init__(self):
        self._repository = {}

    def get_model(self, name):
        return self._repository[name]
