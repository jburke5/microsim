from microsim.treatment import TreatmentStrategiesType

class TreatmentStrategyRepository:
    def __init__(self):
        self._repository = dict()
        for ts in TreatmentStrategiesType:
            self._repository[ts.value] = None
