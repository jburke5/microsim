from microsim.treatment_strategies.treatment_strategies import (
    TreatmentStrategyStatus,
    TreatmentStrategiesType,
)


class Wmd15TreatmentStrategy:
    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if (
            person._treatmentStrategies[TreatmentStrategiesType.WMD15.value]["status"]
            == TreatmentStrategyStatus.BEGIN
        ):
            person._treatmentStrategies[TreatmentStrategiesType.WMD15.value]["wmd15MedsAdded"] = 1
        elif (
            person._treatmentStrategies[TreatmentStrategiesType.WMD15.value]["status"]
            == TreatmentStrategyStatus.END
        ):
            del person._treatmentStrategies[TreatmentStrategiesType.WMD15.value]["wmd15MedsAdded"]
        return dict()


class Wmd20TreatmentStrategy:
    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if (
            person._treatmentStrategies[TreatmentStrategiesType.WMD20.value]["status"]
            == TreatmentStrategyStatus.BEGIN
        ):
            person._treatmentStrategies[TreatmentStrategiesType.WMD20.value]["wmd20MedsAdded"] = 1
        elif (
            person._treatmentStrategies[TreatmentStrategiesType.WMD20.value]["status"]
            == TreatmentStrategyStatus.END
        ):
            del person._treatmentStrategies[TreatmentStrategiesType.WMD20.value]["wmd20MedsAdded"]
        return dict()


class Wmd25TreatmentStrategy:
    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if (
            person._treatmentStrategies[TreatmentStrategiesType.WMD25.value]["status"]
            == TreatmentStrategyStatus.BEGIN
        ):
            person._treatmentStrategies[TreatmentStrategiesType.WMD25.value]["wmd25MedsAdded"] = 1
        elif (
            person._treatmentStrategies[TreatmentStrategiesType.WMD25.value]["status"]
            == TreatmentStrategyStatus.END
        ):
            del person._treatmentStrategies[TreatmentStrategiesType.WMD25.value]["wmd25MedsAdded"]
        return dict()
