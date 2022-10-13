from microsim.outcome_model_repository import OutcomeModelRepository

class RiskFilter:
    # outcomeRiskThresholds takes a dict where hte key is an outcome model type and the value is a float representing a given risk threshold

    def __init__(self, outcomeRiskThresholds):
        self._outcomeRiskThresholds = outcomeRiskThresholds
    
    def exceedsThresholds(self, person):
        exceedsThresolds = False
        for outcomeModelType, threshold in self._outcomeRiskThresholds.items():
            risk = OutcomeModelRepository().get_risk_for_person(person, outcomeModelType)
            if risk > threshold:
                exceedsThresolds = True
                break

        return exceedsThresolds

