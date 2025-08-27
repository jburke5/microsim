from microsim.cv_model_repository import CVModelRepository
from microsim.treatment import TreatmentStrategyStatus, TreatmentStrategiesType

class StatinTreatmentStrategy():
    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN
        self.cvRiskCutoff = 0.075

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["status"]==TreatmentStrategyStatus.BEGIN:
            cvRisk = CVModelRepository().select_outcome_model_for_person(person).get_risk_for_person(person, years=10)
            if cvRisk > self.cvRiskCutoff:
                person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]=1
            else:
                person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]=0
        elif person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["status"]==TreatmentStrategyStatus.MAINTAIN:
            #if "statinsAdded" not in person._treatmentStrategies[TreatmentStrategiesType.STATIN.value].keys():
            if person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]==0:
                cvRisk = CVModelRepository().select_outcome_model_for_person(person).get_risk_for_person(person, years=10)
                if cvRisk > self.cvRiskCutoff:
                    person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]=1
                else:
                    pass
        elif person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["status"]==TreatmentStrategyStatus.END:
            if "statinsAdded" in person._treatmentStrategies[TreatmentStrategiesType.STATIN.value].keys():
                del person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]
        return dict()
