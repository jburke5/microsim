from microsim.cv_model_repository import CVModelRepository
from microsim.treatment import TreatmentStrategyStatus, TreatmentStrategiesType, DefaultTreatmentsType

class StatinTreatmentStrategy():
    '''This class does not directly modify any risk factors or default treatments.
    It does indirectly modify the CV event risk.
    The statinsAdded variable affects the CV risk in cv_model.py.
    Also, currently, the variable statin in default treatments is not affected in any way by statinsAdded.
    But statinsAdded is affected by the presence of statin (see below).'''
    def __init__(self, cvRiskCutoff=0.075, wmhSpecific=True):
        self.status = TreatmentStrategyStatus.BEGIN
        self.wmhSpecific = wmhSpecific
        self.cvModelRepository = CVModelRepository(wmhSpecific=self.wmhSpecific)
        if self.is_valid_risk(cvRiskCutoff):
            self.cvRiskCutoff = cvRiskCutoff
        else:
            raise RuntimeError(f"Cannot create StatinTreatmentStrategy with invalid risk cutoff {cvRiskCutoff}. Risk must not be <0 or >1.")

    def get_updated_treatments(self, person):
        if person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["status"]==TreatmentStrategyStatus.BEGIN:
            cvRisk = self.cvModelRepository.select_outcome_model_for_person(person).get_risk_for_person(person, years=10)
            statin = person.get_last_default_treatment(DefaultTreatmentsType.STATIN.value)
            if (cvRisk>self.cvRiskCutoff) & (not statin):
                person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]=1
            else:
                person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]=0
        elif person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["status"]==TreatmentStrategyStatus.MAINTAIN:
            if person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]==0:
                cvRisk = self.cvModelRepository.select_outcome_model_for_person(person).get_risk_for_person(person, years=10)
                statin = person.get_last_default_treatment(DefaultTreatmentsType.STATIN.value)
                if (cvRisk>self.cvRiskCutoff) & (not statin):
                    person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]=1
                else:
                    pass
        elif person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["status"]==TreatmentStrategyStatus.END:
            if "statinsAdded" in person._treatmentStrategies[TreatmentStrategiesType.STATIN.value].keys():
                del person._treatmentStrategies[TreatmentStrategiesType.STATIN.value]["statinsAdded"]
        return dict()

    def get_updated_risk_factors(self, person):
        return dict() #will not modify any risk factors at this time

    def is_valid_risk(self, risk):
        return False if (risk<0.) | (risk>1.) else True
