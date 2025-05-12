
class RelativeRiskAnalysis:
    def __init__(self):
        pass
    
    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        assessmentFunction = assessmentFunctionDict["outcome"]
        treatedRisk = list(map(assessmentFunction, [trial.treatedPop]))[0] 
        controlRisk = list(map(assessmentFunction, [trial.controlPop]))[0]
        if controlRisk!=0.:
            outcomeRelativeRisk = treatedRisk/controlRisk
            return outcomeRelativeRisk, treatedRisk, controlRisk, abs(treatedRisk-controlRisk)*1000.
        else:
            return float('inf'), treatedRisk, controlRisk, abs(treatedRisk-controlRisk)*1000.
