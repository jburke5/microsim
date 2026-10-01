from microsim.trials.relative_risk_analysis import RelativeRiskAnalysis
from microsim.trials.cox_regression_analysis import CoxRegressionAnalysis
from microsim.trials.linear_regression_analysis import LinearRegressionAnalysis
from microsim.trials.logistic_regression_analysis import LogisticRegressionAnalysis
from microsim.trials.incidence_rate_analysis import IncidenceRateAnalysis

from enum import Enum

class AnalysisType(Enum):
    LINEAR = "linear"
    LOGISTIC = "logistic"
    COX = "cox"
    RELATIVE_RISK = "relativeRisk"
    INCIDENCE_RATE = "incidenceRate"

ANALYSIS_CLASSES = {AnalysisType.LINEAR.value: LinearRegressionAnalysis,
                    AnalysisType.LOGISTIC.value: LogisticRegressionAnalysis,
                    AnalysisType.COX.value: CoxRegressionAnalysis,
                    AnalysisType.RELATIVE_RISK.value: RelativeRiskAnalysis,
                    AnalysisType.INCIDENCE_RATE.value: IncidenceRateAnalysis}

class TrialOutcomeAssessor:
    '''This class will store the specific analyses that will be obtained from a Trial instance.
    This class provides a link between Population-level functions and methodologies used to analyze 
    the results when those Population-level functions are applied to the treated and control trial populations.
    _analysis: initializes classes that are needed in order to perform the analysis of the treated and control population outcomes
    _assessments: a dictionary, keys are the name of the assessments
                                values are dictionaries with two keys, assessmentFunctionDict and assessmentAnalysis
                  assessmentFunctionDict: a dictionary of Population-level functions, keys depend on the analysis
                         outcome: returns the outcome for each member of the population (all analyses except incidenceRate)
                         time: returns the time at which the outcome occured (cox only)
                         eventAndTime: returns (event, personYears) pairs (incidenceRate only)
                  assessmentAnalysis: a string, must be one of the keys of the _analysis dictionary (otherwise the class will not
                         know how to analyze the results.'''
    def __init__(self):
        self._assessments = dict()
        self._analysis = {k: cls() for k, cls in ANALYSIS_CLASSES.items()}

    def add_outcome_assessment(self, assessmentName, assessmentFunctionDict, assessmentAnalysis):
        if assessmentAnalysis not in self._analysis.keys():
            raise RuntimeError(f"Cannot add outcome assessment with analysis {assessmentAnalysis} because this analysis does not exist. "
                               f"Available assessment analyses are: {list(self._analysis.keys())}")
        if assessmentName in self._assessments.keys():
            raise RuntimeError(f"Cannot add outcome assessment {assessmentName} because this assessment name already exists.")
        if assessmentAnalysis == "cox":
            requiredKeys = {"outcome", "time"}
        elif assessmentAnalysis == "incidenceRate":
            requiredKeys = {"eventAndTime"}
        else:
            requiredKeys = {"outcome"}
        if set(assessmentFunctionDict.keys()) != requiredKeys:
            raise RuntimeError(f"Cannot add outcome assessment {assessmentName} because assessmentFunctionDict keys must be exactly {requiredKeys}.")
        self._assessments[assessmentName] = {"assessmentFunctionDict": assessmentFunctionDict,
                                             "assessmentAnalysis": assessmentAnalysis}

    def rm_outcome_assessment(self, assessmentName):
        if assessmentName not in self._assessments.keys():
            raise RuntimeError(f"Cannot remove outcome assessment with name {assessmentName} because this assessment name does not exist.")
        del self._assessments[assessmentName]
            
    def rm_outcome_assessments(self, assessmentNameList):
        for assessmentName in assessmentNameList:
            self.rm_outcome_assessment(assessmentName)
            
    def __str__(self):
        rep = f"Trial Outcome Assessor\n\tAssessments:\n"
        for assessmentName in self._assessments.keys():
            rep += f"\t\tName: {assessmentName:<25}" 
            #rep += f"Function: {self._assessments[assessmentName]['assessmentFunction']},"
            rep += f"Analysis: {self._assessments[assessmentName]['assessmentAnalysis']:<15}\n"
        return rep
    
    def __repr__(self):
        return self.__str__()
