from microsim.outcome import OutcomeType
from microsim.trials.relative_risk_analysis import RelativeRiskAnalysis
from microsim.trials.cox_regression_analysis import CoxRegressionAnalysis
from microsim.trials.linear_regression_analysis import LinearRegressionAnalysis
from microsim.trials.logistic_regression_analysis import LogisticRegressionAnalysis

class TrialOutcomeAssessor:
    '''This class will store the specific analyses that will be obtained from a Trial instance.
    This class provides a link between Population-level functions and methodologies used to analyze 
    the results when those Population-level functions are applied to the treated and control trial populations.
    _analysis: initializes classes that are needed in order to perform the analysis of the treated and control population outcomes
    _assessments: a dictionary, keys are the name of the assessments
                                values are dictionaries with two keys, assessmentFunctionDict and assessmentAnalysis
                  assessmentFunctionDict: a dictionary, two keys, outcome and time
                         outcome: a Population-level function that will return the outcome for each member of the population
                         time: a Population-level function that will return the time at which the outcome occured
                  assessmentAnalysis: a string, must be one of the keys of the _analysis dictionary (otherwise the class will not
                         know how to analyze the results.'''
    def __init__(self, addCommonAssessments=True):
        self._assessments = dict()
        self._analysis = {"linear": LinearRegressionAnalysis(),
                          "logistic": LogisticRegressionAnalysis(),
                          "cox": CoxRegressionAnalysis(),
                          "relRisk": RelativeRiskAnalysis()} 
        if addCommonAssessments:
            self.add_common_assessments()
        
    def add_outcome_assessment(self, assessmentName, assessmentFunctionDict, assessmentAnalysis):
        if assessmentAnalysis in self._analysis.keys():
            if assessmentName not in self._assessments.keys():
                if (((assessmentAnalysis!="cox") & (len(assessmentFunctionDict)==1)) | 
                    ((assessmentAnalysis=="cox") & (len(assessmentFunctionDict)==2))):
                    self._assessments[assessmentName] = {"assessmentFunctionDict": assessmentFunctionDict,
                                                         "assessmentAnalysis": assessmentAnalysis}
                else:
                    print(f"Cannot add outcome assessment {assessmentName} because of incorrect assessmentFunctionDict length.")
            else:
                print(f"Cannot add outcome assessment {assessmentName} because this assessment name already exists.")
        else:
            print(f"Cannot add outcome assessment with analysis {assessmentAnalysis} because this analysis does not exist.")
            print(f"Available assessment analysis are: {[analysis for analysis in self._analysis.keys()]}")
        
    def rm_outcome_assessment(self, assessmentName):
        if assessmentName in self._assessments.keys():
            del self._assessments[assessmentName]
        else:
            print(f"Cannot remove outcome assessment with name {assessmentName} because this assessment name does not exist.")
            
    def rm_outcome_assessments(self, assessmentNameList):
        for assessmentName in assessmentNameList:
            self.rm_outcome_assessment(assessmentName)
            
    def add_common_assessments(self):
        '''This function adds some trial outcome assessments that are likely to be interesting from a trial.
        It also serves as an example of how trial outcome assessments can be added.'''
        self.add_outcome_assessment("death", 
                                    {"outcome": lambda x: x.has_outcome(OutcomeType.DEATH)}, 
                                    "logistic")
        self.add_outcome_assessment("anyEvent", 
                                    {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE,
                                                                  OutcomeType.DEMENTIA, OutcomeType.CI])}, 
                                    "logistic")
        self.add_outcome_assessment("vascularEventOrDeath", 
                                    {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE])}, 
                                    "logistic")
        self.add_outcome_assessment("vascularEvent", 
                                    {"outcome": lambda x: x.has_any_outcome([OutcomeType.MI, OutcomeType.STROKE])}, 
                                    "logistic")
        self.add_outcome_assessment("qalys", 
                                    {"outcome": lambda x: x.get_outcome_item_sum(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")}, 
                                    "linear")
        self.add_outcome_assessment("meanGCP", 
                                    {"outcome": lambda x: x.get_outcome_item_mean(OutcomeType.COGNITION, "gcp")}, 
                                    "linear")
        self.add_outcome_assessment("lastGCP", 
                                    {"outcome": lambda x: x.get_outcome_item_last(OutcomeType.COGNITION, "gcp")}, 
                                    "linear")
        self.add_outcome_assessment("cogEvent", 
                                    {"outcome": lambda x: x.has_any_outcome([OutcomeType.CI, OutcomeType.DEMENTIA])}, 
                                    "logistic")
        self.add_outcome_assessment("deathCox", 
                                    {"outcome": lambda x: x.has_outcome(OutcomeType.DEATH),
                                     "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.DEATH])},
                                    "cox")
        self.add_outcome_assessment("cogEventCox", 
                                    {"outcome": lambda x: x.has_any_outcome([OutcomeType.CI, OutcomeType.DEMENTIA]),
                                     "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.CI, OutcomeType.DEMENTIA])},
                                    "cox")
        self.add_outcome_assessment("vascularEventOrDeathCox",
                                    {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE]),
                                     "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE])},
                                     "cox")
        self.add_outcome_assessment("strokeRR",
                                    {"outcome": lambda x: x.get_outcome_risk(OutcomeType.STROKE)},
                                    "relRisk")
        self.add_outcome_assessment("miRR",
                                    {"outcome": lambda x: x.get_outcome_risk(OutcomeType.MI)},
                                    "relRisk")
      
    def __str__(self):
        rep = f"Trial Outcome Assessor\n\tAssessments:\n"
        for assessmentName in self._assessments.keys():
            rep += f"\t\tName: {assessmentName:<25}" 
            #rep += f"Function: {self._assessments[assessmentName]['assessmentFunction']},"
            rep += f"Analysis: {self._assessments[assessmentName]['assessmentAnalysis']:<15}\n"
        return rep
    
    def __repr__(self):
        return self.__str__()
