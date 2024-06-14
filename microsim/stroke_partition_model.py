from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.regression_model import RegressionModel
from microsim.data_loader import load_model_spec
from microsim.outcome import OutcomeType, Outcome
from microsim.outcome_details.stroke_details import StrokeSubtypeModelRepository, StrokeNihssModel, StrokeTypeModel
from microsim.stroke_outcome import StrokeOutcome, StrokeSubtype, StrokeType, Localization
from microsim.treatment import TreatmentStrategiesType

import scipy.special as scipySpecial

# there are 2 approaches that can be taken with partition models
# 1: this model asks the CVModel to see if there was an cv outcome for this person, and that outcome is just used without being stored
#    on the Person-instance
# 2: the CVModel is asked before StrokePartitionModel is called and the Person-object stores the CV outcome, and the strokeModel
#.  is just checking on the person object to see if there was a cv outcome in the current year
#   We now use the 2nd approach, and we will double check that the outcomes are called in the correct order.

class StrokePartitionModel(StatsModelLinearRiskFactorModel):
    """Fatal stroke probability estimated from our meta-analysis of BASIC, NoMAS, GCNKSS, REGARDS."""

    def __init__(self, intercept=None):
        model_spec = load_model_spec("StrokeMIPartitionModel")
        #the assumption for the subclasses of this class is that the intercept with 0 bpMedsAdded is -2.3109730587083006
        if intercept is not None:
            model_spec["coefficients"]["Intercept"] = intercept
        super().__init__(RegressionModel(**model_spec))
        self._stroke_case_fatality = 0.15
        self._stroke_secondary_case_fatality = 0.15
        #this is the mean change of the intercept for each bpMedAdded
        #Simulations were performed to obtain the optimized intercept for each bpMedAdded (1,2,3,4)
        #Then from those 4 intercepts and the baseline intercept, I obtained the average change of the intercept for each bpMedAdded
        #The optimized intercepts for each bpMedAdded are as follows:
        #intercept = -2.4295668  #1bpMedAdded
        #intercept = -2.5334375  #2
        #intercept = -2.66499999 #3
        #intercept = -2.764950   #4
        self.interceptChangeFor1bpMedsAdded = -0.1134942
   
    def will_have_fatal_stroke(self, person):
        fatalStrokeProb = self._stroke_case_fatality
        fatalProb = self._stroke_secondary_case_fatality if person._stroke else fatalStrokeProb
        return person._rng.uniform() < fatalProb

    def get_intercept_change(self, person):
        tst = TreatmentStrategiesType.BP.value
        if "bpMedsAdded" in person._treatmentStrategies[tst]:
            bpMedsAdded = person._treatmentStrategies[tst]['bpMedsAdded']
            interceptChange = bpMedsAdded * self.interceptChangeFor1bpMedsAdded
        else:
            interceptChange = 0
        return interceptChange

    def get_next_stroke_probability(self, person):
        #Q: I am not sure why it was set to 0 at the beginning
        strokeProbability = 0
        strokeProbability = scipySpecial.expit( super().estimate_next_risk(person) + self.get_intercept_change(person) )
        return strokeProbability
    
    def generate_next_outcome(self, person):
        fatal = self.will_have_fatal_stroke(person)
        nihss = StrokeNihssModel().estimate_next_risk(person)
        strokeSubtype = StrokeSubtypeModelRepository().get_stroke_subtype(person)
        strokeType = StrokeTypeModel().get_stroke_type(person)
        #localization = Localization.LEFT_HEMISPHERE
        #disability = 3 
        #return StrokeOutcome(fatal, nihss, strokeType, strokeSubtype, localization, disability)
        return StrokeOutcome(fatal, nihss, strokeType, strokeSubtype)
        
    def update_cv_outcome(self, person, fatal):
        person._outcomes[OutcomeType.CARDIOVASCULAR][-1][1].fatal = fatal
        
    def get_next_outcome(self, person):
        if person.has_outcome_at_current_age(OutcomeType.CARDIOVASCULAR):
            if person._rng.uniform(size=1) < self.get_next_stroke_probability(person):
                strokeOutcome = self.generate_next_outcome(person)
                #if we decide to use different stroke/mi models, double check if fatality needs to be decided in CVModel
                self.update_cv_outcome(person, strokeOutcome.fatal)
                return strokeOutcome
            else: 
                return None




