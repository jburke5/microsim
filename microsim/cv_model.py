from microsim.ascvd_outcome_model import ASCVDOutcomeModel
from microsim.regression_model import RegressionModel
from microsim.data_loader import load_model_spec
from microsim.outcome import Outcome, OutcomeType
from microsim.treatment import TreatmentStrategiesType

class CVModelBase(ASCVDOutcomeModel):
    """CV is an outcome type that we need to use with some outcome type model implementations (stroke and mi).
       The male and female cv models share the same functions so this base class includes all common elements."""
    def __init__(self, coefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=True):
        self._secondary_prevention_multiplier = 1.0
        self._mi_case_fatality = 0.13
        self._secondary_mi_case_fatality = 0.13
        self._stroke_case_fatality = 0.15
        self._secondary_stroke_case_fatality = 0.15
        self._statinAdded_relative_risk = 0.72 #doi:10.1001/jama.2022.12138
        #This is the average change of the intercept, see the models for male and female below for more details of
        #the optimized intercepts I found from simulations (and then obtained this average I use here)
        self.interceptChangeFor1bpMedsAdded = -0.1103125
        super().__init__(RegressionModel(
                            coefficients=coefficients,
                            coefficient_standard_errors={key: 0 for key in coefficients},
                            residual_mean=0,
                            residual_standard_deviation=0,),
                         tot_chol_hdl_ratio=tot_chol_hdl_ratio,
                         black_race_x_tot_chol_hdl_ratio=black_race_x_tot_chol_hdl_ratio,
                         wmhSpecific=wmhSpecific)

    def get_risk_for_person(self, person, years=1):
        #risk without any adjustment for bp meds
        cvRisk = super().get_risk_for_person(person, person._rng, years=years, interceptChangeFor1bpMedsAdded=self.interceptChangeFor1bpMedsAdded)

        if (person._mi) | (person._stroke):
            cvRisk = cvRisk * self._secondary_prevention_multiplier

        tst = TreatmentStrategiesType.STATIN.value
        if "statinsAdded" in person._treatmentStrategies[tst]:
            statinsAdded = person._treatmentStrategies[tst]['statinsAdded']
            cvRisk = cvRisk * self._statinAdded_relative_risk if statinsAdded>0 else cvRisk

        return cvRisk
        
    def generate_next_outcome(self, person):
        #for now assume it is not a fatal event, and update later at the stroke or mi outcomes
        #if in the future we chose different stroke/mi models that do not update cv outcome fatality, then cv fatality will need to be decided here
        fatal = False
        return Outcome(OutcomeType.CARDIOVASCULAR, fatal)
        
    def get_next_outcome(self, person):
        if person._rng.uniform(size=1) < self.get_risk_for_person(person):
            return self.generate_next_outcome(person)
        else: 
            return None        

    def get_risk_components_for_person(self, person, years=1):
        '''Returns the risk without taking into account silent cerebrovascular disease and the risk just due to scd.
        Does not make adjustments for secondary prevention as get_risk_for_person does.'''
        riskComponents = super().get_risk_components_for_person(person, person._rng, years=years, interceptChangeFor1bpMedsAdded=self.interceptChangeFor1bpMedsAdded)
        return riskComponents

class CVModelMale(CVModelBase):
    """CV model details for male gender."""
    #Some information about intercepts and bpMedsAdded...
    #interceptChangeFor1bpMedsAdded = -0.10537 #this is the mean from all bpMedsAdded 1,2,3,4
    #the optimal intercepts follow...these were found independently from each other from simulations...
    #intercept = -11.7902925  #1bpMedsAdded
    #intercept = -11.91125    #2
    #intercept = -12.00771437 #3
    #intercept = -12.101460   #4

    def __init__(self, intercept = -11.679980, wmhSpecific=True):
        maleCVCoefficients = {
            "lagAge": 0.064200,
            "black": 0.482835,
            "lagSbp#lagSbp": -0.000061,
            "lagSbp": 0.038950,
            "any_antiHypertensive": 2.055533,
            "current_diabetes": 0.842209,
            "current_smoker": 0.895589,
            "lagAge#black": 0,
            "lagSbp#any_antiHypertensive": -0.014207,
            "lagSbp#black": 0.011609,
            "black#any_antiHypertensive": -0.119460,
            "lagAge#lagSbp": 0.000025,
            "black#current_diabetes": -0.077214,
            "black#current_smoker": -0.226771,
            "lagSbp#black#any_antiHypertensive": 0.004190,
            "lagAge#lagSbp#black": -0.000199,
            "Intercept": intercept,
        }
        tot_chol_hdl_ratio=0.193307
        black_race_x_tot_chol_hdl_ratio=-0.117749
        super().__init__(maleCVCoefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=wmhSpecific)

class CVModelFemale(CVModelBase):
    """CV model details for female gender."""
    interceptChangeFor1bpMedsAdded = -0.1103125
    #interceptChangeFor1bpMedsAdded = -0.10537 #mean
    #Some information about the optimal intercepts for 1,2,3,4 bpMedsAdded
    #intercept = -12.9334225 #for 1 bpMedsAdded
    #intercept = -13.03125   #2
    #intercept = -13.150500  #3
    #intercept = -13.244250  #4

    def __init__(self, intercept = -12.823110, wmhSpecific=True):
        femaleCVCoefficients = {
            "lagAge": 0.106501,
            "black": 0.432440,
            "lagSbp#lagSbp": 0.000056,
            "lagSbp": 0.017666,
            "any_antiHypertensive": 0.731678,
            "current_diabetes": 0.943970,
            "current_smoker": 1.009790,
            "lagAge#black": -0.008580,
            "lagSbp#any_antiHypertensive": -0.003647,
            "lagSbp#black": 0.006208,
            "black#any_antiHypertensive": 0.152968,
            "lagAge#lagSbp": -0.000153,
            "black#current_diabetes": 0.115232,
            "black#current_smoker": -0.092231,
            "lagSbp#black#any_antiHypertensive": -0.000173,
            "lagAge#lagSbp#black": -0.000094,
            "Intercept": intercept,
        }
        tot_chol_hdl_ratio=0.151318
        black_race_x_tot_chol_hdl_ratio=0.070498
        super().__init__(femaleCVCoefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=wmhSpecific)


