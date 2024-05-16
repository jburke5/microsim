from microsim.ascvd_outcome_model import ASCVDOutcomeModel
from microsim.regression_model import RegressionModel
from microsim.data_loader import load_model_spec
from microsim.outcome import Outcome, OutcomeType

class CVModelBase(ASCVDOutcomeModel):
    """CV is an outcome type that we need to use with some outcome type model implementations (stroke and mi).
       The male and female cv models share the same functions so this base class includes all common elements."""
    def __init__(self, coefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio):
        self._secondary_prevention_multiplier = 1.0
        self._mi_case_fatality = 0.13
        self._secondary_mi_case_fatality = 0.13
        self._stroke_case_fatality = 0.15
        self._secondary_stroke_case_fatality = 0.15
        super().__init__(RegressionModel(
                            coefficients=coefficients,
                            coefficient_standard_errors={key: 0 for key in coefficients},
                            residual_mean=0,
                            residual_standard_deviation=0,),
                         tot_chol_hdl_ratio=tot_chol_hdl_ratio,
                         black_race_x_tot_chol_hdl_ratio=black_race_x_tot_chol_hdl_ratio,)
        
    def get_risk_for_person(self, person, years=1):
        cvRisk = super().get_risk_for_person(person, person._rng, years=years)
        if (person._mi) | (person._stroke):
            cvRisk = cvRisk * self._secondary_prevention_multiplier
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

class CVModelMale(CVModelBase):
    """CV model details for male gender."""
    interceptChangeFor1bpMedsAdded = -0.1103125
    #interceptChangeFor1bpMedsAdded = -0.10537 #mean

    def __init__(self, intercept = -11.679980):
        maleCVCoefficients = {
            "lagAge": 0.064200,
            "black": 0.482835,
            "lagSbp#lagSbp": -0.000061,
            "lagSbp": 0.038950,
            "current_bp_treatment": 2.055533,
            "current_diabetes": 0.842209,
            "current_smoker": 0.895589,
            "lagAge#black": 0,
            "lagSbp#current_bp_treatment": -0.014207,
            "lagSbp#black": 0.011609,
            "black#current_bp_treatment": -0.119460,
            "lagAge#lagSbp": 0.000025,
            "black#current_diabetes": -0.077214,
            "black#current_smoker": -0.226771,
            "lagSbp#black#current_bp_treatment": 0.004190,
            "lagAge#lagSbp#black": -0.000199,
            "Intercept": intercept,
        }
        tot_chol_hdl_ratio=0.193307
        black_race_x_tot_chol_hdl_ratio=-0.117749
        super().__init__(maleCVCoefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio)

class CVModelMaleFor1bpMedsAdded(CVModelMale):
    def __init__(self):
        #super().__init__(intercept = -11.7902925)
        super().__init__(intercept = -11.679980 + 1*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor2bpMedsAdded(CVModelMale):
    def __init__(self):
        #super().__init__(intercept = -11.91125)
        super().__init__(intercept = -11.679980 + 2*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor3bpMedsAdded(CVModelMale):
    def __init__(self):
        #super().__init__(intercept = -12.007714375)
        super().__init__(intercept = -11.679980 + 3*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor4bpMedsAdded(CVModelMale):
    def __init__(self):
        #super().__init__(intercept = -12.101460)
        super().__init__(intercept = -11.679980 + 4*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor5bpMedsAdded(CVModelMale):
    def __init__(self):
        super().__init__(intercept = -11.679980 + 5*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor6bpMedsAdded(CVModelMale):
    def __init__(self):
        super().__init__(intercept = -11.679980 + 6*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor7bpMedsAdded(CVModelMale):
    def __init__(self):
        super().__init__(intercept = -11.679980 + 7*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor8bpMedsAdded(CVModelMale):
    def __init__(self):
        super().__init__(intercept = -11.679980 + 8*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor9bpMedsAdded(CVModelMale):
    def __init__(self):
        super().__init__(intercept = -11.679980 + 9*self.interceptChangeFor1bpMedsAdded )

class CVModelMaleFor10bpMedsAdded(CVModelMale):
    def __init__(self):
        super().__init__(intercept = -11.679980 + 10*self.interceptChangeFor1bpMedsAdded )

#class CVModelMaleForNbpMedsAdded(CVModelMale):
#    def __init__(self):
#        super().__init__(intercept =  -11.679980) 

class CVModelFemale(CVModelBase):
    """CV model details for female gender."""
    interceptChangeFor1bpMedsAdded = -0.1103125
    #interceptChangeFor1bpMedsAdded = -0.10537 #mean

    def __init__(self, intercept = -12.823110):
        femaleCVCoefficients = {
            "lagAge": 0.106501,
            "black": 0.432440,
            "lagSbp#lagSbp": 0.000056,
            "lagSbp": 0.017666,
            "current_bp_treatment": 0.731678,
            "current_diabetes": 0.943970,
            "current_smoker": 1.009790,
            "lagAge#black": -0.008580,
            "lagSbp#current_bp_treatment": -0.003647,
            "lagSbp#black": 0.006208,
            "black#current_bp_treatment": 0.152968,
            "lagAge#lagSbp": -0.000153,
            "black#current_diabetes": 0.115232,
            "black#current_smoker": -0.092231,
            "lagSbp#black#current_bp_treatment": -0.000173,
            "lagAge#lagSbp#black": -0.000094,
            "Intercept": intercept,
        }
        tot_chol_hdl_ratio=0.151318
        black_race_x_tot_chol_hdl_ratio=0.070498
        super().__init__(femaleCVCoefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio)

class CVModelFemaleFor1bpMedsAdded(CVModelFemale):
    def __init__(self):
        #super().__init__(intercept = -12.9334225)
        super().__init__(intercept = -12.823110 + 1*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor2bpMedsAdded(CVModelFemale):
    def __init__(self):
        #super().__init__(intercept = -13.03125)
        super().__init__(intercept = -12.823110 + 2*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor3bpMedsAdded(CVModelFemale):
    def __init__(self):
        #super().__init__(intercept = -13.150500)
        super().__init__(intercept = -12.823110 + 3*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor4bpMedsAdded(CVModelFemale):
    def __init__(self):
        #super().__init__(intercept = -13.244250 )
        super().__init__(intercept = -12.823110 + 4*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor5bpMedsAdded(CVModelFemale):
    def __init__(self):
        super().__init__(intercept = -12.823110 + 5*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor6bpMedsAdded(CVModelFemale):
    def __init__(self):
        super().__init__(intercept = -12.823110 + 6*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor7bpMedsAdded(CVModelFemale):
    def __init__(self):
        super().__init__(intercept = -12.823110 + 7*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor8bpMedsAdded(CVModelFemale):
    def __init__(self):
        super().__init__(intercept = -12.823110 + 8*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor9bpMedsAdded(CVModelFemale):
    def __init__(self):
        super().__init__(intercept = -12.823110 + 9*self.interceptChangeFor1bpMedsAdded )

class CVModelFemaleFor10bpMedsAdded(CVModelFemale):
    def __init__(self):
        super().__init__(intercept = -12.823110 + 10*self.interceptChangeFor1bpMedsAdded )

#class CVModelFemaleForNbpMedsAdded(CVModelFemale):
#    def __init__(self):
#        super().__init__(intercept =  -12.823110)

