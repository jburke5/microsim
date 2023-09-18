import numpy as np
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.statsmodel_rel_risk_factor_model import StatsModelRelRiskFactorModel
from microsim.stroke_outcome import StrokeOutcome, StrokeSubtype, StrokeType, Localization
from microsim.regression_model import RegressionModel

class StrokeNihssModel(StatsModelLinearRiskFactorModel):

    def __init__(self):
        self._model = {"coefficients": {
            "Intercept":           -2.6063356,   #_cons + aric + glucose_sim * (-46.7)
            "age":                  0.0771289,   #stroke_age
            "gender[T.2]":          0.0512374,   #female0_sim
            "gender[T.1]":          0.,
            "education[T.1]":       0.1658821,   #educ0_sim*1, 1=not a HS graduate
            "education[T.2]":       0.1658821,   #educ0_sim*1, 1=not a HS graduate
            "education[T.3]":       0.3317642,   #educ0_sim*2, 2=HS graduate
            "education[T.4]":       0.4976463,   #educ0_sim*3, 3=some college
            "education[T.5]":       0.6635284,   #educ0_sim*4, 4=college graduate
            "smokingStatus[T.2]":  -0.1257199 ,  #currsmoker_sim
            "anyPhysicalActivity": -0.3476088,   #physact_sim
            "afib":                 1.794008,    #hxafib_sim     
            "current_bp_treatment": 0.6804093,   #htntx_sim
            "statin":              -0.1660485,   #choltx_sim
            "sbp":                  0.0131255 ,  #sbpstkcog_sim
            "dbp":                 -0.0144218,   #dbpstkcog_sim 
            "hdl":                 -0.2642685 ,  #cholhdl_sim
            "totChol":              0.2577162,   #choltot_sim
            "bmi":                 -0.0197889,   #bmi_sim
            "alcoholPerWeek":      -0.26697522,  #alcperwk_sim * 0.6
            "ldl":                 -0.2564912,   #cholldl_sim
            "trig":                -0.0556979,   #trig_sim 
            "a1c":                  0.2374925,   #glucose_sim * 28.7
            "waist":               -0.002543 ,   #waistcm_sim
            "creatinine":           0.2782118    #creatin_sim
            },
                "coefficient_standard_errors": {} ,
                "residual_mean": {},
                "residual_standard_deviation": {} }
        
        self._regressionModel = RegressionModel(**self._model)
        super().__init__(self._regressionModel)

    def estimate_next_risk_vectorized(self, person):
        return min( max(0, round(super().estimate_next_risk_vectorized(person))), 42) #constrain regression results

    def estimate_next_risk(self, person):
        return min( max(0, round(super().estimate_next_risk(person))), 42) #constrain regression results

class StrokeTypeModel():
    
    def __init__(self, rng=None):
        self._ischemicRatio = 0.89956 #1227/1364 using data from Levine et al. 
        self._rng = rng

    def estimate_ischemic_risk(self, person):
        return self._rng.uniform()

    def estimate_ischemic_risk_vectorized(self, person):
        return self._rng.uniform()

    def get_stroke_type(self, person):
        return StrokeType.ISCHEMIC if (self.estimate_ischemic_risk(person)<self._ischemicRatio) else StrokeType.ICH
    
    def get_stroke_type_vectorized(self, person):
        return StrokeType.ISCHEMIC if (self.estimate_ischemic_risk_vectorized(person)<self._ischemicRatio) else StrokeType.ICH

class StrokeSubtypeCEModel(StatsModelRelRiskFactorModel):
    def __init__(self):
        
        #cardioembolic
        self._model = {"coefficients": {
           "Intercept":          -3.13182975,  #_cons + aric + glucose_sim * (-46.7)
           "age":                 0.0499947,   #stroke_age
           "gender[T.2]":        -0.3813221,   #female0_sim
           "gender[T.1]":         0.,
           "education[T.1]":     -0.0143641,   #educ0_sim * 1, 1=not a HS graduate
           "education[T.2]":     -0.0143641,   #educ0_sim * 1, 1=not a HS graduate
           "education[T.3]":     -0.0287282,   #educ0_sim * 2, 2=HS graduate
           "education[T.4]":     -0.0430923,   #educ0_sim * 3, 3=some college
           "education[T.5]":     -0.0574564,   #educ0_sim * 4, 4=college graduate
           "smokingStatus[T.2]":  0.0464248,   #currsmoker_sim
           "anyPhysicalActivity": 0.171529,    #physact_sim
           "afib":                1.858444,    #hxafib_sim     
           "current_bp_treatment":0.1267211,   #htntx_sim
           "statin":              0.0405856,   #choltx_sim
           "sbp":                 0.0044229,   #sbpstkcog_sim
           "dbp":                -0.0211307,   #dbpstkcog_sim 
           "hdl":                 0.0076896,   #cholhdl_sim
           "totChol":            -0.0033463,   #choltot_sim
           "bmi":                 0.0421033,   #bmi_sim
           "alcoholPerWeek":     -0.00259902,  #alcperwk_sim * 0.6
           "ldl":                -0.0031348,   #cholldl_sim
           "trig":               -0.0000179,   #trig_sim 
           "a1c":                -0.11501525,  #glucose_sim * 28.7
           "waist":               0.0011406,   #waistcm_sim
           "creatinine":         -0.1197203    #creatin_sim
           },
               "coefficient_standard_errors": {} ,
               "residual_mean": {},
               "residual_standard_deviation": {} }
        
        self._regressionModel = RegressionModel(**self._model)
        super().__init__(self._regressionModel)

class StrokeSubtypeLVModel(StatsModelRelRiskFactorModel):
    def __init__(self):
        
        #LA_atherosclerosis
        self._model = {"coefficients": {
            "Intercept":          -2.14462214,  #_cons + aric + glucose_sim * (-46.7)
            "age":                 0.00203,     #stroke_age
            "gender[T.2]":        -0.3026703,   #female0_sim
            "gender[T.1]":         0.,
            "education[T.1]":      0.0204964,   #educ0_sim * 1, 1=not a HS graduate
            "education[T.2]":      0.0204964,   #educ0_sim * 1, 1=not a HS graduate
            "education[T.3]":      0.0409928,   #educ0_sim * 2, 2=HS graduate
            "education[T.4]":      0.0614892,   #educ0_sim * 3, 3=some college
            "education[T.5]":      0.0819856,   #educ0_sim * 4, 4=college graduate
            "smokingStatus[T.2]": -0.4087069,   #currsmoker_sim
            "anyPhysicalActivity": 0.3283082,   #physact_sim
            "afib":                0.7040827,   #hxafib_sim     
            "current_bp_treatment":0.1878509,   #htntx_sim
            "statin":              0.1681594,   #choltx_sim
            "sbp":                 0.0105225,   #sbpstkcog_sim
            "dbp":                -0.0154197,   #dbpstkcog_sim 
            "hdl":                -0.0253099 ,  #cholhdl_sim
            "totChol":             0.01298,     #choltot_sim
            "bmi":                -0.0042248,   #bmi_sim
            "alcoholPerWeek":      0.03367986,  #alcperwk_sim * 0.6
            "ldl":                -0.009429,    #cholldl_sim
            "trig":               -0.0005325,   #trig_sim 
            "a1c":                -0.04051866,  #glucose_sim * 28.7
            "waist":               0.0017584,   #waistcm_sim
            "creatinine":         -0.6562215    #creatin_sim
            },
                "coefficient_standard_errors": {} ,
                "residual_mean": {},
                "residual_standard_deviation": {} }     

        self._regressionModel = RegressionModel(**self._model)
        super().__init__(self._regressionModel)

class StrokeSubtypeSVModel(StatsModelRelRiskFactorModel):
    def __init__(self):
        
        #SV_occlusion
        self._model = {"coefficients": {
            "Intercept":            0.17084136,  #_cons + aric + glucose_sim * (-46.7)
            "age":                 -0.008983,    #stroke_age
            "gender[T.2]":          0.1107677,   #female0_sim
            "gender[T.1]":          0.,
            "education[T.1]":      -0.0357296,   #educ0_sim*1, 1=not a HS graduate
            "education[T.2]":      -0.0357296,   #educ0_sim*1, 1=not a HS graduate
            "education[T.3]":      -0.0714592,   #educ0_sim*2, 2=HS graduate
            "education[T.4]":      -0.1071888,   #educ0_sim*3, 3=some college
            "education[T.5]":      -0.1429184,   #educ0_sim*4, 4=college graduate
            "smokingStatus[T.2]":   0.0207556,   #currsmoker_sim
            "anyPhysicalActivity":  0.1915707,   #physact_sim
            "afib":                -0.0698982,   #hxafib_sim     
            "current_bp_treatment":-0.0168805,   #htntx_sim
            "statin":              -0.4778607,   #choltx_sim
            "sbp":                  0.0101797,   #sbpstkcog_sim
            "dbp":                 -0.0110669,   #dbpstkcog_sim 
            "hdl":                 -0.032421,    #cholhdl_sim
            "totChol":              0.0257892,   #choltot_sim
            "bmi":                 -0.0029924,   #bmi_sim
            "alcoholPerWeek":       0.0877935,   #alcperwk_sim * 0.6
            "ldl":                 -0.0302675,   #cholldl_sim
            "trig":                -0.0058004,   #trig_sim 
            "a1c":                 -0.00062566,  #glucose_sim * 28.7
            "waist":                0.0118192,   #waistcm_sim
            "creatinine":          -0.5225816    #creatin_sim
            },
                "coefficient_standard_errors": {} ,
                "residual_mean": {},
                "residual_standard_deviation": {} }
        
        self._regressionModel = RegressionModel(**self._model)
        super().__init__(self._regressionModel)

class StrokeSubtypeModelRepository:
    
    def __init__(self, rng=None):
        self._rng = rng
    
    def get_stroke_subtype(self, person):
        
        ceRelRisk = StrokeSubtypeCEModel().estimate_rel_risk(person)
        lvRelRisk = StrokeSubtypeLVModel().estimate_rel_risk(person)
        svRelRisk = StrokeSubtypeSVModel().estimate_rel_risk(person)
        otRelRisk = 1 #this was the base subtype on the multinomial logistic regression model

        sumRelRisk = otRelRisk + ceRelRisk + lvRelRisk + svRelRisk

        #probabilities are just rescaled relative risks, no need to calculate them, just draw on the rel risk scale
        draw = self._rng.uniform(low=0., high=sumRelRisk)

        if (draw<lvRelRisk):
            return StrokeSubtype.LARGE_VESSEL #most common, so check this first
        elif (draw<lvRelRisk+svRelRisk):
            return StrokeSubtype.SMALL_VESSEL
        elif (draw<lvRelRisk+svRelRisk+otRelRisk):
            return StrokeSubtype.OTHER
        else:
            return StrokeSubtype.CARDIOEMBOLIC

    def get_stroke_subtype_vectorized(self, person):
        
        ceRelRisk = StrokeSubtypeCEModel().estimate_rel_risk_vectorized(person)
        lvRelRisk = StrokeSubtypeLVModel().estimate_rel_risk_vectorized(person)
        svRelRisk = StrokeSubtypeSVModel().estimate_rel_risk_vectorized(person)
        otRelRisk = 1 #this was the base subtype on the multinomial logistic regression model

        sumRelRisk = otRelRisk + ceRelRisk + lvRelRisk + svRelRisk

        #probabilities are just rescaled relative risks, no need to calculate them, just draw on the rel risk scale
        draw = self._rng.uniform(low=0., high=sumRelRisk)

        if (draw<lvRelRisk):
            return StrokeSubtype.LARGE_VESSEL #most common, so check this first
        elif (draw<lvRelRisk+svRelRisk):
            return StrokeSubtype.SMALL_VESSEL
        elif (draw<lvRelRisk+svRelRisk+otRelRisk):
            return StrokeSubtype.OTHER
        else:
            return StrokeSubtype.CARDIOEMBOLIC
        

