from microsim.outcomes.ascvd_outcome_model import ASCVDOutcomeModel
from microsim.regression_models.regression_model import RegressionModel
from microsim.common.data_loader import load_model_spec
from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.outcomes.outcome_prevalence_base import OutcomePrevalenceBase
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.education import Education
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType

class CVModelBase(ASCVDOutcomeModel):
    """CV is an outcome type that we need to use with some outcome type model implementations (stroke and mi).
       The male and female cv models share the same functions so this base class includes all common elements."""
    def __init__(self, coefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=True, riskScaling=1.0):
        self._riskScaling = riskScaling
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

        cvRisk = cvRisk * self._riskScaling

        if (person._mi) | (person._stroke):
            cvRisk = cvRisk * self._secondary_prevention_multiplier

        tst = TreatmentStrategiesType.STATIN.value
        if "statinMedsAdded" in person._treatmentStrategies[tst]:
            statinMedsAdded = person._treatmentStrategies[tst]['statinMedsAdded']
            cvRisk = cvRisk * self._statinAdded_relative_risk if statinMedsAdded>0 else cvRisk

        tst = TreatmentStrategiesType.WMD15.value
        if "wmd15MedsAdded" in person._treatmentStrategies[tst]:
            wmd15MedsAdded = person._treatmentStrategies[tst]['wmd15MedsAdded']
            cvRisk = cvRisk * 0.85 if wmd15MedsAdded>0 else cvRisk

        tst = TreatmentStrategiesType.WMD20.value
        if "wmd20MedsAdded" in person._treatmentStrategies[tst]:
            wmd20MedsAdded = person._treatmentStrategies[tst]['wmd20MedsAdded']
            cvRisk = cvRisk * 0.80 if wmd20MedsAdded>0 else cvRisk

        tst = TreatmentStrategiesType.WMD25.value
        if "wmd25MedsAdded" in person._treatmentStrategies[tst]:
            wmd25MedsAdded = person._treatmentStrategies[tst]['wmd25MedsAdded']
            cvRisk = cvRisk * 0.75 if wmd25MedsAdded>0 else cvRisk

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

    def __init__(self, intercept = -11.679980, wmhSpecific=True, riskScaling=1.0):
        maleCVCoefficients = {
            "lagAge": 0.064200,
            "black": 0.482835,
            "lagSbp#lagSbp": -0.000061,
            "lagSbp": 0.038950,
            "any_antiHypertensive": 2.055533,
            "diabetes": 0.842209,
            "current_smoker": 0.895589,
            "lagAge#black": 0,
            "lagSbp#any_antiHypertensive": -0.014207,
            "lagSbp#black": 0.011609,
            "black#any_antiHypertensive": -0.119460,
            "lagAge#lagSbp": 0.000025,
            "black#diabetes": -0.077214,
            "black#current_smoker": -0.226771,
            "lagSbp#black#any_antiHypertensive": 0.004190,
            "lagAge#lagSbp#black": -0.000199,
            "Intercept": intercept,
        }
        tot_chol_hdl_ratio=0.193307
        black_race_x_tot_chol_hdl_ratio=-0.117749
        super().__init__(maleCVCoefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=wmhSpecific, riskScaling=riskScaling)

class CVModelFemale(CVModelBase):
    """CV model details for female gender."""
    interceptChangeFor1bpMedsAdded = -0.1103125
    #interceptChangeFor1bpMedsAdded = -0.10537 #mean
    #Some information about the optimal intercepts for 1,2,3,4 bpMedsAdded
    #intercept = -12.9334225 #for 1 bpMedsAdded
    #intercept = -13.03125   #2
    #intercept = -13.150500  #3
    #intercept = -13.244250  #4

    def __init__(self, intercept = -12.823110, wmhSpecific=True, riskScaling=1.0):
        femaleCVCoefficients = {
            "lagAge": 0.106501,
            "black": 0.432440,
            "lagSbp#lagSbp": 0.000056,
            "lagSbp": 0.017666,
            "any_antiHypertensive": 0.731678,
            "diabetes": 0.943970,
            "current_smoker": 1.009790,
            "lagAge#black": -0.008580,
            "lagSbp#any_antiHypertensive": -0.003647,
            "lagSbp#black": 0.006208,
            "black#any_antiHypertensive": 0.152968,
            "lagAge#lagSbp": -0.000153,
            "black#diabetes": 0.115232,
            "black#current_smoker": -0.092231,
            "lagSbp#black#any_antiHypertensive": -0.000173,
            "lagAge#lagSbp#black": -0.000094,
            "Intercept": intercept,
        }
        tot_chol_hdl_ratio=0.151318
        black_race_x_tot_chol_hdl_ratio=0.070498
        super().__init__(femaleCVCoefficients, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=wmhSpecific, riskScaling=riskScaling)


class CVPrevalenceModel(OutcomePrevalenceBase):
    """Logistic prevalence model that seeds priorToSim CV at Person construction.
       Age and gender are the only inputs, so the realized prevalence per age group and
       gender matches the GBD rates the coefficients were fit to, with no risk scaling."""

    _outcomeType = OutcomeType.CARDIOVASCULAR

    # Per-gender OLS fit of logit(prevalence) on age-group midpoints to the GBD rates in
    # Reference.prevalence[cv] (ages 50-74, USA, 1999); max residual 0.008, 2026-08-28.
    # Regenerate with calibration.fit_cv_prevalence; test_calibration enforces agreement.
    _coefficients = {
        NHANESGender.MALE: {"Intercept": -6.7306, "age": 0.09978},
        NHANESGender.FEMALE: {"Intercept": -6.7626, "age": 0.09176},
    }

    def __init__(self, riskScaling=1.0):
        self._riskScaling = riskScaling

    def get_linear_predictor_for_person(self, person):
        return self.calc_linear_predictor_for_patient_characteristics(
            person._age[-1],
            person._gender,
        )

    def calc_linear_predictor_for_patient_characteristics(self, age, gender):
        coefficients = self._coefficients[gender]
        return coefficients["Intercept"] + coefficients["age"] * age
