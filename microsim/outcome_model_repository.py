from microsim.outcome_model_type import OutcomeModelType
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.ascvd_outcome_model import ASCVDOutcomeModel
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel
from microsim.cv_outcome_determination import CVOutcomeDetermination
from microsim.person import Person
from microsim.data_loader import load_model_spec
from microsim.regression_model import RegressionModel
from microsim.gcp_model import GCPModel
from microsim.gcp_stroke_model import GCPStrokeModel
from microsim.outcome import Outcome, OutcomeType
from microsim.dementia_model import DementiaModel
from microsim.statsmodel_logistic_risk_factor_model import StatsModelLogisticRiskFactorModel
from microsim.smoking_status import SmokingStatus

import numpy.random as npRand
import numpy as np


# This object is currently serving two purposes.
# First, it is (as its name implies), a repository fo routcome models
# Second, it is the gateway into logic around signing outcomes.
# To the extne that logic is complex, it has been moved into separate "Determination" classe
# e.g. CVOutcomeDetermination


class OutcomeModelRepository:
    def __init__(self):
        self.mi_case_fatality = CVOutcomeDetermination.default_mi_case_fatality
        self.secondary_mi_case_fatality = CVOutcomeDetermination.default_secondary_mi_case_fatality
        self.stroke_case_fatality = CVOutcomeDetermination.default_stroke_case_fatality
        self.secondary_stroke_case_fatality = (
            CVOutcomeDetermination.default_secondary_stroke_case_fatality
        )
        self.secondary_prevention_multiplier = (
            CVOutcomeDetermination.default_secondary_prevention_multiplier
        )

        self.outcomeDet = CVOutcomeDetermination(
            self.mi_case_fatality,
            self.stroke_case_fatality,
            self.secondary_mi_case_fatality,
            self.secondary_stroke_case_fatality,
            self.secondary_prevention_multiplier,
        )

        # variable used in testing to control whether a patient will have a stroke or mi
        self.manualStrokeMIProbability = None

        self._resultReporting = {}
        self._models = {}
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
            "Intercept": -12.823110,
        }

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
            "Intercept": -11.679980,
        }

        self._models[OutcomeModelType.CARDIOVASCULAR] = {
            "female": ASCVDOutcomeModel(
                RegressionModel(
                    coefficients=femaleCVCoefficients,
                    coefficient_standard_errors={key: 0 for key in femaleCVCoefficients},
                    residual_mean=0,
                    residual_standard_deviation=0,
                ),
                tot_chol_hdl_ratio=0.151318,
                black_race_x_tot_chol_hdl_ratio=0.070498,
            ),
            "male": ASCVDOutcomeModel(
                RegressionModel(
                    coefficients=maleCVCoefficients,
                    coefficient_standard_errors={key: 0 for key in maleCVCoefficients},
                    residual_mean=0,
                    residual_standard_deviation=0,
                ),
                tot_chol_hdl_ratio=0.193307,
                black_race_x_tot_chol_hdl_ratio=-0.117749,
            ),
        }
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = {
            "preStroke": GCPModel(self),
            "postStroke": GCPStrokeModel(self)
        }
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()

        # This represents non-cardiovascular mortality..
        nonCVModelSpec = load_model_spec("nhanesMortalityModelLogit")
        mortModel = StatsModelLogisticRiskFactorModel(RegressionModel(**nonCVModelSpec), False)
        # Recalibrate mortalitly model to align with life table data, as explored in notebook
        # buildNHANESMortalityModel
        mortModel.non_intercept_params["age"] = mortModel.non_intercept_params["age"] * -1
        mortModel.non_intercept_params["squareAge"] = (
            mortModel.non_intercept_params["squareAge"] * 4
        )
        self._models[OutcomeModelType.NON_CV_MORTALITY] = mortModel

    def get_random_effects(self, rng=None):
        #rng = np.random.default_rng(rng)
        return {"gcp": rng.normal(0, 4.84)}

    def get_risk_for_person_vectorized(self, x, outcome, years=1, rng=None):
        #rng = np.random.default_rng(rng)
        personWrapper = PersonRowWrapper(x)
        return self.select_model_for_person(personWrapper, outcome).get_risk_for_person(
            x, rng, years, True
        )

    def get_risk_for_person(self, person, outcome, years=1, vectorized=False, rng=None):
        #rng = np.random.default_rng(rng)
        if vectorized:
            return self.get_risk_for_person_vectorized(person, outcome, years, rng)
        else:
            return self.select_model_for_person(person, outcome).get_risk_for_person(
                person, rng, years, False
            )

    def get_gcp(self, person, rng=None):
        gcp = (
            self.get_risk_for_person(person, OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE, rng=rng)
        )
        return gcp if gcp > 0 else 0

    # should the GCP random effct be included here or in the risk model?
    def get_gcp_vectorized(self, person, rng=None):
        #rng = np.random.default_rng(rng)
        gcp = (
            self.get_risk_for_person(
                person, OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE, years=1, vectorized=True, rng=rng
            )
        )
        return gcp if gcp > 0 else 0

    def get_dementia(self, person, rng=None):
        #rng = np.random.default_rng(rng)
        if rng.uniform(size=1) < self.get_risk_for_person(person, OutcomeModelType.DEMENTIA):
            return Outcome(OutcomeType.DEMENTIA, False)
        else:
            return None

    def get_dementia_vectorized(self, person, rng=None):
        #rng = np.random.default_rng(rng)
        return rng.uniform(size=1)[0] < self.get_risk_for_person(
            person, OutcomeModelType.DEMENTIA, years=1, vectorized=True
        )

    def select_model_for_person(self, person, outcome):
        if outcome == OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE:
            return self.select_model_for_stroke(person, outcome)
        else:
            return self.select_model_for_gender(person._gender, outcome)

    def select_model_for_gender(self, gender, outcome):
        models_for_outcome = self._models[outcome]
        if outcome == OutcomeModelType.CARDIOVASCULAR:
            gender_stem = "male" if gender == NHANESGender.MALE else "female"
            return models_for_outcome[gender_stem]
        else:
            return models_for_outcome

    def select_model_for_stroke(self, person, outcome):
        models_for_outcome = self._models[outcome]
        if outcome == OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE:
            #we are interested in strokes that occured during the simulation, not so much on strokes that NHANES had registered for people
            #so we are selecting the gcp stroke model only when there is a stroke during the simulation
            #plus, the gcp stroke model requires quantities that we do not have from NHANES (we would need to come up with estimates)
            strokeStatus = "postStroke" if (len(person._outcomes[OutcomeType.STROKE])>0) else "preStroke" #using the PersonRowWrapper class here is not ideal
            return models_for_outcome[strokeStatus]
        else:
            return models_for_outcome

    def initialize_cox_model(self, modelName):
        model_spec = load_model_spec(modelName)
        return StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None, rng=None):
        #rng = np.random.default_rng(rng)
        return self.outcomeDet.assign_outcome_for_person(
            self, person, False, years, self.manualStrokeMIProbability, rng
        )

    def assign_cv_outcome_vectorized(self, x, years=1, manualStrokeMIProbability=None, rng=None):
        #rng = np.random.default_rng(rng)
        return self.outcomeDet.assign_outcome_for_person(
            self,
            x,
            vectorized=True,
            years=years,
            manualStrokeMIProbability=self.manualStrokeMIProbability,
            rng=rng,
        )

        # Returns True if the model-based logic vs. the random comparison suggests death

    def assign_non_cv_mortality(self, person, years=1, rng=None):
        #rng = np.random.default_rng(rng)
        riskForPerson = self.get_risk_for_person(person, OutcomeModelType.NON_CV_MORTALITY)
        if rng.uniform(size=1)[0] < riskForPerson:
            return True

    def assign_non_cv_mortality_vectorized(self, person, years=1, rng=None):
        #rng = np.random.default_rng(rng)
        riskForPerson = self.get_risk_for_person(
            person, OutcomeModelType.NON_CV_MORTALITY, years, vectorized=True
        )
        return rng.uniform(size=1)[0] < riskForPerson

        # utility class to take a dataframe row and convert some salietn elements to a person to streamline model selection

    # for reporting this is a place to store debugging data to see how elements of outcomes models are behaving
    def report_result(self, resultName, resultDict):
        if not resultName in self._resultReporting:
            self._resultReporting[resultName]= []
        self._resultReporting[resultName].append(resultDict)

class PersonRowWrapper:
    def __init__(self, x):
        if np.isnan(x.gender):
            print("Is nan")
            print(x)
        self._age = x.age
        self._gender = NHANESGender(x.gender)
        self._raceEthnicity = NHANESRaceEthnicity(x.raceEthnicity)
        self._education = Education(x.education)
        self._smokingStatus = SmokingStatus(x.smokingStatus)
        self._outcomes = {OutcomeType.MI: [], OutcomeType.STROKE: [], OutcomeType.DEMENTIA: []}
        if x.stroke:
            #since this is a dummy person class, create a dummy stroke outcome
            self._outcomes[OutcomeType.STROKE].append((-1, Outcome(OutcomeType.STROKE, False))) 
