from mcm.outcome import Outcome
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.gender import NHANESGender
from mcm.ascvd_outcome_model import ASCVDOutcomeModel


class OutcomeModelRepository:

    def __init__(self):
        self._models = {}
        self._models[Outcome.CARDIOVASCULAR] = {
            "female": ASCVDOutcomeModel(
                age=0.106501, black_race=0.432440, sbp_x_sbp=0.000056, sbp=0.017666,
                bp_treatment=0.731678, diabetes=0.943970, current_smoker=1.009790,
                tot_chol_hdl_ratio=0.151318, age_x_black_race=-0.008580,
                sbp_x_treatment=-0.003647, sbp_x_black_race=0.006208,
                black_race_x_treatment=0.152968, age_x_sbp=-0.000153,
                black_race_x_diabetes=0.115232,
                black_race_x_current_smoker=-0.092231,  black_race_x_tot_chol_hdl_ratio=0.070498,
                sbp_x_black_race_x_treatment=-0.000173, age_x_sbp_x_black_race=-0.000094,
                intercept=-12.823110
            ),
            "male": ASCVDOutcomeModel(
                age=0.064200, black_race=0.482835, sbp_x_sbp=-0.000061, sbp=0.038950,
                bp_treatment=2.055533, diabetes=0.842209, current_smoker=0.895589,
                tot_chol_hdl_ratio=0.193307, age_x_black_race=0,
                sbp_x_treatment=-0.014207, sbp_x_black_race=0.011609,
                black_race_x_treatment=-0.119460, age_x_sbp=0.000025,
                black_race_x_diabetes=-0.077214,
                black_race_x_current_smoker=-0.226771,  black_race_x_tot_chol_hdl_ratio=-0.117749,
                sbp_x_black_race_x_treatment=0.004190, age_x_sbp_x_black_race=-0.000199,
                intercept=-11.679980
            ),

        }

    def get_risk_for_person(self, person, outcome, years):
        return self.select_model_for_person(person, outcome).get_risk_for_person(person, years)

    def select_model_for_person(self, person, outcome):
        models_for_outcome = self._models[outcome]
        gender_stem = "male" if person._gender == NHANESGender.MALE else "female"
        return models_for_outcome[gender_stem]
