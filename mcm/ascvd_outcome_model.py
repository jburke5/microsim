import numpy as np
from mcm.smoking_status import SmokingStatus
from mcm.race_ethnicity import NHANESRaceEthnicity

# https://annals.org/aim/fullarticle/2683613/[XSLTImagePath]


class ASCVDOutcomeModel:

    def __init__(self, age, black_race, sbp_x_sbp, sbp, bp_treatment, diabetes, current_smoker,
                 tot_chol_hdl_ratio, age_x_black_race, sbp_x_treatment, sbp_x_black_race,
                 black_race_x_treatment, age_x_sbp, black_race_x_diabetes,
                 black_race_x_current_smoker, black_race_x_tot_chol_hdl_ratio,
                 sbp_x_black_race_x_treatment, age_x_sbp_x_black_race, intercept):

        self._age = age
        self._black_race = black_race
        self._sbp_x_sbp = sbp_x_sbp
        self._sbp = sbp
        self._bp_treatment = bp_treatment
        self._diabetes = diabetes
        self._current_smoker = current_smoker
        self._tot_chol_hdl_ratio = tot_chol_hdl_ratio
        self._age_x_black_race = age_x_black_race
        self._sbp_x_treatment = sbp_x_treatment
        self._sbp_x_black_race = sbp_x_black_race
        self._black_race_x_treatment = black_race_x_treatment
        self._age_x_sbp = age_x_sbp
        self._black_race_x_diabetes = black_race_x_diabetes
        self._black_race_x_current_smoker = black_race_x_current_smoker
        self._black_race_x_tot_chol_hdl_ratio = black_race_x_tot_chol_hdl_ratio
        self._sbp_x_black_race_x_treatment = sbp_x_black_race_x_treatment
        self._age_x_sbp_x_black_race = age_x_sbp_x_black_race
        self._intercept = intercept

    def calc_linear_predictor(self, person):
        anyBpTreatment = person._antiHypertensiveCount[-1] > 0
        
        xb = self._intercept
        xb += self._age * person._age[-1]
        xb += self._sbp_x_sbp * (person._sbp[-1] ** 2)
        xb += self._sbp * person._sbp[-1]
        xb += self._diabetes * person.has_diabetes()
        xb += self._tot_chol_hdl_ratio * person._totChol[-1] / person._hdl[-1]
        xb += self._age_x_sbp * person._age[-1] * person._sbp[-1]

        if person._smokingStatus == SmokingStatus.CURRENT:
            xb += self._current_smoker

        if person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += self._black_race
            xb += self._age_x_black_race * person._age[-1]
            xb += self._sbp_x_black_race * person._sbp[-1]
            xb += self._black_race_x_diabetes * person.has_diabetes()
            xb += self._black_race_x_tot_chol_hdl_ratio * person._totChol[-1] / person._hdl[-1]
            xb += self._age_x_sbp_x_black_race * person._age[-1] * person._sbp[-1]
            if (person._smokingStatus == SmokingStatus.CURRENT):
                xb += self._black_race_x_current_smoker
            xb += self._sbp_x_black_race_x_treatment * person._sbp[-1] * anyBpTreatment
            xb += self._black_race_x_treatment * anyBpTreatment

        xb += self._bp_treatment * 0
        xb += self._sbp_x_treatment * person._sbp[-1] * anyBpTreatment
        return xb

    # TODO : need to figure out how to account fo rtime...which may be trikcy
    def get_risk_for_person(self, person, years):
        return (1 / (1 + np.exp(-1 * self.calc_linear_predictor(person)))) * years / 10
