import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel

# https://annals.org/aim/fullarticle/2683613/[XSLTImagePath]


class ASCVDOutcomeModel(StatsModelLinearRiskFactorModel):

    def __init__(self, regression_model, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio):

        super().__init__(regression_model, False)
        self._tot_chol_hdl_ratio = tot_chol_hdl_ratio
        self._black_race_x_tot_chol_hdl_ratio = black_race_x_tot_chol_hdl_ratio

    def get_manual_parameters(self, vectorized):
        if vectorized:
            return {'tot_chol_hdl_ratio': (self._tot_chol_hdl_ratio, lambda x: x.totChol / x.hdl),
                    'black_race_x_tot_chol_hdl_ratio': (self._black_race_x_tot_chol_hdl_ratio, lambda x: x.totChol / x.hdl * int(x.black))}
        else:
            return {'tot_chol_hdl_ratio': (self._tot_chol_hdl_ratio, lambda person: person._totChol[-1] / person._hdl[-1]),
                    'black_race_x_tot_chol_hdl_ratio': (self._black_race_x_tot_chol_hdl_ratio, lambda person: person._totChol[-1] / person._hdl[-1] * int(person._black))}

    def get_one_year_linear_predictor(self, person, vectorized=False):
        return super(ASCVDOutcomeModel, self).estimate_next_risk_vectorized(
            person) if vectorized else super(ASCVDOutcomeModel, self).estimate_next_risk(person)

    def transform_to_ten_year_risk(self, linearRisk):
        return (1 / (1 + np.exp(-1 * linearRisk))) 
    
    # time is accounted for simply...
    # our model gives us a 10 year risk. yet, we want the risk for the next year, on average, which
    # given that a patient ages over time, is lower than the 10 year risk/10
    # so, we estimate the weighted average of the patient at 5 years younger and older than their current 
    # age. this doesn't perfectly reproduce the 10 year risk, but its within 10%.
    # we can be more precise by building an average of the risk over all 10 years (close to within 1%)
    # but, that is computationally intense and this seems like a resonable compromise
    def get_risk_for_person(self, person, years, vectorized=False):
        linearRisk = self.get_one_year_linear_predictor(person, vectorized)
        fiveYearLinearAgeChange = self.parameters["lagAge"] * 5
        linearRiskMinusFiveYears = linearRisk - fiveYearLinearAgeChange
        linearRiskPlusFiveYears = linearRisk + fiveYearLinearAgeChange
        #print(f"linear risk {linearRisk:.5f} five year change: {fiveYearLinearAgeChange:.5f}, transform plus: {self.transform_to_ten_year_risk(linearRiskPlusFiveYears):.5f}")

        return (self.transform_to_ten_year_risk(linearRiskPlusFiveYears) + self.transform_to_ten_year_risk(linearRiskMinusFiveYears))/20 * years

