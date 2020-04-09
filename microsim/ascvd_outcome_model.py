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

    def get_manual_parameters(self):
        return {'tot_chol_hdl_ratio': (self._tot_chol_hdl_ratio, lambda person : person._totChol[-1] / person._hdl[-1]),
            'black_race_x_tot_chol_hdl_ratio': (self._black_race_x_tot_chol_hdl_ratio, lambda person : person._totChol[-1] / person._hdl[-1] * int(person._black))}


    # TODO : need to figure out how to account fo rtime...which may be trikcy
    def get_risk_for_person(self, person, years):
        linearRisk = super(ASCVDOutcomeModel, self).estimate_next_risk(person)

        return (1 / (1 + np.exp(-1 * linearRisk))) * years / 10
