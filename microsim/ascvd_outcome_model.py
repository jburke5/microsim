import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.treatment import TreatmentStrategiesType

# https://annals.org/aim/fullarticle/2683613/[XSLTImagePath]

class ASCVDOutcomeModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio):

        super().__init__(regression_model, False)
        self._tot_chol_hdl_ratio = tot_chol_hdl_ratio
        self._black_race_x_tot_chol_hdl_ratio = black_race_x_tot_chol_hdl_ratio

    def get_manual_parameters(self):
        return {
                "tot_chol_hdl_ratio": (
                    self._tot_chol_hdl_ratio,
                    lambda person: person._totChol[-1] / person._hdl[-1],
                ),
                "black_race_x_tot_chol_hdl_ratio": (
                    self._black_race_x_tot_chol_hdl_ratio,
                    lambda person: person._totChol[-1] / person._hdl[-1] * int(person._black),
                ),
            }

    def get_intercept_change_for_person(self, person, interceptChangeFor1bpMedsAdded):
        '''Returns a constant factor that is added to the risk for person to reflect the
        adjusted risk when a person is under treatment.'''
        tst = TreatmentStrategiesType.BP.value
        if "bpMedsAdded" in person._treatmentStrategies[tst]:
            bpMedsAdded = person._treatmentStrategies[tst]['bpMedsAdded']
            interceptChange = bpMedsAdded * interceptChangeFor1bpMedsAdded
        else:
            interceptChange = 0
        return interceptChange

    def get_one_year_linear_predictor(self, person, interceptChangeFor1bpMedsAdded=0):
        return super(ASCVDOutcomeModel, self).estimate_next_risk(person) + self.get_intercept_change_for_person(person, interceptChangeFor1bpMedsAdded)

    def transform_to_ten_year_risk(self, linearRisk):
        # bound the calculation to avoid over/under-flow errors
        if linearRisk<-10:
            return 0.
        elif linearRisk>10:
            return 1.
        else:
            return 1 / (1 + np.exp(-1 * linearRisk))

    # time is accounted for simply...
    # our model gives us a 10 year risk. yet, we want the risk for the next year, on average, which
    # given that a patient ages over time, is lower than the 10 year risk/10
    # so, we estimate the weighted average of the patient at 5 years younger and older than their current
    # age. this doesn't perfectly reproduce the 10 year risk, but its within 10%.
    # we can be more precise by building an average of the risk over all 10 years (close to within 1%)
    # but, that is computationally intense and this seems like a resonable compromise
    def get_risk_for_person(self, person, rng, years, interceptChangeFor1bpMedsAdded=0): #rng is included here for compatibility with other get_risk_for_person methods
        linearRisk = self.get_one_year_linear_predictor(person, interceptChangeFor1bpMedsAdded)
        # four years gets us to the middle of hte 10 year window because we're using the 1 year lagged age
        # for the baseline..
        fourYearLinearAgeChange = self.parameters["lagAge"] * 4
        linearRiskMinusFourYears = linearRisk - fourYearLinearAgeChange

        return (self.transform_to_ten_year_risk(linearRiskMinusFourYears)) / 10 * years
