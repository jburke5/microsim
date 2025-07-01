import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import RaceEthnicity
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.treatment import TreatmentStrategiesType
from microsim.outcome import OutcomeType
from microsim.modality import Modality

# https://annals.org/aim/fullarticle/2683613/[XSLTImagePath]

class ASCVDOutcomeModel(StatsModelLinearRiskFactorModel):
    def __init__(self, regression_model, tot_chol_hdl_ratio, black_race_x_tot_chol_hdl_ratio, wmhSpecific=True):

        super().__init__(regression_model, False)
        self._tot_chol_hdl_ratio = tot_chol_hdl_ratio
        self._black_race_x_tot_chol_hdl_ratio = black_race_x_tot_chol_hdl_ratio
        self.wmhSpecific=wmhSpecific

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
        lp = super().estimate_next_risk(person) + self.get_intercept_change_for_person(person, interceptChangeFor1bpMedsAdded)
        lp += self.get_scd_term(person) 
        return lp

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

    def get_risk_components_for_person(self, person, rng, years, interceptChangeFor1bpMedsAdded=0):
        '''This function returns the silent cerebrovascular disease component of the ASCVD risk
        and the ascvd risk without the scd component.
        This function utilizes the get_risk_for_person functionality and the get_one_year_linear_predictor functionality.'''
        lpMinusScd = super().estimate_next_risk(person) + self.get_intercept_change_for_person(person, interceptChangeFor1bpMedsAdded)
        lpScd = self.get_scd_term(person)
        
        fourYearLinearAgeChange = self.parameters["lagAge"] * 4

        lpMinusScdMinusFourYears = lpMinusScd - fourYearLinearAgeChange
        lpScdMinusFourYears = lpScd - fourYearLinearAgeChange
        
        return {"riskMinusScd": (self.transform_to_ten_year_risk(lpMinusScdMinusFourYears)) / 10 * years,
                "riskScd": (self.transform_to_ten_year_risk(lpScdMinusFourYears)) / 10 * years}

    def get_scd_term(self, person):
        '''This is the contribution to the one year linear predictor due to the silent cerebrovascular disease (scd).
        The WMH hazard ratios were taken from the Wang2024 paper. 
        The SBI hazard ratios were taken from the Kent2021 paper.
        Scaling factors were found by optimizing the 4 year microsim stroke rates against the published stroke rates (which have 
        a follow up of around 4 years.''' 
        if not person._modality == Modality.NO.value: #if there was a brain scan 
            if self.wmhSpecific:
                scdTerm = 0.645 #intercept change
                scalingMriSbi = 2.6 #scaling factors to the published hazard ratios so that I can use them in the ascvd logistic model
                scalingCtSbi = 3.8
                scalingCtWmh = 1.8
                window = len(person._age) #how many years since the brain scan
                severityUnknown=person.get_outcome_item_first(OutcomeType.WMH, "wmhSeverityUnknown", inSim=True),
                severity=person.get_outcome_item_first(OutcomeType.WMH, "wmhSeverity", inSim=True)
                if person._outcomes[OutcomeType.WMH][0][1].sbi:
                    if person._modality == Modality.MR.value:
                        if person._age[-1] < 65:
                            if window ==1:
                                scdTerm += scalingMriSbi * np.log(4.75)
                            elif window <=3:
                                scdTerm += scalingMriSbi * np.log(3.45)
                            elif window <=5:
                                scdTerm += scalingMriSbi * np.log(2.18)
                            elif window >5:
                                scdTerm += scalingMriSbi * np.log(1.99)
                        else:
                            if window ==1:
                                scdTerm += scalingMriSbi * np.log(3.)
                            elif window <=3:
                                scdTerm +=  scalingMriSbi * np.log(2.63)
                            elif window <=5:
                                scdTerm += scalingMriSbi * np.log(1.61)
                            elif window >5:
                                scdTerm += scalingMriSbi * np.log(1.43)
                    elif person._modality == Modality.CT.value:
                        if person._age[-1] < 65:
                            if window ==1:
                                scdTerm += scalingCtSbi * np.log(3.91)
                            elif window <=3:
                                scdTerm += scalingCtSbi * np.log(2.34)
                            elif window <=5:
                                scdTerm += scalingCtSbi * np.log(2.33)
                            elif window >5:
                                scdTerm += scalingCtSbi * np.log(2.01)
                        else:
                            if window ==1:
                                scdTerm += scalingCtSbi * np.log(2.47)
                            elif window <=3:
                                scdTerm += scalingCtSbi * np.log(1.79)
                            elif window <=5:
                                scdTerm += scalingCtSbi * np.log(1.71)
                            elif window >5:
                                scdTerm += scalingCtSbi * np.log(1.44)
                    else:
                        raise RuntimeError("Person has SBI but no modality")
                if person._outcomes[OutcomeType.WMH][0][1].wmh:
                    if person._modality == Modality.MR.value: #I did not need to optimize a scaling factor for WMH MRI, rates were close to published data
                        if severityUnknown:
                            scdTerm += np.log(1.89)
                        elif severity == WMHSeverity.MILD:
                            scdTerm += np.log(1.51)
                        elif severity == WMHSeverity.MODERATE:
                            scdTerm += np.log(2.33)
                        elif severity == WMHSeverity.SEVERE:
                            scdTerm += np.log(2.65)
                        elif severity == WMHSeverity.NO:
                            scdTerm += np.log(1.)
                    elif person._modality == Modality.CT.value:
                        if severityUnknown:
                            scdTerm += scalingCtWmh * np.log(2.40)
                        elif severity == WMHSeverity.MILD:
                            scdTerm += scalingCtWmh * np.log(2.15)
                        elif severity == WMHSeverity.MODERATE:
                            scdTerm += scalingCtWmh * np.log(3.01)
                        elif severity == WMHSeverity.SEVERE:
                            scdTerm += scalingCtWmh * np.log(3.23)
                        elif severity == WMHSeverity.NO:
                            scdTerm += scalingCtWmh * np.log(1.39)
                    else:
                        raise RuntimeError("Person has WMH but no modality")
            else:
                scdTerm = 1.25 #represents an average increase in the risk of the kaiser population without taking into account the WMH outcome
        else:
            scdTerm=0.
        return scdTerm            
