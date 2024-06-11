from microsim.person_filter import PersonFilter

class PersonFilterFactory:

    @staticmethod
    def get_person_filter(addCommonFilters=True):
        pf = PersonFilter()
        if addCommonFilters:
            pf.add_filter("df", "lowSBPLimit", lambda x: x[DynamicRiskFactorsType.SBP.value]>126)
            pf.add_filter("df", "lowDBPLimit", lambda x: x[DynamicRiskFactorsType.DBP.value]>85)
            pf.add_filter("df", "highAntiHypertensivesLimit", lambda x: x[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]<=3)
            pf.add_filter("person", "highDemAndCVLimit", 
                                    lambda x: (CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x)< (0.00477) ))
            #self.add_filter("person", "highDemAndCVLimit", 
        #                    lambda x: ((DementiaModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x, years=1)< (9.3*10**(-5)) ) &
            #                           CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x)< (0.00477) ))
            #self.add_filter("person", "lowDemAndCVLimit", 
            #                lambda x: ((DementiaModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x, years=1)> (6.02*10**(-5)) ) &
            #                           CVModelRepository().select_outcome_model_for_person(x).get_risk_for_person(x)> (0.0038) ))
        return pf
