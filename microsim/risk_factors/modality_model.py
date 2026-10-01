from microsim.risk_factors.modality import Modality

class ModalityPrevalenceModel:
    """This model is used currently to initialize the NHANES population.
    This is the most simple model...but it does not reflect the prevalence of modality in the population."""
    def __init__(self):
        pass
 
    def estimate_next_risk(self, person):
        #modality is stored as the Modality value string, matching the Kaiser data and modalityGroupMap
        return  Modality.NO.value

