from microsim.modality import Modality

class ModalityPrevalenceModel:
    """This model is used currently to initialize the NHANES population.
    This is the most simple model...but it does not reflect the prevalence of modality in the population."""
    def __init__(self):
        pass
 
    def estimate_next_risk(self, person):
        return  Modality.NO.value 

