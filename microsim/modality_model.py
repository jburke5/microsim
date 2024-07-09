from microsim.modality import Modality

class ModalityPrevalenceModel:
    def __init__(self):
        pass
 
    def estimate_next_risk(self, person):
        return  Modality.CT 

