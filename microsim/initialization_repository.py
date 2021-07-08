from microsim.gcp_model import GCPModel
from microsim.qaly_assignment_strategy import QALYAssignmentStrategy


class InitializationRepository:
    def __init__(self):
        pass

    def get_initializers(self):
        return {
            "_gcp": GCPModel().get_risk_for_person,
            "_qalys": QALYAssignmentStrategy().get_next_qaly,
        }
