from microsim.qaly_assignment_strategy import QALYAssignmentStrategy

class QALYModelRepository:
    def __init__(self):
        self._model = QALYAssignmentStrategy()
        
    def select_outcome_model_for_person(self, person):
        return self._model
