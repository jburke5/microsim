from microsim.outcome import OutcomeType, Outcome
from microsim.outcome_model_repository import OutcomeModelRepository

class CVModelAlwaysNonFatal:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.CARDIOVASCULAR, False)
class CVModelRepositoryAlwaysNonFatal:
    def __init__(self):
        self._model = CVModelAlwaysNonFatal()
    def select_outcome_model_for_person(self, person):
        return self._model

class CVModelNever:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return None
class CVModelNeverRepository:
    def __init__(self):
        self._model = CVModelNever()
    def select_outcome_model_for_person(self, person):
        return self._model

class StrokePartitionModelAlwaysNonFatalRepository:
    def __init__(self):
        self._model = StrokePartitionModelAlwaysNonFatal()
    def select_outcome_model_for_person(self, person):
        return self._model
class StrokePartitionModelAlwaysNonFatal:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.STROKE, False)

class StrokePartitionModelAlwaysFatalRepository:
    def __init__(self):
        self._model = StrokePartitionModelAlwaysFatal()
    def select_outcome_model_for_person(self, person):
        return self._model
class StrokePartitionModelAlwaysFatal:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.STROKE, True)

class StrokePartitionModelNeverRepository:
    def __init__(self):
        self._model = StrokePartitionModelNever()
    def select_outcome_model_for_person(self, person):
        return self._model
class StrokePartitionModelNever:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return None

class MIPartitionModelAlwaysNonFatalRepository:
    def __init__(self):
        self._model = MIPartitionModelAlwaysNonFatal()
    def select_outcome_model_for_person(self, person):
        return self._model
class MIPartitionModelAlwaysNonFatal:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.MI, False)

class MIPartitionModelNeverRepository:
    def __init__(self):
        self._model = MIPartitionModelNever()
    def select_outcome_model_for_person(self, person):
        return self._model
class MIPartitionModelNever:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return None

class DementiaAlwaysModel:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.DEMENTIA, False)
class DementiaAlwaysModelRepository:
    def __init__(self):
        self._model = DementiaAlwaysModel()
    def select_outcome_model_for_person(self, person):
        return self._model

class AlwaysNonFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysNonFatalRepository()

class AlwaysFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysFatalRepository()

class AlwaysNonFatalMI(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelNeverRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelAlwaysNonFatalRepository()

class AlwaysDementia(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaAlwaysModelRepository()

