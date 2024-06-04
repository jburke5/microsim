from microsim.outcome import OutcomeType, Outcome
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.stroke_partition_model import StrokePartitionModel
from microsim.mi_partition_model import MIPartitionModel

class NonCVModelNever:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return None
class NonCVModelNeverRepository:
    def __init__(self):
        self._model = NonCVModelNever()
    def select_outcome_model_for_person(self, person):
        return self._model

class NonCVModelAlways:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.NONCARDIOVASCULAR, True, False)
class NonCVModelAlwaysRepository:
    def __init__(self):
        self._model = NonCVModelAlways()
    def select_outcome_model_for_person(self, person):
        return self._model

class NonCVModelAlwaysOver50:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.NONCARDIOVASCULAR, True, False) if person._age[-1]>50 else None
class NonCVModelAlwaysOver50Repository:
    def __init__(self):
        self._model = NonCVModelAlwaysOver50()
    def select_outcome_model_for_person(self, person):
        return self._model

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

class CVModelAlwaysFatal:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.CARDIOVASCULAR, True)
class CVModelRepositoryAlwaysFatal:
    def __init__(self):
        self._model = CVModelAlwaysFatal()
    def select_outcome_model_for_person(self, person):
        return self._model

class CVModelAlwaysFatalOver50:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.CARDIOVASCULAR, True) if person._age[-1]>50 else None
class CVModelRepositoryAlwaysFatalOver50:
    def __init__(self):
        self._model = CVModelAlwaysFatalOver50()
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

class StrokePartitionModelAlwaysFatalThroughRateRepository:
    def __init__(self):
        self._model = StrokePartitionModelAlwaysFatalThroughRate()
    def select_outcome_model_for_person(self, person):
        return self._model
class StrokePartitionModelAlwaysFatalThroughRate(StrokePartitionModel):
    def __init__(self):
        super().__init__()
        self._stroke_case_fatality = 1.0
    def get_next_outcome(self, person):
        fatal = self.will_have_fatal_stroke(person)
        return Outcome(OutcomeType.STROKE, fatal)

class StrokePartitionModelAlwaysNonFatalThroughRateRepository:
    def __init__(self):
        self._model = StrokePartitionModelAlwaysNonFatalThroughRate()
    def select_outcome_model_for_person(self, person):
        return self._model
class StrokePartitionModelAlwaysNonFatalThroughRate(StrokePartitionModel):
    def __init__(self):
        super().__init__()
        self._stroke_case_fatality = 0.0
    def get_next_outcome(self, person):
        fatal = self.will_have_fatal_stroke(person)
        return Outcome(OutcomeType.STROKE, fatal)

class MIPartitionModelAlwaysNonFatalThroughRateRepository:
    def __init__(self):
        self._model = MIPartitionModelAlwaysNonFatalThroughRate()
    def select_outcome_model_for_person(self, person):
        return self._model
class MIPartitionModelAlwaysNonFatalThroughRate(MIPartitionModel):
    def __init__(self):
        super().__init__()
        self._mi_case_fatality = 0.0
    def get_next_outcome(self, person):
        fatal = self.will_have_fatal_mi(person)
        return Outcome(OutcomeType.MI, fatal)

class MIPartitionModelAlwaysFatalThroughRateRepository:
    def __init__(self):
        self._model = MIPartitionModelAlwaysFatalThroughRate()
    def select_outcome_model_for_person(self, person):
        return self._model
class MIPartitionModelAlwaysFatalThroughRate(MIPartitionModel):
    def __init__(self):
        super().__init__()
        self._mi_case_fatality = 1.0
    def get_next_outcome(self, person):
        fatal = self.will_have_fatal_mi(person)
        return Outcome(OutcomeType.MI, fatal)

class StrokePartitionModelAlwaysFatalOver50Repository:
    def __init__(self):
        self._model = StrokePartitionModelAlwaysFatalOver50()
    def select_outcome_model_for_person(self, person):
        return self._model
class StrokePartitionModelAlwaysFatalOver50:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return Outcome(OutcomeType.STROKE, True) if person._age[-1]>50 else None

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

class DementiaNeverModel:
    def __init__(self):
        pass
    def get_next_outcome(self, person):
        return None
class DementiaNeverModelRepository:
    def __init__(self):
        self._model = DementiaNeverModel()
    def select_outcome_model_for_person(self, person):
        return self._model

class AlwaysNonFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysNonFatalRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysNonFatalStrokeThroughRate(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysNonFatalThroughRateRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysFatalRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysFatalStrokeThroughRate(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysFatalThroughRateRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysNonFatalMI(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelNeverRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelAlwaysNonFatalRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysNonFatalMIThroughRate(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelNeverRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelAlwaysNonFatalThroughRateRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysFatalMIThroughRate(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysFatal()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelNeverRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelAlwaysFatalThroughRateRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()

class AlwaysDementia(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaAlwaysModelRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()

class NoOutcome(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelNeverRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()

class AgeOver50CausesNonCVMortality(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelNeverRepository()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelAlwaysOver50Repository()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelNeverRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()

class NonFatalStrokeAndNonCVMortality(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysNonFatal()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelAlwaysRepository()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysFatalRepository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()

class AgeOver50CausesFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super().__init__()
        self._repository[OutcomeType.CARDIOVASCULAR] = CVModelRepositoryAlwaysFatalOver50()
        self._repository[OutcomeType.DEMENTIA] = DementiaNeverModelRepository()
        self._repository[OutcomeType.NONCARDIOVASCULAR] = NonCVModelNeverRepository()
        self._repository[OutcomeType.STROKE] = StrokePartitionModelAlwaysFatalOver50Repository()
        self._repository[OutcomeType.MI] = MIPartitionModelNeverRepository()



