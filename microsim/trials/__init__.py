"""Trial framework: experimental design, orchestration, and outcome assessment.

    from microsim.trials import Trial, TrialFactory, NhanesTrialDescription
"""

from microsim.trials.trial import Trial
from microsim.trials.trial_factory import TrialFactory
from microsim.trials.trial_type import TrialType
from microsim.trials.trial_description import (
    TrialDescription,
    NhanesTrialDescription,
    KaiserTrialDescription,
)
from microsim.trials.trial_outcome_assessor import AnalysisType, TrialOutcomeAssessor
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory

__all__ = [
    "Trial",
    "TrialFactory",
    "TrialType",
    "TrialDescription",
    "NhanesTrialDescription",
    "KaiserTrialDescription",
    "AnalysisType",
    "TrialOutcomeAssessor",
    "TrialOutcomeAssessorFactory",
]
