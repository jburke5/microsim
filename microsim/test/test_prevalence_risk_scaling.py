import math
import unittest
from unittest.mock import patch

from scipy.special import expit

from microsim.outcomes import outcome_prevalence_model_repository as opmr_module
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.outcome_prevalence_base import OutcomePrevalenceBase
from microsim.outcomes.outcome_prevalence_model_repository import (
    DEFAULT_PREVALENCE_RISK_SCALING,
    OutcomePrevalenceModelRepository,
)
from microsim.outcomes.epilepsy_model import EpilepsyPrevalenceModel
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.person.person_filter_factory import PersonFilterFactory


def _adults_filter():
    return PersonFilterFactory.get_person_filter(["adult"])


class _StubPrevalenceModel(OutcomePrevalenceBase):
    """Minimal subclass with a fixed linear predictor so we can exercise the base-class
    odds-shift formula without depending on real coefficients."""

    _outcomeType = OutcomeType.CARDIOVASCULAR

    def __init__(self, linearPredictor, riskScaling=1.0):
        self._lp = linearPredictor
        self._riskScaling = riskScaling

    def get_linear_predictor_for_person(self, person):
        return self._lp


class TestOutcomePrevalenceBaseOddsShift(unittest.TestCase):
    def test_default_scaling_is_a_no_op(self):
        model = _StubPrevalenceModel(linearPredictor=-1.0)
        self.assertAlmostEqual(model.get_risk_for_person(person=None), expit(-1.0))

    def test_scaling_above_one_increases_risk(self):
        model = _StubPrevalenceModel(linearPredictor=-1.0, riskScaling=2.0)
        expected = expit(-1.0 + math.log(2.0))
        self.assertAlmostEqual(model.get_risk_for_person(person=None), expected)
        self.assertGreater(model.get_risk_for_person(person=None), expit(-1.0))

    def test_scaling_below_one_decreases_risk(self):
        model = _StubPrevalenceModel(linearPredictor=-1.0, riskScaling=0.5)
        expected = expit(-1.0 + math.log(0.5))
        self.assertAlmostEqual(model.get_risk_for_person(person=None), expected)
        self.assertLess(model.get_risk_for_person(person=None), expit(-1.0))


class TestEpilepsyPrevalenceDirectMultiply(unittest.TestCase):
    """Epilepsy uses linear_predictor / 1000 (a rate, not a logit), so scaling is applied
    as a direct multiplier rather than an odds shift."""

    def _make_model_with_fixed_lp(self, riskScaling, lp):
        model = EpilepsyPrevalenceModel(riskScaling=riskScaling)
        model.get_linear_predictor_for_person = lambda person: lp
        return model

    def test_default_scaling_is_a_no_op(self):
        model = self._make_model_with_fixed_lp(riskScaling=1.0, lp=2.0)
        self.assertAlmostEqual(model.get_risk_for_person(person=None), 2.0 / 1000.)

    def test_scaling_multiplies_rate_directly(self):
        model = self._make_model_with_fixed_lp(riskScaling=3.0, lp=2.0)
        self.assertAlmostEqual(model.get_risk_for_person(person=None), 3.0 * 2.0 / 1000.)


class TestOutcomePrevalenceModelRepositoryDispatch(unittest.TestCase):
    """The dict passed to OutcomePrevalenceModelRepository must reach the per-outcome
    model instances as `_riskScaling`. Mirror image of the OutcomeModelRepository
    riskScaling dispatch test."""

    def test_useDefaults_false_yields_one_for_all_supported_outcomes(self):
        # useDefaults=False bypasses any module-level calibration, giving the pristine
        # 1.0-everywhere baseline regardless of what DEFAULT_PREVALENCE_RISK_SCALING holds.
        opmr = OutcomePrevalenceModelRepository(useDefaults=False)
        for outcomeType in (
            OutcomeType.CARDIOVASCULAR,
            OutcomeType.STROKE,
            OutcomeType.DEMENTIA,
            OutcomeType.EPILEPSY,
            OutcomeType.DIABETES,
            OutcomeType.CHRONIC_KIDNEY_DISEASE,
        ):
            inner = opmr._repository[outcomeType]._model
            self.assertEqual(1.0, inner._riskScaling, msg=f"{outcomeType} pristine")

    def test_dict_propagates_per_outcome(self):
        scaling = {
            OutcomeType.CARDIOVASCULAR: 1.5,
            OutcomeType.STROKE: 2.0,
            OutcomeType.DEMENTIA: 0.5,
            OutcomeType.EPILEPSY: 3.0,
            OutcomeType.DIABETES: 1.25,
            OutcomeType.CHRONIC_KIDNEY_DISEASE: 0.75,
        }
        opmr = OutcomePrevalenceModelRepository(riskScaling=scaling)
        for outcomeType, expected in scaling.items():
            inner = opmr._repository[outcomeType]._model
            self.assertEqual(expected, inner._riskScaling, msg=f"{outcomeType}")

    def test_caller_overrides_default_others_inherit_default(self):
        # Set CV explicitly; verify CV uses the caller's value (overriding any module-level
        # default) while other outcomes inherit whatever DEFAULT_PREVALENCE_RISK_SCALING
        # holds (falling back to 1.0 when not registered).
        opmr = OutcomePrevalenceModelRepository(
            riskScaling={OutcomeType.CARDIOVASCULAR: 4.0}
        )
        self.assertEqual(4.0, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)
        self.assertEqual(
            DEFAULT_PREVALENCE_RISK_SCALING.get(OutcomeType.DEMENTIA, 1.0),
            opmr._repository[OutcomeType.DEMENTIA]._model._riskScaling,
        )
        self.assertEqual(
            DEFAULT_PREVALENCE_RISK_SCALING.get(OutcomeType.EPILEPSY, 1.0),
            opmr._repository[OutcomeType.EPILEPSY]._model._riskScaling,
        )


class TestNhanesTrialDescriptionForwardsPrevalenceRiskScaling(unittest.TestCase):
    def test_unset_inherits_module_level_defaults(self):
        # When prevalenceRiskScaling is not passed, the trial description's OPMR should
        # reflect whatever calibration is baked into DEFAULT_PREVALENCE_RISK_SCALING for
        # each outcome (1.0 when no default is registered).
        desc = NhanesTrialDescription(sampleSize=10, duration=1, year=1999,
                                      personFilters=_adults_filter())
        opmr = desc.peopleArgs["outcomePrevalenceModelRepository"]
        for outcomeType in (
            OutcomeType.CARDIOVASCULAR,
            OutcomeType.STROKE,
            OutcomeType.DEMENTIA,
            OutcomeType.EPILEPSY,
            OutcomeType.DIABETES,
            OutcomeType.CHRONIC_KIDNEY_DISEASE,
        ):
            expected = DEFAULT_PREVALENCE_RISK_SCALING.get(outcomeType, 1.0)
            self.assertEqual(expected, opmr._repository[outcomeType]._model._riskScaling,
                             msg=f"{outcomeType}")

    def test_dict_reaches_inner_models(self):
        scaling = {OutcomeType.DEMENTIA: 2.5, OutcomeType.EPILEPSY: 1.5}
        desc = NhanesTrialDescription(sampleSize=10, duration=1, year=1999,
                                      personFilters=_adults_filter(),
                                      prevalenceRiskScaling=scaling)
        opmr = desc.peopleArgs["outcomePrevalenceModelRepository"]
        self.assertEqual(2.5, opmr._repository[OutcomeType.DEMENTIA]._model._riskScaling)
        self.assertEqual(1.5, opmr._repository[OutcomeType.EPILEPSY]._model._riskScaling)
        # Outcomes the caller didn't specify fall back to the module default (or 1.0).
        cvExpected = DEFAULT_PREVALENCE_RISK_SCALING.get(OutcomeType.CARDIOVASCULAR, 1.0)
        self.assertEqual(cvExpected,
                         opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)


class TestDefaultPrevalenceRiskScalingHook(unittest.TestCase):
    """DEFAULT_PREVALENCE_RISK_SCALING acts as a module-level calibration that the repository
    merges into the caller's riskScaling dict (caller wins per-key). useDefaults=False
    bypasses it entirely. Tests patch a stand-in defaults dict so they remain valid
    regardless of what's currently baked into the real module-level dict."""

    _FAKE_DEFAULTS = {
        OutcomeType.CARDIOVASCULAR: 1.30,
        OutcomeType.DEMENTIA: 0.80,
    }

    def test_defaults_apply_when_caller_omits_key(self):
        with patch.object(opmr_module, "DEFAULT_PREVALENCE_RISK_SCALING", self._FAKE_DEFAULTS):
            opmr = OutcomePrevalenceModelRepository()
        self.assertEqual(1.30, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)
        self.assertEqual(0.80, opmr._repository[OutcomeType.DEMENTIA]._model._riskScaling)
        # An outcome with no fake default and no caller override stays at 1.0.
        self.assertEqual(1.0, opmr._repository[OutcomeType.EPILEPSY]._model._riskScaling)

    def test_caller_riskScaling_overrides_default_per_key(self):
        with patch.object(opmr_module, "DEFAULT_PREVALENCE_RISK_SCALING", self._FAKE_DEFAULTS):
            opmr = OutcomePrevalenceModelRepository(
                riskScaling={OutcomeType.CARDIOVASCULAR: 2.0}
            )
        # Caller-supplied key wins.
        self.assertEqual(2.0, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)
        # Non-overridden default still applies.
        self.assertEqual(0.80, opmr._repository[OutcomeType.DEMENTIA]._model._riskScaling)

    def test_useDefaults_false_bypasses_module_defaults(self):
        with patch.object(opmr_module, "DEFAULT_PREVALENCE_RISK_SCALING", self._FAKE_DEFAULTS):
            opmr = OutcomePrevalenceModelRepository(useDefaults=False)
        for outcomeType in (
            OutcomeType.CARDIOVASCULAR,
            OutcomeType.DEMENTIA,
            OutcomeType.STROKE,
            OutcomeType.EPILEPSY,
            OutcomeType.DIABETES,
            OutcomeType.CHRONIC_KIDNEY_DISEASE,
        ):
            self.assertEqual(1.0, opmr._repository[outcomeType]._model._riskScaling,
                             msg=f"{outcomeType}")

    def test_useDefaults_false_still_honors_caller_riskScaling(self):
        with patch.object(opmr_module, "DEFAULT_PREVALENCE_RISK_SCALING", self._FAKE_DEFAULTS):
            opmr = OutcomePrevalenceModelRepository(
                riskScaling={OutcomeType.STROKE: 1.7}, useDefaults=False
            )
        self.assertEqual(1.7, opmr._repository[OutcomeType.STROKE]._model._riskScaling)
        # Module default not applied since useDefaults=False.
        self.assertEqual(1.0, opmr._repository[OutcomeType.CARDIOVASCULAR]._model._riskScaling)


if __name__ == "__main__":
    unittest.main()
