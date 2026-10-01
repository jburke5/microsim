import unittest

from scipy.special import expit

from microsim.calibration import (
    fit_cv_prevalence,
    fit_stroke_prevalence,
    fit_prevalence_scaling_to_stroke_mi,
    _age_group_midpoint,
)
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.reference import Reference
from microsim.outcomes.cv_model import CVPrevalenceModel
from microsim.outcomes.stroke_partition_model import StrokePrevalenceModel
from microsim.risk_factors.gender import NHANESGender


class TestCalibrationFits(unittest.TestCase):
    """Pure-math tests: no population construction anywhere."""

    def assert_coefficients_close(self, fitted, baked, tol):
        for gender in NHANESGender:
            for name in ["Intercept", "age"]:
                self.assertAlmostEqual(fitted[gender][name], baked[gender][name], delta=tol)

    def test_cv_baked_coefficients_match_fit(self):
        #baked values are rounded to 4-5 decimals, hence the tolerance
        self.assert_coefficients_close(fit_cv_prevalence(), CVPrevalenceModel._coefficients, 1e-3)

    def test_stroke_baked_coefficients_match_fit(self):
        self.assert_coefficients_close(
            fit_stroke_prevalence(), StrokePrevalenceModel._coefficients, 1e-3
        )

    def test_cv_fit_reproduces_reference_rates(self):
        coefficients = fit_cv_prevalence()
        reference = Reference.prevalence[OutcomeType.CARDIOVASCULAR.value]
        for gender in NHANESGender:
            c = coefficients[gender]
            for group, rate in reference[gender.name.lower()].items():
                fitted = expit(c["Intercept"] + c["age"] * _age_group_midpoint(group))
                self.assertAlmostEqual(fitted, rate, delta=0.01)

    def test_stroke_fit_reproduces_reference_rates(self):
        #the realized stroke prevalence is the CV probability times the fitted conditional
        cvCoefficients = fit_cv_prevalence()
        strokeCoefficients = fit_stroke_prevalence(cvCoefficients=cvCoefficients)
        reference = Reference.prevalence[OutcomeType.STROKE.value]
        for gender in NHANESGender:
            cv, stroke = cvCoefficients[gender], strokeCoefficients[gender]
            for group, rate in reference[gender.name.lower()].items():
                mid = _age_group_midpoint(group)
                realized = expit(cv["Intercept"] + cv["age"] * mid) * expit(
                    stroke["Intercept"] + stroke["age"] * mid
                )
                self.assertAlmostEqual(realized, rate, delta=0.01)

    def test_baked_scalings_match_stroke_mi_fit(self):
        from microsim.outcomes.outcome_prevalence_model_repository import (
            DEFAULT_PREVALENCE_RISK_SCALING,
        )

        scalings = fit_prevalence_scaling_to_stroke_mi(verbose=False)
        for outcomeType in [OutcomeType.CARDIOVASCULAR, OutcomeType.STROKE]:
            self.assertAlmostEqual(
                scalings[outcomeType],
                DEFAULT_PREVALENCE_RISK_SCALING[outcomeType],
                delta=1e-3 * DEFAULT_PREVALENCE_RISK_SCALING[outcomeType],
            )

    def test_stroke_mi_scalings_improve_fit(self):
        import math

        from microsim.outcomes.cv_model import CVPrevalenceModel as CV
        from microsim.outcomes.stroke_partition_model import StrokePrevalenceModel as Stroke

        scalings = fit_prevalence_scaling_to_stroke_mi(verbose=False)

        def squared_error(sCv, sStroke):
            error = 0.
            for gender in NHANESGender:
                cv, stroke = CV._coefficients[gender], Stroke._coefficients[gender]
                strokeRates = Reference.prevalence[OutcomeType.STROKE.value][gender.name.lower()]
                miRates = Reference.prevalence[OutcomeType.MI.value][gender.name.lower()]
                for group in strokeRates:
                    mid = _age_group_midpoint(group)
                    pCv = expit(cv["Intercept"] + cv["age"] * mid + math.log(sCv))
                    q = expit(stroke["Intercept"] + stroke["age"] * mid + math.log(sStroke))
                    self.assertTrue(0. < pCv * q < 1.)
                    error += (pCv * q - strokeRates[group]) ** 2
                    error += (pCv * (1. - q) - miRates[group]) ** 2
            return error

        fitted = squared_error(
            scalings[OutcomeType.CARDIOVASCULAR], scalings[OutcomeType.STROKE]
        )
        self.assertLess(fitted, squared_error(1., 1.))

    def test_stroke_rate_above_cv_prevalence_raises(self):
        #a conditional probability above 1 is unfittable and must fail loudly
        impossibleCv = {
            gender: {"Intercept": -10.0, "age": 0.0} for gender in NHANESGender
        }
        with self.assertRaises(ValueError):
            fit_stroke_prevalence(cvCoefficients=impossibleCv)


if __name__ == "__main__":
    unittest.main()
