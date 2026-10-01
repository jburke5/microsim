import numpy as np
from scipy.optimize import least_squares
from scipy.special import logit, expit

from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.reference import Reference
from microsim.outcomes.cv_model import CVPrevalenceModel
from microsim.outcomes.stroke_partition_model import StrokePrevalenceModel
from microsim.risk_factors.gender import NHANESGender

'''Analytic fits of prevalence model coefficients to the external reference rates in
   Reference.prevalence. Workflow: update Reference.prevalence, run the fit, bake the
   returned values into the model's _coefficients; test_calibration enforces agreement.
   Distinct from PopulationFactory.calibrate_prevalence, which empirically root-finds a
   riskScaling on a constructed population.'''


def _age_group_midpoint(group):
    '''"50-54" -> 52.0, the low bound plus 2, the midpoint convention of the fits.'''
    lo, hi = group.split("-")
    return float(lo) + 2.


def _fit_logit_linear(ratesByAgeGroup):
    '''OLS of logit(rate) on age-group midpoints.
       Returns {"Intercept": ..., "age": ...}, the shape the prevalence models bake.'''
    ages = np.array([_age_group_midpoint(group) for group in ratesByAgeGroup])
    rates = np.array(list(ratesByAgeGroup.values()))
    slope, intercept = np.polyfit(ages, logit(rates), 1)
    return {"Intercept": float(intercept), "age": float(slope)}


def fit_cv_prevalence():
    '''Fits CVPrevalenceModel._coefficients to the GBD CV rates in Reference.prevalence,
       per gender.'''
    reference = Reference.prevalence[OutcomeType.CARDIOVASCULAR.value]
    return {gender: _fit_logit_linear(reference[gender.name.lower()])
            for gender in NHANESGender}


def fit_stroke_prevalence(cvCoefficients=None):
    '''Fits StrokePrevalenceModel._coefficients to the GBD stroke rates in
       Reference.prevalence, per gender. The fitted quantity is the probability of stroke
       conditional on prevalent CV, so each GBD rate is divided by the CV model probability
       first: cvCoefficients defaults to CVPrevalenceModel._coefficients and must be
       refit before this whenever the CV rates change.'''
    if cvCoefficients is None:
        cvCoefficients = CVPrevalenceModel._coefficients
    reference = Reference.prevalence[OutcomeType.STROKE.value]
    coefficients = dict()
    for gender in NHANESGender:
        rates = reference[gender.name.lower()]
        cv = cvCoefficients[gender]
        conditionalRates = dict()
        for group, rate in rates.items():
            cvProbability = expit(cv["Intercept"] + cv["age"] * _age_group_midpoint(group))
            q = rate / cvProbability
            if q >= 1.:
                raise ValueError(
                    f"stroke rate {rate} in {gender.name.lower()} {group} exceeds the CV "
                    f"prevalence {cvProbability:.4f}; conditional probability is unfittable."
                )
            conditionalRates[group] = q
        coefficients[gender] = _fit_logit_linear(conditionalRates)
    return coefficients


def fit_prevalence_scaling_to_stroke_mi(verbose=True):
    '''Finds the pair of prevalence riskScalings on CV and STROKE that makes the realized
       stroke AND MI prevalences jointly match the GBD rates in Reference.prevalence as well
       as two scalars can. The MI partition model is untouched (MI = prevalent CV without
       prevalent stroke), so stroke + MI = CV: under these scalings the seeded CV prevalence
       is the GBD stroke+MI level, not the GBD all-CVD level the CV coefficients were fit to.
       Closed-form since both models use only age and gender: least squares over
       (ln sCv, ln sStroke) on the 20 stroke and MI cells.
       Returns {OutcomeType.CARDIOVASCULAR: sCv, OutcomeType.STROKE: sStroke}.'''
    strokeReference = Reference.prevalence[OutcomeType.STROKE.value]
    miReference = Reference.prevalence[OutcomeType.MI.value]

    cells = []  #(cvLp, strokeLp, strokeTarget, miTarget) per gender and age group
    for gender in NHANESGender:
        cv = CVPrevalenceModel._coefficients[gender]
        stroke = StrokePrevalenceModel._coefficients[gender]
        strokeRates = strokeReference[gender.name.lower()]
        miRates = miReference[gender.name.lower()]
        for group in strokeRates:
            mid = _age_group_midpoint(group)
            cells.append((cv["Intercept"] + cv["age"] * mid,
                          stroke["Intercept"] + stroke["age"] * mid,
                          strokeRates[group], miRates[group], gender, group))

    def residuals(logScalings):
        logSCv, logSStroke = logScalings
        res = []
        for cvLp, strokeLp, strokeTarget, miTarget, _, _ in cells:
            pCv = expit(cvLp + logSCv)
            q = expit(strokeLp + logSStroke)
            res += [pCv * q - strokeTarget, pCv * (1. - q) - miTarget]
        return res

    solution = least_squares(residuals, x0=[0., 0.])
    sCv, sStroke = float(np.exp(solution.x[0])), float(np.exp(solution.x[1]))

    if verbose:
        print(f"sCv={sCv:.4f} sStroke={sStroke:.4f}")
        print(f"{'gender':>8}{'group':>8}{'stroke':>9}{'gbd':>8}{'mi':>9}{'gbd':>8}")
        for cvLp, strokeLp, strokeTarget, miTarget, gender, group in cells:
            pCv = expit(cvLp + solution.x[0])
            q = expit(strokeLp + solution.x[1])
            print(f"{gender.name.lower():>8}{group:>8}{pCv*q:>9.4f}{strokeTarget:>8.4f}"
                  f"{pCv*(1.-q):>9.4f}{miTarget:>8.4f}")

    return {OutcomeType.CARDIOVASCULAR: sCv, OutcomeType.STROKE: sStroke}
