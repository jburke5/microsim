from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
import numpy as np

# will use the CKD-EPI equation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2763564/
# because it prediicts better in blacks, https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-017-0788-y
# Levey, A. S. et al. A New Equation to Estimate Glomerular Filtration Rate. Ann Intern Med 150,
# 604 (2009).


class GFREquation:
    def __init__(self):
        pass

    def get_gfr_for_person(self, person, wave=-1):
        return self.get_gfr_for_person_attributes(
            person._gender, person._raceEthnicity, person._creatinine[wave], person._age[wave]
        )

    def get_gfr_for_person_attributes(self, gender, raceEthnicity, creatinine, age):
        female = gender == NHANESGender.FEMALE
        crThreshold = 0.7 if female else 0.9
        # CKD-EPI exponent by gender and creatinine at or below the threshold
        if creatinine <= crThreshold:
            exponent = -0.329 if female else -0.411
        else:
            exponent = -1.209
        # CKD-EPI constant by race and gender
        if raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            constant = 166 if female else 163
        else:
            constant = 144 if female else 141

        # Q: creatinine and exponent are both negative and fractional...what do we return in this
        # case?
        if (
            (crThreshold < 0.001)
            | (creatinine / crThreshold < 0)
            | np.isnan(exponent)
            | np.isinf(exponent)
            | np.isnan(creatinine / crThreshold)
            | np.isinf(creatinine / crThreshold)
        ):
            print(
                f"thresholds: {crThreshold} constant: {constant} exponent: {exponent} "
                f"female: {female}, "
                f"black: {raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK}, cr: {creatinine}"
            )
        return constant * (creatinine / crThreshold) ** exponent * 0.993**age
