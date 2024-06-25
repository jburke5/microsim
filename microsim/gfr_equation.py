import pandas as pd
from microsim.gender import NHANESGender
from microsim.race_ethnicity import RaceEthnicity
import numpy as np

# will use the CKD-EPI equation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2763564/
# because it prediicts better in blacks, https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-017-0788-y
#  Levey, A. S. et al. A New Equation to Estimate Glomerular Filtration Rate. Ann Intern Med 150, 604 (2009).


class GFREquation:
    exponentForGenderCr = pd.DataFrame(
        {
            "female": [True, True, False, False],
            "underThreshold": [True, False, True, False],
            "exponent": [-0.329, -1.209, -0.411, -1.209],
        }
    )

    constantForRaceGender = pd.DataFrame(
        {
            "black": [True, True, False, False],
            "female": [True, False, True, False],
            "constant": [166, 163, 144, 141],
        }
    )

    def __init__(self):
        pass

    def get_gfr_for_person(self, person, wave=-1):
        try:
            float(person._creatinine[-1])
            float(person._age[-1])
        except TypeError:
            print(f"pop index: {person._populationIndex} dfIndex: {person.dfIndex} cr: {person._creatinine}")
        return self.get_gfr_for_person_attributes(person._gender, person._raceEthnicity,
            person._creatinine[wave], person._age[wave])

    def get_gfr_for_person_attributes(self, gender, raceEthnicity, creatinine, age):
        crThreshold = 0.7 if gender == NHANESGender.FEMALE else 0.9
        
        exponent = GFREquation.exponentForGenderCr.loc[
            (GFREquation.exponentForGenderCr["female"] == (gender == NHANESGender.FEMALE))
            & (
                GFREquation.exponentForGenderCr["underThreshold"]
                == (creatinine <= crThreshold)
            )
        ].iloc[0]["exponent"]
        constant = GFREquation.constantForRaceGender.loc[
            (
                GFREquation.constantForRaceGender["black"]
                == (raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK)
            )
            & (
                GFREquation.constantForRaceGender["female"]
                == (gender == NHANESGender.FEMALE)
            )
        ].iloc[0]["constant"]

        #Q: creatinine and exponent are both negative and fractional...what do we return in this case?
        if (crThreshold < 0.001) | (creatinine/crThreshold<0) | np.isnan(exponent) | np.isinf(exponent) | np.isnan(creatinine / crThreshold) | np.isinf(creatinine / crThreshold):
            print(f"thresholds: {crThreshold} constant: {constant} exponent: {exponent} female: {gender==NHANESGender.FEMALE}, black: {raceEthnicity==NHANESRaceEthnicity.NON_HISPANIC_BLACK}, cr: {creatinine}")
        return (
            constant
            * (creatinine / crThreshold) ** exponent
            * 0.993 ** age
        )

