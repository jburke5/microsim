import unittest

import pandas as pd

from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.gfr_equation import GFREquation
from microsim.risk_factors.race_ethnicity import RaceEthnicity

# the lookup tables the comparisons replaced
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


def table_gfr(gender, raceEthnicity, creatinine, age):
    female = gender == NHANESGender.FEMALE
    crThreshold = 0.7 if female else 0.9
    exponent = exponentForGenderCr.loc[
        (exponentForGenderCr["female"] == female)
        & (exponentForGenderCr["underThreshold"] == (creatinine <= crThreshold))
    ].iloc[0]["exponent"]
    constant = constantForRaceGender.loc[
        (constantForRaceGender["black"] == (raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK))
        & (constantForRaceGender["female"] == female)
    ].iloc[0]["constant"]
    return constant * (creatinine / crThreshold) ** exponent * 0.993**age


class TestGFREquation(unittest.TestCase):
    def test_matches_lookup_tables(self):
        gfr = GFREquation()
        for gender in NHANESGender:
            for race in RaceEthnicity:
                for creatinine in [0.3, 0.69, 0.7, 0.71, 0.89, 0.9, 0.91, 1.5, 4.0]:
                    for age in [18, 50, 90]:
                        self.assertEqual(
                            table_gfr(gender, race, creatinine, age),
                            gfr.get_gfr_for_person_attributes(gender, race, creatinine, age),
                        )


if __name__ == "__main__":
    unittest.main()
