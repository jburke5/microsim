import unittest
import numpy as np
import pandas as pd
from microsim.person import Person
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import RaceEthnicity
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.population_factory import PopulationFactory
from microsim.person_factory import PersonFactory
from microsim.treatment import DefaultTreatmentsType

class TestCKDEquation(unittest.TestCase):
    def setUp(self):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        self.x_black_female_high_cr = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 52,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.HIGHSCHOOLGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0.8,
                               "name": "black_female_high_cr"}, index=[0])

        self._black_female_high_cr = PersonFactory.get_nhanes_person(self.x_black_female_high_cr.iloc[0], initializationModelRepository)
        self._black_female_high_cr._afib = [False]

        self.x_black_female_low_cr = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 52,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.HIGHSCHOOLGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0.4,
                               "name": "black_female_low_cr"}, index=[0])

        self._black_female_low_cr = PersonFactory.get_nhanes_person(self.x_black_female_low_cr.iloc[0], initializationModelRepository)
        self._black_female_low_cr._afib = [False]

        self.x_white_male_high_cr = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 52,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.HIGHSCHOOLGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 1.2,
                               "name": "white_male_high_cr"}, index=[0])

        self._white_male_high_cr = PersonFactory.get_nhanes_person(self.x_white_male_high_cr.iloc[0], initializationModelRepository)
        self._white_male_high_cr._afib = [False]

        self.x_white_male_low_cr = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 52,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_WHITE.value,
                               DynamicRiskFactorsType.SBP.value: 120,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: Person.convert_fasting_glucose_to_a1c(100),
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 150,
                               DynamicRiskFactorsType.BMI.value: 26.6,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 94,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                               StaticRiskFactorsType.EDUCATION.value: Education.HIGHSCHOOLGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0.1,
                               "name": "white_male_low_cr"}, index=[0])

        self._white_male_low_cr = PersonFactory.get_nhanes_person(self.x_white_male_low_cr.iloc[0], initializationModelRepository)
        self._white_male_low_cr._afib = [False]

    def testGFRs(self):
        blackFemaleHighCr = (
            166
            * (self._black_female_high_cr._creatinine[0] / 0.7) ** -1.209
            * 0.993 ** self._black_female_high_cr._age[-1]
        )
        self.assertAlmostEqual(blackFemaleHighCr, self._black_female_high_cr._gfr)

        blackFemaleLowCr = (
            166
            * (self._black_female_low_cr._creatinine[0] / 0.7) ** -0.329
            * 0.993 ** self._black_female_low_cr._age[-1]
        )
        self.assertAlmostEqual(blackFemaleLowCr, self._black_female_low_cr._gfr)

        whiteMaleHighCr = (
            141
            * (self._white_male_high_cr._creatinine[0] / 0.9) ** -1.209
            * 0.993 ** self._white_male_high_cr._age[-1]
        )
        self.assertAlmostEqual(whiteMaleHighCr, self._white_male_high_cr._gfr)

        whiteMaleLowCr = (
            141
            * (self._white_male_low_cr._creatinine[0] / 0.9) ** -0.411
            * 0.993 ** self._white_male_low_cr._age[-1]
        )
        self.assertAlmostEqual(whiteMaleLowCr, self._white_male_low_cr._gfr)


if __name__ == "__main__":
    unittest.main()
