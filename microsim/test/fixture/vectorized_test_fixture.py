import unittest

from microsim.alcohol_category import AlcoholCategory
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.population import Population
from microsim.person import Person
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus


class VectorizedTestFixture(unittest.TestCase):
    """
    Provides Pandas `DataFrame`s suitable for testing vectorized code.
    """

    def setUpClass(self):
        self._people = [
            Person(
                age=71,
                gender=NHANESGender.MALE,
                raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
                sbp=144.667,
                dbp=52.6667,
                a1c=9.5,
                hdl=34,
                totChol=191,
                bmi=30.05,
                ldl=110.0,
                trig=128,
                waist=45,
                anyPhysicalActivity=0,
                education=Education.COLLEGEGRADUATE,
                smokingStatus=SmokingStatus.FORMER,
                alcohol=AlcoholCategory.NONE,
                antiHypertensiveCount=0,
                statin=0,
                otherLipidLoweringMedicationCount=0,
                initializeAfib=(lambda _: None),
            ),
        ]
        self._population = Population(self._people)
        self.population_dataframe = self._population.get_people_current_state_as_dataframe()
