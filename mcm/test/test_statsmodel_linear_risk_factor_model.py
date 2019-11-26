from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.regression_model import RegressionModel
from mcm.education import Education
from mcm.alcohol_category import AlcoholCategory


from mcm.person import Person

import unittest
import pandas as pd
import numpy as np
import statsmodels.formula.api as statsmodel


def initializeAfib(person):
    return None


class TestStatsModelLinearRiskFactorModel(unittest.TestCase):

    def setUp(self):
        popSize = 100
        age = np.random.normal(loc=70, scale=20, size=popSize)
        sbp = age * 1.05 + np.random.normal(loc=40, scale=30, size=popSize)
        df = pd.DataFrame({'age': age, 'sbp': sbp})
        simpleModel = statsmodel.ols(formula="sbp ~ age", data=df)
        self.simpleModelResultSM = simpleModel.fit()
        self.simpleModelResult = RegressionModel(
            self.simpleModelResultSM.params.to_dict(),
            self.simpleModelResultSM.bse.to_dict(),
            self.simpleModelResultSM.resid.mean(),
            self.simpleModelResultSM.resid.std())

        self.person = Person(age=80, gender=NHANESGender.MALE,
                             raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE, sbp=120,
                             dbp=80, a1c=5.5, hdl=50, totChol=200, bmi=27, ldl=90, trig=150,
                             waist=70, anyPhysicalActivity=0, education=Education.COLLEGEGRADUATE,
                             smokingStatus=SmokingStatus.NEVER, alcohol=AlcoholCategory.NONE,
                             antiHypertensiveCount=0,
                             statin=0, otherLipidLoweringMedicationCount=0,
                             initializeAfib=initializeAfib)

        self.people = [Person(age=80, gender=NHANESGender.MALE,
                              raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
                              sbp=bpinstance, dbp=80, a1c=5.5, hdl=50, totChol=200, bmi=27, ldl=90,
                              trig=150, waist=70, anyPhysicalActivity=0,
                              education=Education.COLLEGEGRADUATE,
                              smokingStatus=SmokingStatus.NEVER,  alcohol=AlcoholCategory.NONE,
                              antiHypertensiveCount=0,
                              statin=0, otherLipidLoweringMedicationCount=0,
                              initializeAfib=initializeAfib)
                       for bpinstance in sbp]
        for person in self.people:
            self.advancePerson(person)

        df2 = pd.DataFrame({'age': age, 'sbp': [person._sbp[-1] for person in self.people],
                            'meanSbp': [np.array(person._sbp).mean() for person in self.people]})

        self.meanModelResultSM = statsmodel.ols(formula="sbp ~ age + meanSbp", data=df2).fit()
        self.meanModelResult = RegressionModel(
            self.meanModelResultSM.params.to_dict(),
            self.meanModelResultSM.bse.to_dict(),
            self.meanModelResultSM.resid.mean(),
            self.meanModelResultSM.resid.std())

        df3 = pd.DataFrame({'age': age, 'sbp': [person._sbp[-1] for person in self.people],
                            'logMeanSbp':
                            [np.log(np.array(person._sbp).mean()) for person in self.people]})

        self.logMeanModelResultSM = statsmodel.ols(
            formula="sbp ~ age + logMeanSbp", data=df3).fit()
        self.logMeanModelResult = RegressionModel(
            self.logMeanModelResultSM.params.to_dict(),
            self.logMeanModelResultSM.bse.to_dict(),
            self.logMeanModelResultSM.resid.mean(),
            self.logMeanModelResultSM.resid.std())

        race = np.random.randint(1, 5, size=popSize)
        df4 = pd.DataFrame({'age': age, 'sbp': sbp, 'raceEthnicity': race})
        df4.raceEthnicity = df4.raceEthnicity.astype('category')
        self.raceModelResultSM = statsmodel.ols(
            formula="sbp ~ age + raceEthnicity", data=df4).fit()
        self.raceModelResult = RegressionModel(
            self.raceModelResultSM.params.to_dict(),
            self.raceModelResultSM.bse.to_dict(),
            self.raceModelResultSM.resid.mean(),
            self.raceModelResultSM.resid.std())

        dfMeanAndLag = pd.DataFrame({
            'age': age,
            'sbp': [person._sbp[-1] for person in self.people],
            'meanSbp': [np.array(person._sbp).mean() for person in self.people],
            'lagSbp': [person._sbp[-1] for person in self.people],
        })

        self.meanLagModelResultSM = statsmodel.ols(
            formula="sbp ~ age + meanSbp + lagSbp", data=dfMeanAndLag).fit()
        self.meanLagModelResult = RegressionModel(
            self.meanLagModelResultSM.params.to_dict(),
            self.meanLagModelResultSM.bse.to_dict(),
            self.meanLagModelResultSM.resid.mean(),
            self.meanLagModelResultSM.resid.std())

    def advancePerson(self, person):
        person._age.append(person._age[-1] + 1)
        person._dbp.append(person._dbp[-1])
        person._a1c.append(person._a1c[-1])
        person._hdl.append(person._hdl[-1])
        person._totChol.append(person._totChol[-1])
        person._bmi.append(person._bmi[-1])
        person._sbp.append(person._sbp[-1] * 0.8 + 0.2 *
                           np.random.normal(loc=120, scale=20, size=1)[0])

    def testSimpleModel(self):
        self.assertEqual(self.simpleModelResultSM.params['age'] * self.person._age[-1] +
                         self.simpleModelResultSM.params['Intercept'],
                         StatsModelLinearRiskFactorModel(
            self.simpleModelResult).estimate_next_risk(self.person))

    def testModelWithMeanParameter(self):
        testPerson = self.people[5]

        self.assertAlmostEqual(self.meanModelResultSM.params['age'] * testPerson._age[-1] +
                               self.meanModelResultSM.params['meanSbp'] *
                               np.array(testPerson._sbp).mean() +
                               self.meanModelResultSM.params['Intercept'],
                               StatsModelLinearRiskFactorModel(
            self.meanModelResult).estimate_next_risk(testPerson), 5)

    def testLagAndMean(self):
        testPerson = self.people[12]

        self.assertAlmostEqual(self.meanLagModelResultSM.params['age'] * testPerson._age[-1] +
                               self.meanLagModelResultSM.params['meanSbp'] *
                               np.array(testPerson._sbp).mean() +
                               self.meanLagModelResultSM.params['lagSbp'] * testPerson._sbp[-1] +
                               self.meanLagModelResultSM.params['Intercept'],
                               StatsModelLinearRiskFactorModel(
            self.meanLagModelResult).estimate_next_risk(testPerson), 5)

    def testModelWithLogMeanParameter(self):
        testPerson = self.people[10]
        self.assertAlmostEqual(self.logMeanModelResultSM.params['age'] * testPerson._age[-1] +
                               self.logMeanModelResultSM.params['logMeanSbp'] *
                               np.log(np.array(testPerson._sbp).mean()) +
                               self.logMeanModelResultSM.params['Intercept'],
                               StatsModelLinearRiskFactorModel(
            self.logMeanModelResult).estimate_next_risk(testPerson), 5)

    def testModelWithCategoricalParameter(self):
        testPerson = self.people[21]
        testRace = testPerson._raceEthnicity
        self.assertAlmostEqual(self.raceModelResultSM.params['age'] * testPerson._age[-1] +
                               self.raceModelResultSM.params['raceEthnicity[T.' +
                                                             str(int(testRace)) + ']'] +
                               self.raceModelResultSM.params['Intercept'],
                               StatsModelLinearRiskFactorModel(
            self.raceModelResult).estimate_next_risk(testPerson), 5)
