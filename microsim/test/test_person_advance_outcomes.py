from microsim.person import Person
from microsim.gender import NHANESGender
from microsim.race_ethnicity import RaceEthnicity
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome import Outcome
from microsim.outcome import OutcomeType
from microsim.education import Education
from microsim.alcohol_category import AlcoholCategory
from microsim.smoking_status import SmokingStatus
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe,
)
from microsim.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository, 
                                                   CohortStaticRiskFactorModelRepository,
                                                   CohortDefaultTreatmentModelRepository)
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.test.outcome_models_repositories import *
from microsim.treatment import DefaultTreatmentsType
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.person_factory import PersonFactory
from microsim.initialization_repository import InitializationRepository
from microsim.population_factory import PopulationFactory

import unittest
import copy
import numpy as np
import pandas as pd

class TestPersonAdvanceOutcomes(unittest.TestCase):
    def setUp(self):
        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()
        xJoe = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 42.,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 140,
                               DynamicRiskFactorsType.DBP.value: 90,
                               DynamicRiskFactorsType.A1C.value: 5.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 200,
                               DynamicRiskFactorsType.BMI.value: 25.,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 45,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "joe"}, index=[0])
        self.joe = PersonFactory.get_nhanes_person(xJoe.iloc[0], initializationModelRepository)
        self.joe._afib = [False]

        self.joe_with_cv = self.joe.__deepcopy__()
        self.joe_with_cv._outcomes[OutcomeType.CARDIOVASCULAR] = [(self.joe_with_cv._age[-1], Outcome(OutcomeType.CARDIOVASCULAR, False))]

        self.joe_with_mi = self.joe.__deepcopy__()
        self.joe_with_mi._outcomes[OutcomeType.MI] = [(self.joe_with_mi._age[-1], Outcome(OutcomeType.MI, False))]
        self.joe_with_mi._outcomes[OutcomeType.CARDIOVASCULAR] = [(self.joe_with_mi._age[-1], Outcome(OutcomeType.CARDIOVASCULAR, False))]

        self.joe_with_stroke = self.joe.__deepcopy__()
        self.joe_with_stroke._outcomes[OutcomeType.STROKE] = [(self.joe_with_stroke._age[-1], Outcome(OutcomeType.STROKE, False))]
        self.joe_with_stroke._outcomes[OutcomeType.CARDIOVASCULAR] = [(self.joe_with_stroke._age[-1], Outcome(OutcomeType.CARDIOVASCULAR, False))]

        self.miPartitionModel = MIPartitionModel()

    #Q: the person advance method does not predict a future if person.is_dead
    #I do not see a usefulness for raise an error, you should be able to attempt to advance a dead person
    #but it should not do anything
    #Q: also, the user should not advance just risk factors or treatments, the code is designed for use
    #with the advance method

    #def test_dead_is_dead_advance_year(self):
    #    self.joe._alive[-1] = False
    #    with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance year"):
    #        self.joe.advance_year(None, None)

    #def test_dead_is_dead_advance_risk_factors(self):
    #    self.joe._alive[-1] = False
    #    with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance risk factors"):
    #        self.joe.advance_risk_factors(None)

    #def test_dead_is_dead_advance_outcomes(self):
    #    self.joe._alive[-1] = False
    #    with self.assertRaises(RuntimeError, msg="Person is dead. Can not advance outcomes"):
    #        self.joe.advance_outcomes(None)

    def test_will_have_fatal_mi(self):

        miPartitionModel = MIPartitionModel()
        miPartitionModel._mi_case_fatality = 1.0
        is_max_prob_mi_fatal = miPartitionModel.will_have_fatal_mi(self.joe)
        miPartitionModel._mi_case_fatality = 0.0
        is_min_prob_mi_fatal = miPartitionModel.will_have_fatal_mi(self.joe)        

        self.assertTrue(is_max_prob_mi_fatal)
        self.assertFalse(is_min_prob_mi_fatal)

    def test_fatal_mi_secondary_prob(self):

        miPartitionModel = MIPartitionModel()
        miPartitionModel._mi_case_fatality = 0.0
        miPartitionModel._mi_secondary_case_fatality = 1.0
        will_have_fatal_first_mi = miPartitionModel.will_have_fatal_mi(self.joe)
        will_have_fatal_second_mi = miPartitionModel.will_have_fatal_mi(self.joe_with_mi)

        self.assertFalse(will_have_fatal_first_mi)
        # even though the passed fatality rate is zero, it should be overriden by the
        # secondary rate given that joe had a prior MI
        self.assertTrue(will_have_fatal_second_mi)

    def test_fatal_stroke_secondary_prob(self):
      
        strokePartitionModel = StrokePartitionModel()
        strokePartitionModel._stroke_case_fatality = 0.0
        strokePartitionModel._stroke_secondary_case_fatality = 1.0
        will_have_fatal_first_stroke = strokePartitionModel.will_have_fatal_stroke(self.joe)
        will_have_fatal_second_stroke = strokePartitionModel.will_have_fatal_stroke(self.joe_with_stroke)

        self.assertFalse(will_have_fatal_first_stroke)
        # even though the passed fatality rate is zero, it shoudl be overriden by the
        # secondary rate given that joeclone had a prior stroke
        self.assertTrue(will_have_fatal_second_stroke)

    def test_will_have_fatal_stroke(self):

        strokePartitionModel = StrokePartitionModel()
        strokePartitionModel._stroke_case_fatality = 0.0
        is_min_prob_stroke_fatal = strokePartitionModel.will_have_fatal_stroke(self.joe)
        strokePartitionModel._stroke_case_fatality = 1.0
        is_max_prob_stroke_fatal = strokePartitionModel.will_have_fatal_stroke(self.joe)

        self.assertTrue(is_max_prob_stroke_fatal)
        self.assertFalse(is_min_prob_stroke_fatal)

    def test_has_mi_vs_stroke(self):
 
        miPartitionModel = MIPartitionModel()

        has_mi_with_cv_outcome_and_no_stroke = miPartitionModel.get_next_outcome(self.joe_with_cv)
        has_mi_with_cv_outcome_and_stroke = miPartitionModel.get_next_outcome(self.joe_with_stroke)

        self.assertTrue(has_mi_with_cv_outcome_and_no_stroke is not None)
        self.assertTrue(has_mi_with_cv_outcome_and_stroke is None)

    def test_advance_outcomes_fatal_mi(self):

        self.joe.advance(1, CohortDynamicRiskFactorModelRepository(), 
                                 CohortDefaultTreatmentModelRepository(), 
                                   AlwaysFatalMIThroughRate(),
                                   None)         

        self.assertTrue(self.joe.has_mi_during_simulation())
        self.assertFalse(self.joe.has_stroke_during_simulation())
        self.assertTrue(self.joe.is_dead)

    def test_advance_outcomes_fatal_stroke(self):

        self.joe.advance(1, CohortDynamicRiskFactorModelRepository(),     
                                 CohortDefaultTreatmentModelRepository(),
                                   AlwaysFatalStrokeThroughRate(),
                                   None)

        self.assertFalse(self.joe.has_mi_during_simulation())
        self.assertTrue(self.joe.has_stroke_during_simulation())
        self.assertTrue(self.joe.is_dead)

    def test_advance_outcomes_nonfatal_mi(self):

        self.joe.advance(1, CohortDynamicRiskFactorModelRepository(),     
                                 CohortDefaultTreatmentModelRepository(),
                                   AlwaysNonFatalMIThroughRate(),
                                   None)

        self.assertTrue(self.joe.has_mi_during_simulation())
        self.assertFalse(self.joe.has_stroke_during_simulation())

    def test_advance_outcomes_nonfatal_stroke(self):

        self.joe.advance(1, CohortDynamicRiskFactorModelRepository(), 
                                 CohortDefaultTreatmentModelRepository(),
                                   AlwaysNonFatalStrokeThroughRate(),
                                   None)

        self.assertFalse(self.joe.has_mi_during_simulation())
        self.assertTrue(self.joe.has_stroke_during_simulation())


if __name__ == "__main__":
    unittest.main()



