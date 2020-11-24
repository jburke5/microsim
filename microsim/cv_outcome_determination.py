from microsim.outcome import OutcomeType
from microsim.outcome_model_type import OutcomeModelType
from microsim.outcome import Outcome
from microsim.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from microsim.regression_model import RegressionModel
from microsim.data_loader import load_model_spec


import numpy.random as npRand
import numpy as np
import scipy.special as scipySpecial


class CVOutcomeDetermination:
    default_mi_case_fatality = 0.13
    default_secondary_mi_case_fatality = 0.13
    default_stroke_case_fatality = 0.15
    default_secondary_stroke_case_fatality = 0.15
    default_secondary_prevention_multiplier = 1.0

    # fatal stroke probability estimated from our meta-analysis of BASIC, NoMAS, GCNKSS, REGARDS
    # fatal mi probability from: Wadhera, R. K., Joynt Maddox, K. E., Wang, Y., Shen, C., Bhatt,
    # D. L., & Yeh, R. W.
    # (2018). Association Between 30-Day Episode Payments and Acute Myocardial Infarction Outcomes
    # Among Medicare
    # Beneficiaries. Circ. Cardiovasc. Qual. Outcomes, 11(3), e46–9.
    # http://doi.org/10.1161/CIRCOUTCOMES.117.004397

    def __init__(self,
                 mi_case_fatality=default_mi_case_fatality,
                 stroke_case_fatality=default_stroke_case_fatality,
                 mi_secondary_case_fatality=default_secondary_mi_case_fatality,
                 stroke_secondary_case_fatality=default_secondary_stroke_case_fatality,
                 secondary_prevention_multiplier=default_secondary_prevention_multiplier):
        self.mi_case_fatality = mi_case_fatality
        self.mi_secondary_case_fatality = mi_secondary_case_fatality,
        self.stroke_case_fatality = stroke_case_fatality
        self.stroke_secondary_case_fatality = stroke_secondary_case_fatality
        self.secondary_prevention_multiplier = secondary_prevention_multiplier

        model_spec = load_model_spec("StrokeMIPartitionModel")
        self.strokePartitionModel = StatsModelLinearRiskFactorModel(RegressionModel(**model_spec))

    def _will_have_cvd_event(self, ascvdProb):
        return npRand.uniform(size=1) < ascvdProb

    def _will_have_mi(self, person, outcome_model_repository, vectorized=False, manualMIProb=None):
        if manualMIProb is not None:
            return npRand.uniform(size=1) < manualMIProb
        # if no manual MI probablity, estimate it from oru partitioned model
        strokeProbability = self.get_stroke_probability(person, vectorized)

        return npRand.uniform(size=1) < (1 - strokeProbability)

    def get_stroke_probability(self, person, vectorized=False ):
        strokeProbability = 0
        if vectorized:
            strokeProbability = scipySpecial.expit(
                self.strokePartitionModel.estimate_next_risk_vectorized(person))
        else:
            strokeProbability = scipySpecial.expit(
                self.strokePartitionModel.estimate_next_risk(person))
        return strokeProbability

    def _will_have_fatal_mi(self, person, vectorized=False, overrideMIProb=None):
        fatalMIProb = overrideMIProb if overrideMIProb is not None else self.mi_case_fatality
        fatalProb = self.mi_secondary_case_fatality if self.has_prior_mi(person, vectorized) else fatalMIProb
        return npRand.uniform(size=1) < fatalProb

    def _will_have_fatal_stroke(self, person, vectorized=False, overrideStrokeProb=None):
        fatalStrokeProb = overrideStrokeProb if overrideStrokeProb is not None else self.stroke_case_fatality
        fatalProb = self.stroke_secondary_case_fatality if self.has_prior_stroke(person, vectorized) else fatalStrokeProb
        return npRand.uniform(size=1) < fatalProb

    def get_risk_for_person(self, outcome_model_repository,  person, vectorized):
        if vectorized:
            return outcome_model_repository.get_risk_for_person(person,
                                                                OutcomeModelType.CARDIOVASCULAR,
                                                                years=1, vectorized=True)
        else:
            return outcome_model_repository.get_risk_for_person(person,
                                                                OutcomeModelType.CARDIOVASCULAR,
                                                                years=1)

    def has_prior_stroke(self, person, vectorized):
        return (person.has_stroke_during_simulation() or person.has_stroke_prior_to_simulation()) if vectorized else person._stroke

    def has_prior_mi(self, person, vectorized):
        return (person.has_mi_during_simulation or person.has_mi_prior_to_simulation) if vectorized else person._mi
    
    def has_prior_stroke_mi(self, person, vectorized):
        return self.has_prior_stroke(person, vectorized) or self.has_prior_mi(person, vectorized)

    def assign_outcome_for_person(
            self,
            outcome_model_repository,
            person,
            vectorized=False,
            years=1,
            manualStrokeMIProbability=None):

        cvRisk = self.get_risk_for_person(outcome_model_repository, person, vectorized)

        if self.has_prior_stroke_mi(person, vectorized):
            cvRisk = cvRisk * self.secondary_prevention_multiplier

        return self.get_or_assign_outcomes(cvRisk, person, outcome_model_repository, vectorized, manualStrokeMIProbability)

    def get_or_assign_outcomes(self, cvRisk,  person, outcome_model_repository, vectorized, manualStrokeMIProbability):
        if self._will_have_cvd_event(cvRisk):
            if self._will_have_mi(person, outcome_model_repository, vectorized, manualStrokeMIProbability):
                return self.get_outcome(person, True, self._will_have_fatal_mi(person, vectorized),  vectorized)
            else:
                return self.get_outcome(person, False, self._will_have_fatal_stroke(person, vectorized), vectorized)
        elif vectorized:
            person.miNext = False
            person.strokeNext = False
            person.deadNext = False
            return person

    def get_outcome(self, person, mi, fatal, vectorized):
        if vectorized:
            person.miNext = mi
            person.strokeNext = not mi
            person.deadNext = fatal
            person.miFatal = mi and fatal
            person.strokeFatal = not mi and fatal
            person.ageAtFirstMI = person.age if np.isnan(person.ageAtFirstMI) else person.ageAtFirstMI 
            person.ageAtFirstStroke = person.age if np.isnan(person.ageAtFirstStroke) else person.ageAtFirstStroke 
            return person
        else:
            return Outcome(OutcomeType.MI if mi else OutcomeType.STROKE, fatal)
