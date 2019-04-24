from mcm.outcome import OutcomeType
from mcm.outcome_model_type import OutcomeModelType
from mcm.outcome import Outcome
from mcm.statsmodel_linear_risk_factor_model import StatsModelLinearRiskFactorModel
from mcm.regression_model import RegressionModel


import numpy.random as npRand
import scipy.special as scipySpecial
import os
import json


class CVOutcomeDetermination:
    default_mi_case_fatality = 0.13
    default_stroke_case_fatality = 0.15
    default_secondary_prevention_multiplier = 1.0


    # fatal stroke probability estimated from our meta-analysis of BASIC, NoMAS, GCNKSS, REGARDS
    # fatal mi probability from: Wadhera, R. K., Joynt Maddox, K. E., Wang, Y., Shen, C., Bhatt,
    # D. L., & Yeh, R. W.
    # (2018). Association Between 30-Day Episode Payments and Acute Myocardial Infarction Outcomes
    # Among Medicare
    # Beneficiaries. Circ. Cardiovasc. Qual. Outcomes, 11(3), e46–9.
    # http://doi.org/10.1161/CIRCOUTCOMES.117.004397
    
    def __init__(self, outcome_model_repository,
                 mi_case_fatality=default_mi_case_fatality,
                 stroke_case_fatality=default_stroke_case_fatality,
                 secondary_prevention_multiplier=default_secondary_prevention_multiplier):
        self.mi_case_fatality = mi_case_fatality
        self.stroke_case_fatality = stroke_case_fatality
        self.secondary_prevention_multiplier = secondary_prevention_multiplier
        self.outcome_model_repository = outcome_model_repository

    def _will_have_cvd_event(self, ascvdProb):
        return npRand.uniform(size=1) < ascvdProb

    def _will_have_mi(self, person, manualMIProb=None):
        if manualMIProb is not None:
            return manualMIProb
        # if no manual MI probablity, estimate it from oru partitioned model
        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        model_spec_path = os.path.normpath(os.path.join(abs_module_path,
                                                        "./data/StrokeMIPartitionModelSpec.json"))
        with open(model_spec_path, 'r') as model_spec_file:
            model_spec = json.load(model_spec_file)
        strokePartitionModel = StatsModelLinearRiskFactorModel(RegressionModel(**model_spec))
        strokeProbability = scipySpecial.expit(strokePartitionModel.estimate_next_risk(person))

        return npRand.uniform(size=1) < (1 - strokeProbability)

    def _will_have_fatal_mi(self, fatalMIProb):
        return npRand.uniform(size=1) < fatalMIProb

    def _will_have_fatal_stroke(self, fatalStrokeProb):
        return npRand.uniform(size=1) < fatalStrokeProb

    def assign_outcome_for_person(self, person, years=1, manualStrokeMIProbability=None):
        if self._will_have_cvd_event(
                self.outcome_model_repository.get_risk_for_person(
                person,
                OutcomeModelType.CARDIOVASCULAR,
                years=1)):
            
            if self._will_have_mi(person, manualStrokeMIProbability):
                if self._will_have_fatal_mi(self.mi_case_fatality):
                    return Outcome(OutcomeType.MI, True)
                else:
                    return Outcome(OutcomeType.MI, False)
            else:
                if self._will_have_fatal_stroke(self.stroke_case_fatality):
                    return Outcome(OutcomeType.STROKE, True)
                else:
                    return Outcome(OutcomeType.STROKE, False)
