import dataclasses
import numpy as np
import pandas as pd
from microsim.alcohol_category import AlcoholCategory
from microsim.smoking_status import SmokingStatus
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.data_loader import get_absolute_datafile_path
from microsim.outcome import Outcome, OutcomeType
from microsim.person.bpcog_person_records import BPCOGPersonRecord


def get_nhanes_imputed_dataset(year):
    dataset_path = get_absolute_datafile_path("fullyImputedDataset.dta")
    dataset = pd.read_stata(dataset_path)
    dataset = dataset.loc[dataset.year == year]
    return dataset


def build_prior_mi_event(prior_mi_age, current_age):
    if prior_mi_age is None or prior_mi_age <= 1:
        return None
    if prior_mi_age == 99999:
        prior_mi_age = np.random.randint(18, current_age)
    prior_mi_age = prior_mi_age if prior_mi_age <= current_age else current_age
    return Outcome(OutcomeType.MI, False, age=prior_mi_age)


def build_prior_stroke_event(prior_stroke_age, current_age):
    if prior_stroke_age is None or prior_stroke_age <= 1:
        return None
    prior_stroke_age = prior_stroke_age if prior_stroke_age <= current_age else current_age
    return Outcome(OutcomeType.STROKE, False, age=prior_stroke_age)


class NHANESPersonRecordFactory:
    def __init__(self, outcome_model_repository, init_afib, init_gcp, init_qalys):
        self._outcome_model_repository = outcome_model_repository
        self._init_afib = init_afib
        self._init_gcp = init_gcp
        self._init_qalys = init_qalys

    def from_nhanes_dataset_row(self, row):
        random_effects = self._outcome_model_repository.get_random_effects()
        prior_mi = build_prior_mi_event(row.selfReportMIAge, row.age)
        prior_stroke = build_prior_stroke_event(row.selfReportStrokeAge)
        person_record = BPCOGPersonRecord(
            gender=NHANESGender(int(row.gender)),
            raceEthnicity=NHANESRaceEthnicity(int(row.raceEthnicity)),
            education=Education(int(row.education)),
            smokingStatus=SmokingStatus(int(row.smokingStatus)),
            randomEffectsGcp=random_effects["gcp"],
            alive=True,
            age=row.age,
            sbp=row.meanSBP,
            dbp=row.meanDBP,
            a1c=row.a1c,
            hdl=row.hdl,
            ldl=row.ldl,
            trig=row.trig,
            totChol=row.totChol,
            bmi=row.bmi,
            waist=row.waist,
            anyPhysicalActivity=row.anyPhysicalActivity,
            alcoholPerWeek=AlcoholCategory.get_category_for_consumption(row.alcoholPerWeek),
            antiHypertensiveCount=row.antiHypertensive,
            statin=row.statin,
            otherLipidLowerMedication=row.otherLipidLowering,
            bpMedsAdded=0,
            afib=False,
            qalys=0,
            gcp=0,
            mi=prior_mi,
            stroke=prior_stroke,
            dementia=None,
        )

        afib = self._init_afib(person_record)
        gcp = self._init_gcp(person_record)
        person_record = dataclasses.replace(person_record, {"afib": afib, "gcp": gcp})

        qalys = self._init_qalys(person_record)
        person_record = dataclasses.replace(person_record, {"qalys": qalys})

        return person_record


class NHANESPersonRecordLoader:
    """Loads PersonRecords from a sample of a given NHANES year."""

    def __init__(self, n, year, nhanes_person_record_factory, weights=None, random_seed=None):
        self._nhanes_dataset = get_nhanes_imputed_dataset(year)
        self._n = n
        self._weights = weights
        self._random_seed = random_seed
        self._factory = nhanes_person_record_factory

    def get_num_people(self):
        return self._n

    def get_all_people(self):
        sample = self._nhanes_dataset.sample(
            self._n, weights=self._weights, random_state=self._random_seed, replace=True
        )
        person_records = [
            self._factory.from_nhanes_dataset_row(row) for _, row in sample.iterrows()
        ]
        return person_records
