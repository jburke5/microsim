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
from microsim.person.bpcog_person_records import (
    BPCOGPersonRecord,
    BPCOGPersonDynamicRecord,
    BPCOGPersonEventRecord,
    BPCOGPersonStaticRecord,
)


def get_nhanes_imputed_dataset(year):
    dataset_path = get_absolute_datafile_path("fullyImputedDataset.dta")
    dataset = pd.read_stata(dataset_path)
    dataset = dataset.loc[dataset.year == year]
    return dataset


def build_prior_mi_event(prior_mi_age, current_age):
    if np.isnan(prior_mi_age) or prior_mi_age is None or prior_mi_age <= 1:
        return None
    if prior_mi_age == 99999:
        prior_mi_age = np.random.randint(18, current_age)
    prior_mi_age = prior_mi_age if prior_mi_age <= current_age else current_age
    return Outcome(OutcomeType.MI, False, age=prior_mi_age)


def build_prior_stroke_event(prior_stroke_age, current_age):
    if np.isnan(prior_stroke_age) or prior_stroke_age is None or prior_stroke_age <= 1:
        return None
    prior_stroke_age = prior_stroke_age if prior_stroke_age <= current_age else current_age
    return Outcome(OutcomeType.STROKE, False, age=prior_stroke_age)


def init_dataclass_from_dict(dataclass_type, init_values):
    """Creates a new dataclass instance from a subset of dict values"""
    field_names = set(f.name for f in dataclasses.fields(dataclass_type))
    init_kwargs = {k: init_values[k] for k in field_names}
    return dataclass_type(**init_kwargs)


class NHANESPersonRecordFactory:
    def __init__(self, init_random_effects, init_afib, init_gcp, init_qalys):
        self._init_random_effects = init_random_effects
        self._init_afib = init_afib
        self._init_gcp = init_gcp
        self._init_qalys = init_qalys

    @property
    def required_nhanes_column_names(self):
        return [
            "gender",
            "raceEthnicity",
            "education",
            "smokingStatus",
            "selfReportMIAge",
            "selfReportStrokeAge",
            "age",
            "meanSBP",
            "meanDBP",
            "a1c",
            "hdl",
            "ldl",
            "trig",
            "tot_chol",
            "bmi",
            "waist",
            "anyPhysicalActivity",
            "alcoholPerWeek",
            "antiHypertensive",
            "statin",
            "otherLipidLowering",
        ]

    def from_nhanes_dataset_row(
        self,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        selfReportMIAge,
        selfReportStrokeAge,
        age,
        meanSBP,
        meanDBP,
        a1c,
        hdl,
        ldl,
        trig,
        tot_chol,
        bmi,
        waist,
        anyPhysicalActivity,
        alcoholPerWeek,
        antiHypertensive,
        statin,
        otherLipidLowering,
    ):
        random_effects = self._init_random_effects()
        prior_mi = build_prior_mi_event(selfReportMIAge, age)
        prior_stroke = build_prior_stroke_event(selfReportStrokeAge, age)
        person_record = BPCOGPersonRecord(
            gender=NHANESGender(int(gender)),
            raceEthnicity=NHANESRaceEthnicity(int(raceEthnicity)),
            education=Education(int(education)),
            smokingStatus=SmokingStatus(int(smokingStatus)),
            randomEffectsGcp=random_effects["gcp"],
            alive=True,
            age=age,
            sbp=meanSBP,
            dbp=meanDBP,
            a1c=a1c,
            hdl=hdl,
            ldl=ldl,
            trig=trig,
            totChol=tot_chol,
            bmi=bmi,
            waist=waist,
            anyPhysicalActivity=anyPhysicalActivity,
            alcoholPerWeek=AlcoholCategory.get_category_for_consumption(alcoholPerWeek),
            antiHypertensiveCount=antiHypertensive,
            statin=statin,
            otherLipidLowerMedication=otherLipidLowering,
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
        person_record = dataclasses.replace(person_record, afib=afib, gcp=gcp)

        qalys = self._init_qalys(person_record)
        person_record = dataclasses.replace(person_record, qalys=qalys)

        return person_record


class NHANESPersonRecordLoader:
    """Loads PersonRecords from a sample of a given NHANES year."""

    def __init__(self, n, year, nhanes_person_record_factory, weights=None, random_state=None):
        self._nhanes_dataset = get_nhanes_imputed_dataset(year)
        self._n = n
        self._weights = weights
        self._random_state = random_state
        self._factory = nhanes_person_record_factory

    def get_num_persons(self):
        return self._n

    def iter_person_records(self):
        sample = self._nhanes_dataset.sample(
            self._n, weights=self._weights, random_state=self._random_state, replace=True
        )
        column_names = self._factory.required_nhanes_column_names
        for row_data in zip(*[sample[k] for k in column_names]):
            person_record = self._factory.from_nhanes_dataset_row(*row_data)
            yield person_record
