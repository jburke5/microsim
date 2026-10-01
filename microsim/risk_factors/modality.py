from enum import Enum


class Modality(Enum):
    CT = "ct"
    MR = "mr"
    NO = "no"


# Map to be used for the classification of person objects regarding modality.
# This serves as the categorical variable to be used later on with regression, as a covariate.
modalityGroupMap = {Modality.CT.value: 0, Modality.MR.value: 1, Modality.NO.value: 2}
