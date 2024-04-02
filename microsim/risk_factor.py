from enum import Enum

class DynamicRiskFactorsType(Enum):
    AGE = "age"   # int
    SBP = "sbp"   # int
    DBP = "dbp"   # int
    A1C = "a1c"   # float
    HDL = "hdl"   # int
    LDL = "ldl"   # int
    TRIG = "trig"  # int
    TOT_CHOL = "totChol"   # int
    BMI = "bmi"   #float
    ANY_PHYSICAL_ACTIVITY = "anyPhysicalActivity" # boolean
    AFIB = "afib"  # boolean 
    WAIST = "waist"  # int, waist circumference in cm
    ALCOHOL_PER_WEEK = "alcoholPerWeek"  # AlcoholCategory
    CREATININE = "creatinine" # float
    PVD = "pvd"  # boolean

class StaticRiskFactorsType(Enum):
    RACE_ETHNICITY = "raceEthnicity" # NHANESRaceEthnicity
    EDUCATION = "education"          # Education
    GENDER = "gender"                # NHANESGender
    SMOKING_STATUS = "smokingStatus" # SmokingStatus, TODO : change smoking status into a factor that changes over time

