from enum import Enum

#Important: any risk factor must be listed as Static/Dynamic AND Categorical/Continuous

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
    RACE_ETHNICITY = "raceEthnicity" # RaceEthnicity
    EDUCATION = "education"          # Education
    GENDER = "gender"                # NHANESGender
    SMOKING_STATUS = "smokingStatus" # SmokingStatus
    MODALITY = "modality"  # Modality, originated from the Kaiser WMH work 

class CategoricalRiskFactorsType(Enum):
    RACE_ETHNICITY = "raceEthnicity" 
    EDUCATION = "education"          
    GENDER = "gender"                
    SMOKING_STATUS = "smokingStatus" 
    PVD = "pvd"
    ALCOHOL_PER_WEEK = "alcoholPerWeek"
    AFIB = "afib"
    ANY_PHYSICAL_ACTIVITY = "anyPhysicalActivity" 
    MODALITY = "modality"
    
class ContinuousRiskFactorsType(Enum):
    AGE = "age"   # int
    SBP = "sbp"   # int
    DBP = "dbp"   # int
    A1C = "a1c"   # float
    HDL = "hdl"   # int
    LDL = "ldl"   # int
    TRIG = "trig"  # int
    TOT_CHOL = "totChol"   # int
    BMI = "bmi"   #float
    WAIST = "waist"  # int, waist circumference in cm
    CREATININE = "creatinine" # float

