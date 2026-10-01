import numpy as np
import pandas as pd
from microsim.outcomes.cognition_outcome import CognitionOutcome
from microsim.outcomes.outcome import OutcomeType
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.a1c import convert_a1c_to_fasting_glucose
from microsim.person.person import Person
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType
from collections import OrderedDict


class GCPModel:
    def __init__(self, outcomeModelRepository=None):
        #Q why are we doing this? I am not sure how/if this was used in the past
        self._outcome_model_repository = outcomeModelRepository
        pass

    def generate_next_outcome(self, person):
        fatal = False
        gcp = self.get_risk_for_person(person, person._rng)
        selfReported = False
        return CognitionOutcome(fatal, selfReported, gcp)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person)

    #Q: I am not sure what the issue is here...
    # TODO — what do we need to do with the random intercept? shouls we take a draw per person and assign it?
    # if we don't do that there is going to be mroe change in cognitive trajectory per person that we'd expect...
    def calc_linear_predictor_for_patient_characteristics(
        self,
        yearsInSim,
        raceEthnicity,
        gender,
        baseAge,
        education,
        alcohol,
        smokingStatus,
        bmi,
        waist,
        totChol,
        meanSBP,
        anyAntiHpertensive,
        fastingGlucose,
        physicalActivity,
        afib,
        test=False,
    ):
        #reportingDict = {}
        xb = 55.6090
        #reportingDict['intercept'] = xb
        xb += yearsInSim * -0.2031
        #reportingDict['yearsInSim'] = xb - pd.Series(reportingDict.values()).sum()
        if raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -5.6818
            xb += yearsInSim * -0.00870
        #reportingDict['raceEthnicity'] = xb - pd.Series(reportingDict.values()).sum()
        if gender == NHANESGender.FEMALE:
            xb += 2.0863
            xb += yearsInSim * -0.06184
        #reportingDict['gender'] = xb - pd.Series(reportingDict.values()).sum()
        xb += -2.0109 * (baseAge - 65) / 10
        #reportingDict['baseAge'] = xb - pd.Series(reportingDict.values()).sum()
        xb += -0.1266 * yearsInSim * baseAge / 10
        #reportingDict['baseAgeYears'] = xb - pd.Series(reportingDict.values()).sum()
        # are we sure that the educatino categories align?
        if education == Education.LESSTHANHIGHSCHOOL:
            xb += -9.5559
        elif education == Education.SOMEHIGHSCHOOL:
            xb += -6.6495
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += -3.1954
        elif education == Education.SOMECOLLEGE:
            xb += -2.3795
        #reportingDict['educcation'] = xb - pd.Series(reportingDict.values()).sum()

        alcCoeffs = [0, 0.8071, 0.6943, 0.7706]
        xb += alcCoeffs[int(alcohol)]
        #reportingDict['alcohol'] = xb - pd.Series(reportingDict.values()).sum()

        if smokingStatus == SmokingStatus.CURRENT:
            xb += -1.1678
        #reportingDict['smoking'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (bmi - 26.6) * 0.1309
        #reportingDict['bmi'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (waist - 94) * -0.05754
        #reportingDict['waist'] = xb - pd.Series(reportingDict.values()).sum()
        # note...not 100% sure if this should be LDL vs. tot chol...
        xb += (totChol - 127) / 10 * 0.002690
        #reportingDict['totChol'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (meanSBP - 120) / 10 * -0.2663
        #reportingDict['meanSbp'] = xb - pd.Series(reportingDict.values()).sum()
        xb += (meanSBP - 120) / 10 * yearsInSim * -0.01953
        #reportingDict['sbpYears'] = xb - pd.Series(reportingDict.values()).sum()

        xb += anyAntiHpertensive * 0.04410
        #reportingDict['antiHypertensive'] = xb - pd.Series(reportingDict.values()).sum()
        xb += anyAntiHpertensive * yearsInSim * 0.01984
        #reportingDict['antiHypertensiveYears'] = xb - pd.Series(reportingDict.values()).sum()

        # need to turn off the residual for hte simulation...also need to make sure that we're correctly centered...
        xb += (fastingGlucose - 100) / 10 * -0.09362
        #reportingDict['glucose'] = xb - pd.Series(reportingDict.values()).sum()
        if physicalActivity:
            xb += 0.6065
        #reportingDict['activity'] = xb - pd.Series(reportingDict.values()).sum()
        if afib:
            xb += -1.6579
        #reportingDict['afib'] = xb - pd.Series(reportingDict.values()).sum()
        #reportingDict['totalYears'] = yearsInSim
        #reportingDict['meanSBPValue'] = meanSBP
        #reportingDict['antiHypertensiveValue'] = anyAntiHpertensive
        #reportingDict['finalXb'] = xb

        #if self._outcome_model_repository is not None:
        #    self._outcome_model_repository.report_result('gcp', reportingDict)
        return xb

    def get_risk_for_person(self, person, rng=None, years=1, test=False):
        if "gcp" not in list(person._randomEffects.keys()):
            person._randomEffects["gcp"] = person._rng.normal(0, 4.84)
        random_effect = person._randomEffects["gcp"]
        residual = 0 if test else rng.normal(0.38, 6.99)

        yearsInSim = person.get_years_in_simulation()

        #tst = TreatmentStrategiesType.WMD15.value
        #if "wmd15MedsAdded" in person._treatmentStrategies[tst]:
        #    wmd15MedsAdded = person._treatmentStrategies[tst]['wmd15MedsAdded']
        #    yearsInSim = yearsInSim * 0.2401 if wmd15MedsAdded>0 else yearsInSim #(1-0.15*2)^4

        #tst = TreatmentStrategiesType.WMD25.value
        #if "wmd25MedsAdded" in person._treatmentStrategies[tst]:
        #    wmd25MedsAdded = person._treatmentStrategies[tst]['wmd25MedsAdded']
        #    yearsInSim = yearsInSim * 0.0625 if wmd25MedsAdded>0 else yearsInSim #(1-0.25*2)^4


        linPred = 0
        linPred = self.calc_linear_predictor_for_patient_characteristics(
                yearsInSim=yearsInSim,
                raceEthnicity=person._raceEthnicity,
                gender=person._gender,
                baseAge=person._age[0],
                education=person._education,
                alcohol=AlcoholCategory.get_category_for_consumption(person._alcoholPerWeek[-1]),
                smokingStatus=person._smokingStatus,
                bmi=person._bmi[-1],
                waist=person._waist[-1],
                totChol=person._totChol[-1],
                meanSBP=np.array(person._sbp).mean(),
                anyAntiHpertensive=((person._antiHypertensiveCount[-1]>0) | person.is_in_bp_treatment),
                fastingGlucose=person.get_fasting_glucose(not test),
                physicalActivity=person._anyPhysicalActivity[-1],
                afib=person._afib[-1],
            )

        #tst = TreatmentStrategiesType.WMD15.value
        #if "wmd15MedsAdded" in person._treatmentStrategies[tst]:
        #    wmd15MedsAdded = person._treatmentStrategies[tst]['wmd15MedsAdded']
        #    linPred  = linPred * (1+0.0241) if wmd15MedsAdded>0 else linPred # x^4 = 1.15-0.05

        #THIS ONE IS WORKING WELL by itself..
        #tst = TreatmentStrategiesType.WMD20.value
        #if "wmd20MedsAdded" in person._treatmentStrategies[tst]:
        #    wmd20MedsAdded = person._treatmentStrategies[tst]['wmd20MedsAdded']
        #    linPred  = linPred * (1+0.0356) if wmd20MedsAdded>0 else linPred # solve x^4 = 1.2-0.05 for x

        #tst = TreatmentStrategiesType.WMD25.value
        #if "wmd25MedsAdded" in person._treatmentStrategies[tst]:
        #    wmd25MedsAdded = person._treatmentStrategies[tst]['wmd25MedsAdded']
        #    linPred  = linPred * (1+0.0466) if wmd25MedsAdded>0 else linPred # x^4 = 1.25-0.05

        risk = linPred + random_effect
        return risk + residual


# based on https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805003, Model M2
# the gcp stroke model will need to be adjusted, there seems to be a difference between
# the stroke population in the paper and the microsim stroke population, which leads to an increase in gcp after a stroke in microsim....
class GCPStrokeModel:
    def __init__(self, outcomeModelRepository=None):
        #Q why are we passing an outcome model repo here?
        self._outcome_model_repository = outcomeModelRepository

    def generate_next_outcome(self, person):
        fatal = False
        selfReported = False
        gcp = self.get_risk_for_person(person, person._rng)
        return CognitionOutcome(fatal, selfReported, gcp)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person)

    def calc_linear_predictor_for_patient_characteristics(
        self,
        ageAtLastStroke,
        yearsSinceStroke,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        #diabetestx, #simulation does not currently include diabetes treatment
        physicalActivity,
        alcoholPerWeek,
        meanBmiPrestroke,
        meanSBP,
        meanSBPPrestroke,
        meanLdlPrestroke,
        meanLdl,
        gfr,
        meanWaistPrestroke,
        meanFastingGlucose,
        meanFastingGlucosePrestroke,
        anyAntiHypertensive,
        anyLipidLowering,
        afib,
        mi,
        meanGCPPrestroke):

       #standardize some variables first, if a variable is ending on "med10" that meant it was centered and standardized by 10
       #initially I thought the centers were the actual means as described in the original publication, that was wrong
       #the centers used were included in an excel file that the group sent us via email
       ageAtLastStrokeS = (ageAtLastStroke-65.)/10.
       meanBmiPrestrokeS = (meanBmiPrestroke-25.)
       meanWaistPrestrokeS = (meanWaistPrestroke-100.)/10.
       meanSBPS = (meanSBP-130.)/10.
       meanFastingGlucoseS = (meanFastingGlucose-100.)/10.
       meanFastingGlucosePrestrokeS = (meanFastingGlucosePrestroke-100.)/10.
       meanGCPPrestrokeS = meanGCPPrestroke - 50.
       meanSBPPrestrokeS = (meanSBPPrestroke-130.)/10.
       meanLdlS = (meanLdl-93.)/10.
       meanLdlPrestrokeS = (meanLdlPrestroke - 93.)/10.

       xb = 51.9602                                                #Intercept
       xb += yearsSinceStroke * (-0.5249)                          #slope, t_gcp_stk
       xb += (-1.4919) * ageAtLastStrokeS                          #agemed10
       xb += (-0.1970) * ageAtLastStrokeS * yearsSinceStroke       #t_gcp_stk*agemed10
       if gender == NHANESGender.FEMALE:
           xb += 1.4858                                            #female0
           xb += yearsSinceStroke * (-0.2864)                      #t_gcp_stk*female0, change in the slope due to gender
       if raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
           xb += -1.5739                                           #black
       if education == Education.HIGHSCHOOLGRADUATE:
           xb += 0.9930                                            #educ2
       elif education == Education.SOMECOLLEGE:
           xb += -0.00267                                          #educ3
       elif education == Education.COLLEGEGRADUATE:
           xb += 0.5712                                            #educ4
       if smokingStatus == SmokingStatus.CURRENT:
           xb += 0.4707                                            #currsmoker
       if physicalActivity:
           xb += 1.0047                                            #physact
       if alcoholPerWeek == AlcoholCategory.ONETOSIX:              #alcperwk
           xb += 0.05502 * 3.5
       elif alcoholPerWeek == AlcoholCategory.SEVENTOTHIRTEEN:
           xb += 0.05502 * 10
       elif alcoholPerWeek == AlcoholCategory.FOURTEENORMORE:
           xb += 0.05502 * 17
       xb += -0.1372 * meanBmiPrestrokeS                           #bs_bmimed
       xb += 0.2726 * meanWaistPrestrokeS                          #bs_waistcmmed10
       xb += 0.2301 * meanSBPS                                     #sbpmed10
       xb += 0.04248 * meanSBPS * yearsSinceStroke                 #sbpmed10*t_gcp_stk
       if anyAntiHypertensive:
           xb += (-1.3711)                                         #htntx
           xb += yearsSinceStroke * (0.2271)                       #t_gcp_stk*htntx
       xb += 0.1562 * meanFastingGlucoseS                          #glucosefmed10
       xb += -0.04266 * meanFastingGlucoseS * yearsSinceStroke     #t_gcp_stk*glucosefme
       xb += -0.02933 * meanFastingGlucosePrestrokeS               #bs_glucosefmed10
       if afib:
           xb += -1.5329                                           #Hxafib
       if mi:
           xb += 0.4470                                            #HxMI
       #if diabetestx:                                                #currently simulation does not include diabetes medication
       #    xb += -1.4601                                           #diabetestx
       #    xb += (-0.03788) * yearsSinceStroke                     #t_gcp_stk*diabetestx
       xb += 0.01751 * gfr                                         #gfr
       xb += 0.6632 * meanGCPPrestrokeS                            #bs_fgcpmed
       xb += -0.2535 * meanSBPPrestrokeS                           #bs_sbpstkcogmed10
       if anyLipidLowering:
           xb += -0.7570                                           #choltx
           xb += 0.1035 * yearsSinceStroke                         #t_gcp_stk*choltx
       xb += -0.1866 * meanLdlPrestrokeS                           #bs_cholldlmed10
       xb += -0.09122 * meanLdlS                                   #cholldlmed10
       xb += 0.007825 * meanLdlS * yearsSinceStroke                #t_gcp_stk*cholldlmed
       #weighted average to account for cohort (aric,chs,fos,regards-assumed to be baseline)
       xb += (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.)
       return xb

    def get_risk_for_person(self, person, rng=None, years=1, test=False):

        if "gcpStroke" not in list(person._randomEffects.keys()):
            person._randomEffects["gcpStroke"] = person._rng.normal(0., 3.90)
        random_effect = person._randomEffects["gcpStroke"]

        if "gcpStrokeSlope" not in list(person._randomEffects.keys()):
            person._randomEffects["gcpStrokeSlope"] = person._rng.normal(0., 0.264)
        random_effect_slope = person._randomEffects["gcpStrokeSlope"]
        residual = 0 if test else rng.normal(0, 6.08)

        linPred = 0
        ageAtLastStroke=person.get_age_at_last_outcome(OutcomeType.STROKE)
        yearsSinceStroke=person._age[-1]-ageAtLastStroke
        personGCP = list(map(lambda x: x[1].gcp, person._outcomes[OutcomeType.COGNITION]))
        waveAtLastStroke=person.get_wave_for_age(ageAtLastStroke)
        linPred = self.calc_linear_predictor_for_patient_characteristics(
                ageAtLastStroke=ageAtLastStroke,
                yearsSinceStroke=yearsSinceStroke,
                gender=person._gender,
                raceEthnicity=person._raceEthnicity,
                education=person._education,
                smokingStatus=person._smokingStatus,
                #diabetes=person.has_diabetestx(),
                physicalActivity=person._anyPhysicalActivity[-1],
                alcoholPerWeek=AlcoholCategory.get_category_for_consumption(person._alcoholPerWeek[-1]),
                meanBmiPrestroke=np.mean(np.array(person._bmi[:waveAtLastStroke+1])),
                meanSBP=np.array(person._sbp[waveAtLastStroke+1:]).mean(),
                meanSBPPrestroke=np.array(person._sbp[:waveAtLastStroke+1]).mean(),
                meanLdlPrestroke=np.array(person._ldl[:waveAtLastStroke+1]).mean(),
                meanLdl=np.array(person._ldl[waveAtLastStroke+1:]).mean(),
                gfr=person._gfr,
                meanWaistPrestroke=np.mean(np.array(person._waist[:waveAtLastStroke+1])),
                meanFastingGlucose=convert_a1c_to_fasting_glucose(np.array(person._a1c[waveAtLastStroke+1:]).mean()),
                meanFastingGlucosePrestroke=convert_a1c_to_fasting_glucose(np.array(person._a1c[:waveAtLastStroke+1]).mean()),
                anyAntiHypertensive=person._any_antiHypertensive,
                #Q: how to deal with otherLipidlowering meds? We used to use this attribute but now that I have not
                #   included a treatment model for this (and I think I do not even bring it in from NHANES)
                #   is it ok to use just statin for the gcp stroke model, like I do below?
                anyLipidLowering= person._statin[-1],
                afib=person._afib[-1],
                mi=person._mi,
                meanGCPPrestroke=np.mean(np.array(personGCP[:waveAtLastStroke+1])))
        random_effect_slope_term = random_effect_slope * yearsSinceStroke

        return linPred + random_effect + random_effect_slope_term + residual


class CognitionPrevalenceModel:
    """Seeds the baseline GCP cognition outcome at Person construction.

    Cognition is a continuous score (GCP), not a binary event, so this model always emits a
    CognitionOutcome (priorToSim=True) rather than gating on a probability. The post-stroke
    GCP model is not used here because priorToSim outcomes carry age=None and the post-stroke
    model needs an age-at-last-stroke."""

    _outcomeType = OutcomeType.COGNITION

    def __init__(self):
        self._model = GCPModel()

    def get_prevalent_outcome(self, person):
        gcp = self._model.get_risk_for_person(person, person._rng)
        return CognitionOutcome(fatal=False, priorToSim=True, gcp=gcp)
