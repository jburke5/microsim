from microsim.outcome import OutcomeType
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cv_model_repository import CVModelRepository
from microsim.risk_factor import DynamicRiskFactorsType
from microsim.treatment import TreatmentStrategyStatus, TreatmentStrategiesType

from abc import ABC

# tryign to turn off ABC as it causes pickling problems...this class shoudl extend ABC
# BP treamtent classees modify, only, the number of BP medications added
# the total number of BP medications are the number added at baseline (antiHypertensiveCount)
# combined with the number added, additionalliy, via treatment algorithms
class BaseTreatmentStrategy:
    MAX_BP_MEDS = 4

    def get_changes_vectorized(self, x):
        raise RuntimeError("Vectorized Treatment not implemented for this strategy")

class AddNBPMedsTreatmentStrategy(BaseTreatmentStrategy):
    def __init__(self, n):
        self.bpMedsAdded = n
        self.sbpLowering = 5.5*self.bpMedsAdded
        self.dbpLowering = 3.1*self.bpMedsAdded
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.BEGIN:
            person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]=self.bpMedsAdded
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] - self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] - self.dbpLowering}
        elif person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.END:
            del person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] + self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] + self.dbpLowering}
        else:
            return dict()

class AddASingleBPMedTreatmentStrategy(BaseTreatmentStrategy):
    sbpLowering = 5.5
    dbpLowering = 3.1
    bpMedsAdded = 1

    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.BEGIN:
            person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]=self.bpMedsAdded
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] - self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] - self.dbpLowering}
        elif person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.END:
            del person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] + self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] + self.dbpLowering}
        else:
            return dict()

    #def get_changes_for_person(self, person):
    #    return (
    #        {"_antiHypertensiveCount": 1},
    #        {"_bpMedsAdded": 1},
    #        {
    #            "_sbp": -1 * AddASingleBPMedTreatmentStrategy.sbpLowering,
    #            "_dbp": -1 * AddASingleBPMedTreatmentStrategy.dbpLowering,
    #        },
    #    )

    #def get_treatment_recalibration_for_population(self):
    #    return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    #def get_treatment_recalibration_for_person(self):
    #    return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    #def repeat_treatment_strategy(self):
    #    return False

    #def get_changes_vectorized(self, x):
    #    x.bpMedsAddedNext = 1
    #    x.totalBPMedsAddedNext = 1
    #    x.sbpNext = x.sbpNext - AddASingleBPMedTreatmentStrategy.sbpLowering
    #    x.dbpNext = x.dbpNext - AddASingleBPMedTreatmentStrategy.dbpLowering
    #    return x

class AddBPTreatmentMedsToGoal120(BaseTreatmentStrategy):
    sbpLowering = 5.5
    dbpLowering = 3.1

    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_goal(self, person):
        return {"sbp": 120, "dbp": 65}

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        if person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.END:
            bpMedsAdded = person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]
            del person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"]
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] + bpMedsAdded*self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] + bpMedsAdded*self.dbpLowering}
        elif person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.BEGIN:
            bpMedsAdded = self.get_meds_needed_for_goal(person, self.get_goal(person))
            person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"] = bpMedsAdded
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] - bpMedsAdded*self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] - bpMedsAdded*self.dbpLowering} 
        elif person._treatmentStrategies[TreatmentStrategiesType.BP.value]["status"]==TreatmentStrategyStatus.MAINTAIN:
            bpMedsAdded = self.get_meds_needed_for_goal(person, self.get_goal(person))
            person._treatmentStrategies[TreatmentStrategiesType.BP.value]["bpMedsAdded"] += bpMedsAdded
            return {DynamicRiskFactorsType.SBP.value: getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] - bpMedsAdded*self.sbpLowering,
                    DynamicRiskFactorsType.DBP.value: getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] - bpMedsAdded*self.dbpLowering}
        else:
            raise RuntimeError("Unrecognized TreatmentStrategiesType status for person in AddBPTreatmentMedsToGoal120.")

    def get_meds_needed_for_goal(self, person, goal):
        sbpMedCount = int((getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] - goal["sbp"]) 
                          / AddBPTreatmentMedsToGoal120.sbpLowering)
        dbpMedCount = int((getattr(person, "_"+DynamicRiskFactorsType.DBP.value)[-1] - goal["dbp"]) 
                          / AddBPTreatmentMedsToGoal120.dbpLowering)
        medCount = dbpMedCount if dbpMedCount < sbpMedCount else sbpMedCount
        return 0 if medCount < 0 else int(medCount)

    #def get_changes_for_person(self, person):        
    #    medsForGoal = self.get_meds_needed_for_goal(
    #        person._sbp[-1],
    #        person._dbp[-1],
    #        person._antiHypertensiveCount[-1] + person._bpMedsAdded[-1],
    #        self.getGoal(person),
    #    )
    #    return (
    #        {},
    #        {"_bpMedsAdded": medsForGoal},
    #        {
    #            "_sbp": -1 * medsForGoal * AddBPTreatmentMedsToGoal120.sbpLowering,
    #            "_dbp": -1 * medsForGoal * AddBPTreatmentMedsToGoal120.dbpLowering,
    #        },
    #    )

    #def get_treatment_recalibration_for_population(self):
    #    return {OutcomeType.STROKE: 0.79, OutcomeType.MI: 0.87}

    #def get_treatment_recalibration_for_person(self, person):
    #    medsForGoal = self.get_meds_needed_for_goal(person.sbp[-1], person.dbp[-1])
    #    return {OutcomeType.STROKE: 0.79**medsForGoal, OutcomeType.MI: 0.87**medsForGoal}

    #def repeat_treatment_strategy(self):
    #    return True

    #def get_goal_vectorized(self, x):
    #    return self.getGoal()

    #def get_changes_vectorized(self, x):
    #    medsNeeded = self.get_meds_needed_for_goal(
    #        x.sbpNext, x.dbpNext, x.antiHypertensiveCountNext + x.totalBPMedsAdded, self.get_goal_vectorized(x)
    #    )
    #    # how many medications to add in addition to baseline
    #    x.bpMedsAddedNext =  medsNeeded 
    #    x.totalBPMedsAddedNext = x.totalBPMedsAdded + medsNeeded
    #    x.sbpNext = x.sbpNext - medsNeeded * AddBPTreatmentMedsToGoal120.sbpLowering
    #    x.dbpNext = x.dbpNext - medsNeeded * AddBPTreatmentMedsToGoal120.dbpLowering
    #    return x

class NoBPTreatment(BaseTreatmentStrategy):
    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        return dict()

    #def get_changes_for_person(self, person):
    #    current = person._antiHypertensiveCount[-1]
    #    return (
    #        {"_antiHypertensiveCount": 0},
    #        {"_bpMedsAdded": 0},
    #        {"_sbp": 0, "_dbp": 0},
    #    )

    #def get_treatment_recalibration_for_population(self):
    #    return None

    #def get_treatment_recalibration_for_person(self):
    #    return None

    #def repeat_treatment_strategy(self):
    #    return True

    # BP meds can change via the usual care risk model...but, we won't add any meds...
    #def get_changes_vectorized(self, x):
    #    x.bpMedsAddedNext = 0
    #    x.totalBPMedsAddedNext = 0
    #    return x

# James, P. A. et al. 2014 Evidence-Based Guideline for the Management of High Blood Pressure in Adults: Report From the Panel Members Appointed to the Eighth Joint National Committee (JNC 8). Jama 311, 507–520 (2014).
class jnc8Treatment(AddBPTreatmentMedsToGoal120):
    def low_target(self, person):
        return person._diabetes or person._ckd or getattr(person, "_"+DynamicRiskFactorsType.AGE.value)[-1] < 60
    
    def get_goal(self, person):
        return {'sbp' : 140, 'dbp' : 90} if self.low_target(person) else {'sbp' : 150, 'dbp' : 90}
        
    #def lowTargetVectorized(self, x):
    #    #print(list(x.index))
    #    return x.current_diabetes or x.gfr < 60 or x.age < 60
        
    #def get_goal_vectorized(self, x):
    #    return {'sbp' : 140, 'dbp' : 90} if self.lowTargetVectorized(x) else {'sbp' : 150, 'dbp' : 90}

    def get_meds_needed_for_goal(self, person, goal):
        meds = super().get_meds_needed_for_goal(person, goal)
        currentMeds = person._antiHypertensiveCountPlusBPMedsAdded()
        cappedMeds = BaseTreatmentStrategy.MAX_BP_MEDS if meds > BaseTreatmentStrategy.MAX_BP_MEDS else meds
        medsToReturn = BaseTreatmentStrategy.MAX_BP_MEDS - currentMeds  if cappedMeds + currentMeds > BaseTreatmentStrategy.MAX_BP_MEDS else cappedMeds
        #print(f"meds: {meds} sbp: {sbp} dbp: {dbp} goal: {goal} currentMeds: {currentMeds}, cappedMeds: {cappedMeds}, medsToReturn: {medsToReturn}")
        return int(medsToReturn) if medsToReturn > 0 else 0

class jnc8ForHighRisk(jnc8Treatment):
    def __init__(self, targetRisk):
        self.targetRisk = targetRisk
        self.cvModelRepository = CVModelRepository()
    
    def low_target(self, person):
        risk = self.cvModelRepository.select_outcome_model_for_person(person).get_risk_for_person(person, years=10)
        return risk >  self.targetRisk
    
    #def lowTargetVectorized(self, x):
    #    risk = OutcomeModelRepository().get_risk_for_person(x, OutcomeModelType.CARDIOVASCULAR, 10, True)
    #    return risk >  self.targetRisk

class jnc8ForHighRiskLowBpTarget(jnc8ForHighRisk):
    def __init__(self, targetRisk, targetBP):
        super().__init__(targetRisk)
        self.targetBP = targetBP
        
    def get_goal(self, person):
        return self.targetBP 
    
    #def get_goal_vectorized(self, x):
    #    return self.targetBP

# simplified class to represent SPRINT.
class SprintTreatment(jnc8ForHighRiskLowBpTarget):
    def __init__(self):
        super().__init__(0.075, {'sbp' : 126, 'dbp': 85})
        self.status = TreatmentStrategyStatus.BEGIN
