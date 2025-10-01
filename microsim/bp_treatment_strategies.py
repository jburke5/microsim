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

# bpMedsAdded is the total number of bp medications that have been added so far over all completed waves

# some of these treatment strategies are purely theoretical due to their assumptions 
# for example the assumption that the 10th bpMed will lower SBP by the same amount as the 1st bpMed
# even though all of these classes may be used in a simulation, some are more practical, some are more theoretical

class BaseTreatmentStrategy:
    MAX_BP_MEDS = 4
    SBP_MULTIPLIER = 5.5
    DBP_MULTIPLIER = 3.1

class AddNBPMedsTreatmentStrategy(BaseTreatmentStrategy):
    def __init__(self, n):
        self.bpMedsAdded = n
        self.sbpLowering = self.SBP_MULTIPLIER * self.bpMedsAdded
        self.dbpLowering = self.DBP_MULTIPLIER * self.bpMedsAdded
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

class AddASingleBPMedTreatmentStrategy(AddNBPMedsTreatmentStrategy):
    sbpLowering = BaseTreatmentStrategy.SBP_MULTIPLIER
    dbpLowering = BaseTreatmentStrategy.DBP_MULTIPLIER

    def __init__(self):
        super().__init__(1)

class AddBPTreatmentMedsToGoal120(BaseTreatmentStrategy):
    sbpLowering = BaseTreatmentStrategy.SBP_MULTIPLIER
    dbpLowering = BaseTreatmentStrategy.DBP_MULTIPLIER

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

class NoBPTreatment(BaseTreatmentStrategy):
    def __init__(self):
        self.status = TreatmentStrategyStatus.BEGIN

    def get_updated_treatments(self, person):
        return dict()

    def get_updated_risk_factors(self, person):
        return dict()

# James, P. A. et al. 2014 Evidence-Based Guideline for the Management of High Blood Pressure in Adults: Report From the Panel Members Appointed to the Eighth Joint National Committee (JNC 8). Jama 311, 507–520 (2014).
class jnc8Treatment(AddBPTreatmentMedsToGoal120):
    def low_target(self, person):
        return person._diabetes or person._ckd or getattr(person, "_"+DynamicRiskFactorsType.AGE.value)[-1] < 60
    
    def get_goal(self, person):
        return {'sbp' : 140, 'dbp' : 90} if self.low_target(person) else {'sbp' : 150, 'dbp' : 90}
        
    def get_meds_needed_for_goal(self, person, goal):
        meds = super().get_meds_needed_for_goal(person, goal)
        currentMeds = person._antiHypertensiveCountPlusBPMedsAdded()
        cappedMeds = BaseTreatmentStrategy.MAX_BP_MEDS if meds > BaseTreatmentStrategy.MAX_BP_MEDS else meds
        medsToReturn = BaseTreatmentStrategy.MAX_BP_MEDS - currentMeds  if cappedMeds + currentMeds > BaseTreatmentStrategy.MAX_BP_MEDS else cappedMeds
        #print(f"meds: {meds} sbp: {sbp} dbp: {dbp} goal: {goal} currentMeds: {currentMeds}, cappedMeds: {cappedMeds}, medsToReturn: {medsToReturn}")
        return int(medsToReturn) if medsToReturn > 0 else 0

class jnc8ForHighRisk(jnc8Treatment):
    def __init__(self, targetRisk, wmhSpecific=True):
        self.targetRisk = targetRisk
        self.cvModelRepository = CVModelRepository(wmhSpecific=wmhSpecific)
    
    def low_target(self, person):
        risk = self.cvModelRepository.select_outcome_model_for_person(person).get_risk_for_person(person, years=10)
        return risk >  self.targetRisk
    
class jnc8ForHighRiskLowBpTarget(jnc8ForHighRisk):
    def __init__(self, targetRisk, targetBP, wmhSpecific=True):
        super().__init__(targetRisk, wmhSpecific)
        self.targetBP = targetBP
        self.status = TreatmentStrategyStatus.BEGIN
    
    def get_goal(self, person):
        return self.targetBP 
    
# simplified class to represent SPRINT.
class SprintTreatment(jnc8ForHighRiskLowBpTarget):
    def __init__(self, wmhSpecific=True):
        super().__init__(0.075, {'sbp' : 126, 'dbp': 85}, wmhSpecific)
        self.status = TreatmentStrategyStatus.BEGIN

class SprintForLowerDbpGoalTreatment(jnc8ForHighRiskLowBpTarget):
    def __init__(self):
        super().__init__(0.075, {'sbp' : 126, 'dbp': 65})
        self.status = TreatmentStrategyStatus.BEGIN

class SprintForSbpOnlyTreatment(jnc8ForHighRiskLowBpTarget):
    '''This treatment strategy practically implements an SBP only goal for blood pressure treatment.
    There are formally two goals for both SBP and DBP but the DBP goal is set so high that it 
    will be unlikely ever used.'''
    def __init__(self, cvRiskCutoff=0.075, wmhSpecific=True):
        super().__init__(cvRiskCutoff, {'sbp' : 126, 'dbp': 200}, wmhSpecific)
        self.status = TreatmentStrategyStatus.BEGIN

    def get_meds_needed_for_goal(self, person, goal):
        '''The Sprint-based classes utilize the minimum of the SBP and DBP meds needed to reach the goal...
        essentially I cannot just initialize the super class with an extremely high DBP goal because then the DBP meds needed
        would always be 0 and that is the minimum no matter how many SBP meds are needed to reach the goal.
        So, I actually need to implement this function here based on SBP only.'''
        sbpMedCount = int((getattr(person, "_"+DynamicRiskFactorsType.SBP.value)[-1] - goal["sbp"])
                          / AddBPTreatmentMedsToGoal120.sbpLowering)
        currentMeds = person._antiHypertensiveCountPlusBPMedsAdded()
        cappedMeds = BaseTreatmentStrategy.MAX_BP_MEDS if sbpMedCount > BaseTreatmentStrategy.MAX_BP_MEDS else sbpMedCount
        medsToReturn = BaseTreatmentStrategy.MAX_BP_MEDS - currentMeds  if cappedMeds + currentMeds > BaseTreatmentStrategy.MAX_BP_MEDS else cappedMeds
        return int(medsToReturn) if medsToReturn > 0 else 0

class SprintForSbpRiskThreshold(SprintForSbpOnlyTreatment):
    '''This strategy will be use an SBP goal only and it will implement the goal only if the CV risk is above 
    a threshold.'''
    def __init__(self, cvRiskCutoff=0.075, wmhSpecific=True):
        super().__init__(cvRiskCutoff, wmhSpecific)
        self.status = TreatmentStrategyStatus.BEGIN

    def get_meds_needed_for_goal(self, person, goal):
        '''The low_target function determines if the CV risk is above the threshold.'''
        return super().get_meds_needed_for_goal(person, goal) if self.low_target(person) else 0

