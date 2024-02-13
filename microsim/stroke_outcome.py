from microsim.outcome import Outcome, OutcomeType
from enum import Enum
import numpy as np

class StrokeOutcome(Outcome):

    #phenotypeItems = ["nihss","strokeSubtype","strokeType","localization","disability"]
    phenotypeItems = ["nihss","strokeSubtype","strokeType", "priorToSim"]

    #def __init__(self, fatal, nihss, strokeType, strokeSubtype, location, disability):
    def __init__(self, fatal, nihss, strokeType, strokeSubtype, priorToSim=False):
        self.fatal = fatal
        self.priorToSim = priorToSim
        super().__init__(OutcomeType.STROKE, self.fatal, self.priorToSim)
        self.nihss = nihss
        self.strokeType = strokeType
        self.strokeSubtype = strokeSubtype
        #self.location = location 
        #self.disability = disability

    @staticmethod
    def add_outcome_vars(outcomeDict, numberVars):
        outcomeDict['nihssNext'] = [np.nan] * numberVars
        outcomeDict['strokeSubtypeNext'] = [None] * numberVars
        outcomeDict['strokeTypeNext'] = [None] * numberVars
        #outcomeDict['localizationNext'] = [None] * numberVars
        #outcomeDict['disabilityNext'] = [None] * numberVars
        outcomeDict["strokeFatal"] = [False] * numberVars
        return outcomeDict
    
    def __repr__(self):
        return f"""Stroke Outcome: {self.type}, fatal: {self.fatal}, nihss: {self.nihss}, 
                stroke subtype: {self.strokeSubtype}, stroke type: {self.strokeType},"""
                #stroke location: {self.location}, disability: {self.disability}"""

    def __eq__(self, other):
        if not isinstance(other, StrokeOutcome):
            return False
        return ((self.type == other.type) and (self.fatal == other.fatal) and 
                (self.nihss == other.nihss) and (self.strokeType == other.strokeType) ) #and 
                #(self.location == other.location) and (self.disability ==other.disability ))
    
class StrokeType(Enum):
    ISCHEMIC = "ischemic"
    ICH = "ich"

class StrokeSubtype(Enum):
    LARGE_VESSEL = "largeVessel"
    SMALL_VESSEL = "smallVessel"
    CARDIOEMBOLIC = "cardioembolic"
    OTHER = "other"

class Localization(Enum):
    RIGHT_HEMISPHERE = "rightHemisphere"
    LEFT_HEMISPHERE = "leftHemisphere"
    CEREBELLUM_BRAINSTEM = "cerebellumBrainstem"



