#from microsim.cv_model import (CVModelMale, CVModelMaleFor1bpMedsAdded, CVModelMaleFor2bpMedsAdded, 
#                               CVModelMaleFor3bpMedsAdded, CVModelMaleFor4bpMedsAdded,
#                               CVModelFemale, CVModelFemaleFor1bpMedsAdded, CVModelFemaleFor2bpMedsAdded, 
#                               CVModelFemaleFor3bpMedsAdded, CVModelFemaleFor4bpMedsAdded)
from microsim.cv_model import *
from microsim.gender import NHANESGender
from microsim.treatment import TreatmentStrategiesType

class CVModelRepository:
    def __init__(self):
        self._models = {"male": CVModelMale(),
                        "female": CVModelFemale()}
        #self._models = {"male": 
        #                   {"0bpMedsAdded": CVModelMale(),
        #                    "1bpMedsAdded": CVModelMaleFor1bpMedsAdded(),
        #                    "2bpMedsAdded": CVModelMaleFor2bpMedsAdded(),
        #                    "3bpMedsAdded": CVModelMaleFor3bpMedsAdded(),
        #                    "4bpMedsAdded": CVModelMaleFor4bpMedsAdded(),
        #                    "5bpMedsAdded": CVModelMaleFor5bpMedsAdded(),
        #                    "6bpMedsAdded": CVModelMaleFor6bpMedsAdded(),
        #                    "7bpMedsAdded": CVModelMaleFor7bpMedsAdded(),
        #                    "8bpMedsAdded": CVModelMaleFor8bpMedsAdded(),
        #                    "9bpMedsAdded": CVModelMaleFor9bpMedsAdded(),
        #                    "10bpMedsAdded": CVModelMaleFor10bpMedsAdded()},
        #                "female": 
        #                    {"0bpMedsAdded": CVModelFemale(),
        #                     "1bpMedsAdded": CVModelFemaleFor1bpMedsAdded(),
        #                     "2bpMedsAdded": CVModelFemaleFor2bpMedsAdded(),
        #                     "3bpMedsAdded": CVModelFemaleFor3bpMedsAdded(),
        #                     "4bpMedsAdded": CVModelFemaleFor4bpMedsAdded(),
        #                     "5bpMedsAdded": CVModelFemaleFor5bpMedsAdded(),
        #                     "6bpMedsAdded": CVModelFemaleFor6bpMedsAdded(),
        #                     "7bpMedsAdded": CVModelFemaleFor7bpMedsAdded(),
        #                     "8bpMedsAdded": CVModelFemaleFor8bpMedsAdded(),
        #                     "9bpMedsAdded": CVModelFemaleFor9bpMedsAdded(),
        #                     "10bpMedsAdded": CVModelFemaleFor10bpMedsAdded()}}

    def select_outcome_model_for_person(self, person):
        gender = "male" if person._gender==NHANESGender.MALE else "female"
        return self._models[gender]

#    def select_outcome_model_for_person(self, person):
#        gender = "male" if person._gender==NHANESGender.MALE else "female"
#        tst = TreatmentStrategiesType.BP.value
#        if "bpMedsAdded" in person._treatmentStrategies[tst]:
#            bpMedsAdded = person._treatmentStrategies[tst]['bpMedsAdded']
#            if bpMedsAdded < 11:
#                ts = f"{bpMedsAdded}bpMedsAdded"
#            else:
#                ts = "10bpMedsAdded"
#        else:
#            ts = "0bpMedsAdded"
#        return self._models[gender][ts] 

