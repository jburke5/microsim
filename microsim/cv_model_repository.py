from microsim.cv_model import (CVModelMale, CVModelMaleFor1bpMedsAdded, CVModelMaleFor2bpMedsAdded, 
                               CVModelMaleFor3bpMedsAdded, CVModelMaleFor4bpMedsAdded,
                               CVModelFemale, CVModelFemaleFor1bpMedsAdded, CVModelFemaleFor2bpMedsAdded, 
                               CVModelFemaleFor3bpMedsAdded, CVModelFemaleFor4bpMedsAdded)
from microsim.gender import NHANESGender
from microsim.treatment import TreatmentStrategiesType

class CVModelRepository:
    def __init__(self):
        self._models = {"male": 
                           {"0bpMedsAdded": CVModelMale(),
                            "1bpMedsAdded": CVModelMaleFor1bpMedsAdded(),
                            "2bpMedsAdded": CVModelMaleFor2bpMedsAdded(),
                            "3bpMedsAdded": CVModelMaleFor3bpMedsAdded(),
                            "4bpMedsAdded": CVModelMaleFor4bpMedsAdded()},
                        "female": 
                            {"0bpMedsAdded": CVModelFemale(),
                             "1bpMedsAdded": CVModelFemaleFor1bpMedsAdded(),
                             "2bpMedsAdded": CVModelFemaleFor2bpMedsAdded(),
                             "3bpMedsAdded": CVModelFemaleFor3bpMedsAdded(),
                             "4bpMedsAdded": CVModelFemaleFor4bpMedsAdded()}}

    def select_outcome_model_for_person(self, person):
        gender = "male" if person._gender==NHANESGender.MALE else "female"
        tst = TreatmentStrategiesType.BP.value
        if "bpMedsAdded" in person._treatmentStrategies[tst]:
            ts = f"{person._treatmentStrategies[tst]['bpMedsAdded']}bpMedsAdded"
        else:
            ts = "0bpMedsAdded"
        return self._models[gender][ts] 

