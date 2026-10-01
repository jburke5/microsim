from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType


class MCIModel:
    """Mild cognitive impairment model."""

    def __init__(self, riskScaling=1.0):
        self._riskScaling = riskScaling

    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.MCI, False)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person) if self.get_mci_for_person(person) else None

    def get_mci_for_person(self, person):
        mci = person.has_mci()

        if mci:
            mci = (
                mci if person._rng.uniform(size=1) > (self._riskScaling) ** (1.0 / 4.0) else False
            )

        # if mci: #need to test this
        #    # 0.90727 = (2/3)^(1/4), because I want the risk to be 2/3 over a 4 year simulation
        #    mci = mci if person._rng.uniform(size=1)>0.75 else False

        if mci:
            tst = TreatmentStrategiesType.WMD15.value
            if "wmd15MedsAdded" in person._treatmentStrategies[tst]:
                wmd15MedsAdded = person._treatmentStrategies[tst]["wmd15MedsAdded"]
                mci = (
                    True
                    if ((wmd15MedsAdded > 0) and (person._rng.uniform(size=1) < 0.8))
                    else False
                )

            tst = TreatmentStrategiesType.WMD20.value
            if "wmd20MedsAdded" in person._treatmentStrategies[tst]:
                wmd20MedsAdded = person._treatmentStrategies[tst]["wmd20MedsAdded"]
                mci = (
                    True
                    if ((wmd20MedsAdded > 0) and (person._rng.uniform(size=1) < 0.75))
                    else False
                )

            tst = TreatmentStrategiesType.WMD25.value
            if "wmd25MedsAdded" in person._treatmentStrategies[tst]:
                wmd25MedsAdded = person._treatmentStrategies[tst]["wmd25MedsAdded"]
                mci = (
                    True
                    if ((wmd25MedsAdded > 0) and (person._rng.uniform(size=1) < 0.69))
                    else False
                )
        return mci
