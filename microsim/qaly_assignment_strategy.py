import numpy as np

from microsim.outcome import OutcomeType


class QALYAssignmentStrategy:
    def __init__(self):
        # the first element in the list is the QALY for the first year after the event, the next qaly for the next year...
        # if the patient is more than the length of hte list out from the last index, the last index is repeated
        self._qalysForOutcome = {}
        self._qalysForOutcome[OutcomeType.STROKE] = [0.67, 0.90]
        self._qalysForOutcome[OutcomeType.MI] = [0.88, 0.90]
        # hacktag: this is crude for now...we shoudl be able to map this to GCP
        self._qalysForOutcome[OutcomeType.DEMENTIA] = list(np.arange(0.80, 0, -0.01))

    def get_next_qaly(self, person):
        base = self.get_base_qaly_for_age(person)
        return 0 if person.is_dead() else base * np.prod(self.get_multipliers_for_conditions(person))

    # simple age-based approximation that after age 70, you lose about 0.01 QALYs per year
    # from: Netuveli, G. (2006). Quality of life at older ages: evidence from the English longitudinal study of aging (wave 1).
    # Journal of Epidemiology and Community Health, 60(4), 357–363. http://doi.org/10.1136/jech.2005.040071
    #  i think we can get away with something htat sets QALYS at 1 < 70 and then applies a baseline reduction of
    # something like 10% per decade

    def get_base_qaly_for_age(self, person):
        yearsOver70 = person._age[-1] - 70 if person._age[-1] > 70 else 0
        return 1 - yearsOver70 * 0.01

    # for  dementia... Jönsson, L., Andreasen, N., Kilander, L., Soininen, H., Waldemar, G., Nygaard, H., et al. (2006).
    # Patient- and proxy-reported utility in Alzheimer disease using the EuroQoL. Alzheimer Disease and Associated Disorders,
    # 20(1), 49–55. http://doi.org/10.1097/01.wad.0000201851.52707.c9
    # it has utilizites at bsaeline and with dementia follow-up...seems like a simple thing  to use...

    # for stroke/MI...Sussman, J., Vijan, S., & Hayward, R. (2013). Using benefit-based tailored treatment to improve the use of
    # antihypertensive medications. Circulation, 128(21), 2309–2317. http://doi.org/10.1161/CIRCULATIONAHA.113.002290
    # same idea, has utilities at bsaeline and in sugsequent years...so, it shoudl be relatifely easy to use.


    def get_multipliers_for_conditions(self, person):
        multipliers = []
        # each element is a tuple where the first element is age and the second element is the outcome
        for outcomeList in person._outcomes.values():
            if len(outcomeList) > 0:
                outcomeTuple = outcomeList[0]
                qalyListForOutcome = self._qalysForOutcome[outcomeTuple[1].type] if outcomeTuple[1].type in self._qalysForOutcome else None
                if qalyListForOutcome is not None:
                    ageAtEvent = person._age[-1] - 1 if outcomeTuple[0] == -1 else outcomeTuple[0]
                    yearsFromEvent = int(person._age[-1] - ageAtEvent)
                    qalys = qalyListForOutcome[-1] if yearsFromEvent >= len(qalyListForOutcome) else qalyListForOutcome[yearsFromEvent]
                    multipliers.append(qalys)

        return multipliers
