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
        conditions = self.get_conditions_for_person(person)
        return self.get_qalys_for_age_and_conditions(person._age[-1], conditions, person.is_dead())

    def get_qalys_for_age_and_conditions(self, currentAge, conditions, dead, x=None):
        base = self.get_base_qaly_for_age(currentAge)
        return 0 if dead else base * np.prod(self.get_multipliers_for_conditions(conditions, currentAge, x))

    def get_conditions_for_person(self, person):
        return {OutcomeType.DEMENTIA: (person._dementia, person.get_age_at_first_outcome(OutcomeType.DEMENTIA)),
                OutcomeType.STROKE: (person._stroke, person.get_age_at_first_outcome(OutcomeType.STROKE)),
                OutcomeType.MI: (person._mi, person.get_age_at_first_outcome(OutcomeType.MI))}

    # simple age-based approximation that after age 70, you lose about 0.01 QALYs per year
    # from: Netuveli, G. (2006). Quality of life at older ages: evidence from the English longitudinal study of aging (wave 1).
    # Journal of Epidemiology and Community Health, 60(4), 357–363. http://doi.org/10.1136/jech.2005.040071
    #  i think we can get away with something htat sets QALYS at 1 < 70 and then applies a baseline reduction of
    # something like 10% per decade

    def get_base_qaly_for_age(self, age):
        yearsOver70 = age - 70 if age > 70 else 0
        return 1 - yearsOver70 * 0.01

    # for  dementia... Jönsson, L., Andreasen, N., Kilander, L., Soininen, H., Waldemar, G., Nygaard, H., et al. (2006).
    # Patient- and proxy-reported utility in Alzheimer disease using the EuroQoL. Alzheimer Disease and Associated Disorders,
    # 20(1), 49–55. http://doi.org/10.1097/01.wad.0000201851.52707.c9
    # it has utilizites at bsaeline and with dementia follow-up...seems like a simple thing  to use...

    # for stroke/MI...Sussman, J., Vijan, S., & Hayward, R. (2013). Using benefit-based tailored treatment to improve the use of
    # antihypertensive medications. Circulation, 128(21), 2309–2317. http://doi.org/10.1161/CIRCULATIONAHA.113.002290
    # same idea, has utilities at bsaeline and in sugsequent years...so, it shoudl be relatifely easy to use.

    def get_multipliers_for_conditions(self, conditions, currentAge, x=None):
        multipliers = []
        # each element is a tuple where the first element is age and the second element is the outcome
        for outcomeType, outcomeTuple in conditions.items():
            hasOutcome = outcomeTuple[0]
            ageAtEvent = outcomeTuple[1]
            qalyListForOutcome = self._qalysForOutcome[outcomeType] if outcomeType in self._qalysForOutcome else None
            if qalyListForOutcome is not None and hasOutcome:
                if ageAtEvent is None:
                    print(f"current age: {currentAge}, age at event: {ageAtEvent}, outcome type: {outcomeType}, conditions: {conditions}, x: {x}")
                yearsFromEvent = int(currentAge - ageAtEvent)
                qalys = qalyListForOutcome[-1] if yearsFromEvent >= len(
                    qalyListForOutcome) else qalyListForOutcome[yearsFromEvent]
                multipliers.append(qalys)

        return multipliers

    def get_qalys_vectorized(self, x):
        conditions = {OutcomeType.DEMENTIA: (x.dementia, x.ageAtFirstDementia),
                      OutcomeType.STROKE: (x.stroke, x.ageAtFirstStroke),
                      OutcomeType.MI: (x.mi, x.ageAtFirstMI)}
        return self.get_qalys_for_age_and_conditions(x.age, conditions, x.dead, x)
