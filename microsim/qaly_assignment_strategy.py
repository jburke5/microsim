import numpy as np

from microsim.qaly_outcome import QALYOutcome
from microsim.outcome import OutcomeType

class QALYAssignmentStrategy:
    def __init__(self):
        # the first element in the list is the QALY for the first year after the event, the next qaly for the next year...
        # if the patient is more than the length of hte list out from the last index, the last index is repeated
        self._qalysForOutcome = {}
        self._qalysForOutcome[OutcomeType.STROKE] = [0.67, 0.90]
        self._qalysForOutcome[OutcomeType.MI] = [0.88, 0.90]
        self._qalysForOutcome[OutcomeType.DEMENTIA] = list(np.arange(0.80, 0, -0.01))

    def generate_next_outcome(self, person):
        qaly = self.get_next_qaly(person)
        fatal = False
        selfReported = False
        return QALYOutcome(fatal, selfReported, qaly)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person)

    def get_next_qaly(self, person, rng=None, age=-1):
        if age==-1:
            age=person._age[-1]
        
        # qaly assignment happens prior to advancing an age, but after condtiions are set...
        wave = person.get_wave_for_age(age) 
        conditions = self.get_conditions_for_person(person, wave)
        return self.get_qalys_for_age_and_conditions(age, conditions, person.is_dead)

    def get_qalys_for_age_and_conditions(self, age, conditions, dead, x=None):
        base = self.get_base_qaly_for_age(age)
        return (
            0
            if dead
            else base * np.prod(self.get_multipliers_for_conditions(conditions, age, x))
        )

    def get_conditions_for_person(self, person, wave):
        return {
            OutcomeType.DEMENTIA: (
                person.has_outcome_during_or_prior_to_wave(wave, OutcomeType.DEMENTIA),
                person.get_age_at_first_outcome(OutcomeType.DEMENTIA),
            ),
            OutcomeType.STROKE: (
                person.has_outcome_during_or_prior_to_wave(wave, OutcomeType.STROKE),
                person.get_age_at_first_outcome(OutcomeType.STROKE),
            ),
            OutcomeType.MI: (person.has_outcome_during_or_prior_to_wave(wave, OutcomeType.MI), person.get_age_at_first_outcome(OutcomeType.MI)),
        }

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

    def get_multipliers_for_conditions(self, conditions, age, x=None):
        multipliers = []
        # each element is a tuple where the first element is age and the second element is the outcome
        for outcomeType, outcomeTuple in conditions.items():
            hasOutcome = outcomeTuple[0]
            ageAtEvent = outcomeTuple[1]
            qalyListForOutcome = (
                self._qalysForOutcome[outcomeType]
                if outcomeType in self._qalysForOutcome
                else None
            )
            if qalyListForOutcome is not None and hasOutcome:
                if np.isnan(age) or np.isnan(ageAtEvent):
                    print(
                        f"ABOUT TO BREAK...current age: {age}, age at event: {ageAtEvent}, outcomeTYpe: {outcomeType}, outcomeTuple: {outcomeTuple}"
                    )
                yearsFromEvent = int(age - ageAtEvent)
                # print(f"current age: {currentAge}, age at event: {ageAtEvent}, outcome type: {outcomeType}, conditions: {conditions}, x: {x}")
                if yearsFromEvent >= len(qalyListForOutcome) and len(qalyListForOutcome) < 1:
                    raise RuntimeError(f"error 1: qalyListForOutcome: {qalyListForOutcome}")
                elif yearsFromEvent < len(qalyListForOutcome) and yearsFromEvent < 0:
                    raise RuntimeError(
                        f"error 2 qalyListForOutcome: {qalyListForOutcome} years from event: {yearsFromEvent} currentAge : {age}, age at event: {ageAtEvent}"
                    )
                qalys = (
                    qalyListForOutcome[-1]
                    if yearsFromEvent >= len(qalyListForOutcome)
                    else qalyListForOutcome[yearsFromEvent]
                )
                multipliers.append(qalys)

        return multipliers

    def get_qalys_vectorized(self, x):
        conditions = {
            OutcomeType.DEMENTIA: (x.dementia or x.dementiaNext, x.ageAtFirstDementia),
            OutcomeType.STROKE: (x.strokeNext or x.stroke, x.ageAtFirstStroke),
            OutcomeType.MI: (x.miNext or x.mi, x.ageAtFirstMI),
        }
        return self.get_qalys_for_age_and_conditions(x.age, conditions, x.dead, x)
