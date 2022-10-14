from microsim.trials.trial import Trial

class Trialset:

    def __init__(self, trialDescription, pop, trialCount):
        self.trials = []
        self.trialDescription = trialDescription
        for i in range(0, trialCount):
            self.trials.append(Trial(trialDescription, pop))

    def run(self):
        for trial in self.trials:
            trial.run()
            trial.analyze()