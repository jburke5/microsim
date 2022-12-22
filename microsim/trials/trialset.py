from microsim.trials.trial import Trial
from microsim.trials.trial_utils import get_analysis_name
from microsim.outcome_model_type import OutcomeModelType
from microsim.sim_settings import simSettings
import pandas as pd
import multiprocessing as mp

class Trialset:
    
    def __init__(self, 
                 trialDescription, pop, trialCount, 
                 additionalLabels=None): #additional labels are additional columns on the trialset results dataframe, eg risks used in inclusionFilter
        self.trialDescription = trialDescription
        self.pop = pop
        self.trialCount = trialCount 
        self.additionalLabels = additionalLabels  
    
    def prepareArgsForRun(self): #prepare all arguments needed to run the entire trial set 
        argsForRun = []	 #arguments will be stored in a list of tuples (multiprocessing map functions accept only tuples to be sent to processes)
        for iTrial in range(0,self.trialCount):  #for as many trials as the set asks for
                argsForRun.append((iTrial)) #trialset instances contain all information needed by their trials, so just pass a trial index
        return argsForRun
    
    #having the population as an argument passed to each trial is the reason why python makes copies of the population
    #when it sends information to the cores running the processes with multiprocessing
    #currently, this adds a RAM cost = (number of processes) * (population size in RAM)
    #which suggests a solution: do not pass the actual population to the function the processes run but
    #some kind of index/link/list/function as an interface to the population
    #also, to maximize efficiency with TrialsetParallel, a single core must prepare, run and analyze a trial
    def prepareRunAnalyzeTrial(self, iTrial):
        print(f'starting trial {iTrial} now', flush=True) #helps to see when trials start
        trial = Trial(self.trialDescription, 
                      self.pop, 
                      additionalLabels=self.additionalLabels) #initialize trial
        trial.run() #run trial
        resultsForTrial = [] #and now analyze the trial
        for analysis in trial.trialDescription.analyses:
            for duration in trial.trialDescription.durations:
                for sampleSize in trial.trialDescription.sampleSizes:
                    resultsForTrial.append(trial.analyticResults[get_analysis_name(analysis, duration, sampleSize)])
        dfForTrial = pd.DataFrame(resultsForTrial)
        if trial.additionalLabels is not None:
            for label, labelVal in trial.additionalLabels.items():
                dfForTrial[label] = labelVal
        del trial #trial instance is no longer needed, so release the memory (but python keeps track of references to objects anyway)
        print(f'ending trial {iTrial} now', flush=True) #helps to see when trials end
        return dfForTrial #return only results
    
class TrialsetParallel(Trialset): #Parallel refers to how trials are run, at any moment there are self.processesCount trials running in parallel

    #TrialsetParallel needs an extra argument than TrialsetSerial, the number of processes to be launched
    def __init__(self, trialDescription, pop, trialCount, processesCount, additionalLabels=None):
        self.processesCount = processesCount
        super().__init__(trialDescription, pop, trialCount, additionalLabels=additionalLabels)

    def run(self): 
        if simSettings.pandarallelFlag: #multiprocessing processes must not launch pandarallel
            raise Exception("pandarallelFlag must be set to False prior to running a TrialsetParallel instance")     
        else:
            with mp.Pool(self.processesCount) as myPool: #context manager will terminate this pool of processes
                 #run trials and get back the list of dataframes with the results (trial instance is not returned to save memory)
                 resultsTrialsetList = myPool.map(self.prepareRunAnalyzeTrial, self.prepareArgsForRun())
                 resultsTrialsetPd = pd.concat(resultsTrialsetList).reset_index(drop=True) #convert list of dataframes to a single dataframe
            return resultsTrialsetPd

class TrialsetSerial(Trialset): #Serial refers to how trials are run, at any moment there is only one trial running (which may or may not use pandarallel)

    def __init__(self, trialDescription, pop, trialCount, additionalLabels=None):
        super().__init__(trialDescription, pop, trialCount, additionalLabels=additionalLabels)

    def run(self):
        print(f'pandarallelFlag is set to {simSettings.pandarallelFlag}')
        resultsTrialsetList = []
        argsForRun = self.prepareArgsForRun()
        for iTrial in range(self.trialCount):
                resultsTrialsetList.append(self.prepareRunAnalyzeTrial(argsForRun[iTrial])) #prepare,run,analyze trial and append trial results to list
                if (iTrial+1) % 10 == 0:
                        print(f"#################\n#################    Trial Completed: {iTrial}")
        resultsTrialsetPd = pd.concat(resultsTrialsetList).reset_index(drop=True) #convert list of dataframes to a single dataframe
        return resultsTrialsetPd

