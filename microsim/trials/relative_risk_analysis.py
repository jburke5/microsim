from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
import numpy as np

class RelativeRiskAnalysis:
    #one name per element of the tuple returned by analyze, in order
    columns = ("relativeRisk", "rrCiLow", "rrCiUpp",
               "tRisk", "tRiskCiLow", "tRiskCiUpp", "tRiskCiLowWilson", "tRiskCiUppWilson",
               "cRisk", "cRiskCiLow", "cRiskCiUpp", "cRiskCiLowWilson", "cRiskCiUppWilson",
               "riskDiffx100", "rdCiLowx100", "rdCiUppx100", "tEfficiencyx100")

    def __init__(self):
        pass
    
    def inverse_logit(self, lp):
        if lp<-10:
            risk = 0.
        elif lp>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-lp))
        return risk

    def get_absolute_risk(self, nSuccesses, nTotal):
        if nTotal>0.:
            risk = nSuccesses/nTotal
            if nSuccesses == 0 or nSuccesses == nTotal:
                ciLower = None
                ciUpper = None
            else:
                #will use this transformation to ensure that the confidence interval is bounded between 0 and 1
                logitRisk = np.log( risk / (1.-risk) )
                #standard error of logit of R, see chapter 17, Modern Epidemiology
                seLogitRisk = np.sqrt( 1/nSuccesses + 1./(nTotal-nSuccesses) )
                ciLowerLogit = logitRisk - 1.96*seLogitRisk #while this is a linear transformation
                ciUpperLogit = logitRisk + 1.96*seLogitRisk
                ciLower = self.inverse_logit(ciLowerLogit) #this is not, so I cannot just report the SE
                ciUpper = self.inverse_logit(ciUpperLogit) #I need to report both lower and upper points
            #wilson score is better than the normal approximation to get the CI, fyi the midpoint of the wilson interval might be different from MLE
            ciLowerWilson, ciUpperWilson = proportion_confint(count=nSuccesses, nobs=nTotal, alpha=0.05, method='wilson')
            return risk, ciLower, ciUpper, ciLowerWilson, ciUpperWilson
        else:
            return float('nan'), None, None, float('nan'), float('nan')

    def get_risk_ratio_ci(self, nSuccessesTreated, nTotalTreated, nSuccessesControl, nTotalControl):
        ciLow, ciUpp = confint_proportions_2indep(count1=nSuccessesTreated, nobs1=nTotalTreated, count2=nSuccessesControl, nobs2=nTotalControl,     
                                                  compare='ratio', method='score', alpha=0.05)
        return ciLow, ciUpp

    def get_risk_difference_ci(self, nSuccessesTreated, nTotalTreated, nSuccessesControl, nTotalControl):
        ciLow, ciUpp = confint_proportions_2indep(count1=nSuccessesTreated, nobs1=nTotalTreated, count2=nSuccessesControl, nobs2=nTotalControl,
                                                  compare='diff', method='newcomb', alpha=0.05)
        return ciLow, ciUpp

    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        assessmentFunction = assessmentFunctionDict["outcome"]
        treatedCounts = list(map(assessmentFunction, [trial.treatedPop]))[0] #an integer
        controlCounts = list(map(assessmentFunction, [trial.controlPop]))[0] #an integer
        #arm sizes can differ, eg with bernoulli or block randomization, so each arm uses its own denominator
        nTotalTreated = trial.treatedPop._n
        nTotalControl = trial.controlPop._n
        tRisk, tRiskCiLower, tRiskCiUpper, tRiskCiLowerWilson, tRiskCiUpperWilson = self.get_absolute_risk(treatedCounts, nTotalTreated) #treated
        cRisk, cRiskCiLower, cRiskCiUpper, cRiskCiLowerWilson, cRiskCiUpperWilson = self.get_absolute_risk(controlCounts, nTotalControl) #control
        tAnyMedsAdded = trial.treatedPop.has_any_meds_added() #alive, treated, is None when person is not in any treatment strategies
        tAnyMedsAdded = list(filter(lambda x: x is not None, tAnyMedsAdded)) #filter out the Nones
        tProportionWithMedsAdded = sum(tAnyMedsAdded)/len(tAnyMedsAdded) if len(tAnyMedsAdded)>0 else 0
        diff = tRisk-cRisk #definition is treated - control, consistent with confint_proportions_2indep
        tEfficiency = diff/tProportionWithMedsAdded if tProportionWithMedsAdded!=0 else float('nan') #undefined when no meds were added
        rdCiLow, rdCiUpp = self.get_risk_difference_ci(treatedCounts, nTotalTreated, controlCounts, nTotalControl)
        scaleFactor = 100.
        if cRisk!=0.:
            relativeRisk = tRisk/cRisk
            rrCiLow, rrCiUpp = self.get_risk_ratio_ci(treatedCounts, nTotalTreated, controlCounts, nTotalControl)
            return (relativeRisk, rrCiLow, rrCiUpp,
                    tRisk, tRiskCiLower, tRiskCiUpper, tRiskCiLowerWilson, tRiskCiUpperWilson, #treated
                    cRisk, cRiskCiLower, cRiskCiUpper, cRiskCiLowerWilson, cRiskCiUpperWilson, #control
                    diff*scaleFactor, rdCiLow*scaleFactor, rdCiUpp*scaleFactor, tEfficiency*scaleFactor)
        else:
            return (float('inf'), float('inf'), float('inf'), 
                   tRisk, tRiskCiLower, tRiskCiUpper, tRiskCiLowerWilson, tRiskCiUpperWilson, #treated
                   cRisk, cRiskCiLower, cRiskCiUpper, cRiskCiLowerWilson, cRiskCiUpperWilson, #control
                   diff*scaleFactor, rdCiLow*scaleFactor, rdCiUpp*scaleFactor, tEfficiency*scaleFactor)
