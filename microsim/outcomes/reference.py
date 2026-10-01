from microsim.outcomes.outcome import OutcomeType

class Reference:

    incidence = {
        #https://vizhub.healthdata.org/gbd-compare/
        #advanced settings, choose idiopathic epilepsy as the cause and measure the incidence (new cases), choose USA as location, 2023
        OutcomeType.EPILEPSY.value:
            {"15-19": 36.47/100000, #new cases per 100,000
             "20-24": 32.39/100000,
             "25-29": 28.9/100000,
             "30-34": 25.86/100000,
             "35-39": 23.27/100000,
             "40-44": 22/100000,
             "45-49": 22.03/100000,
             "50-54": 24.2/100000,
             "55-59": 28.45/100000,
             "60-64": 30.6/100000,
             "65-69": 28.99/100000,
             "70-74": 30.53/100000,
             "75-79": 35.27/100000,
             "80-84": 38.16/100000,  
             "85-89": 39.19/100000,  
             "90-94": 43.53/100000,  
             "95-99": 51.47/100000}
    }
    
    prevalence = {
        #https://vizhub.healthdata.org/gbd-compare/
        #cardiovascular diseases as the cause, prevalence as the measure, USA, 1999
        OutcomeType.CARDIOVASCULAR.value:
            {"male":
                {"50-54": 0.173,
                 "55-59": 0.262,
                 "60-64": 0.372,
                 "65-69": 0.495,
                 "70-74": 0.604},
             "female":
                {"50-54": 0.121,
                 "55-59": 0.176,
                 "60-64": 0.253,
                 "65-69": 0.356,
                 "70-74": 0.459}},
        #https://vizhub.healthdata.org/gbd-compare/
        #stroke as the cause, prevalence as the measure, USA, 1999
        OutcomeType.STROKE.value:
            {"male":
                {"50-54": 0.0217,
                 "55-59": 0.0315,
                 "60-64": 0.0448,
                 "65-69": 0.0608,
                 "70-74": 0.0793},
             "female":
                {"50-54": 0.0208,
                 "55-59": 0.0289,
                 "60-64": 0.0397,
                 "65-69": 0.0529,
                 "70-74": 0.0681}},
        #https://vizhub.healthdata.org/gbd-compare/
        #myocardial infarction as the cause, prevalence as the measure, USA, 1999
        #reference only: the MI prevalence partition model is deliberately not fit to these
        OutcomeType.MI.value:
            {"male":
                {"50-54": 0.0678,
                 "55-59": 0.106,
                 "60-64": 0.151,
                 "65-69": 0.203,
                 "70-74": 0.255},
             "female":
                {"50-54": 0.0302,
                 "55-59": 0.0460,
                 "60-64": 0.0662,
                 "65-69": 0.0928,
                 "70-74": 0.124}},
        #same source as the incidence for epilepsy
        OutcomeType.EPILEPSY.value:
            {"15-19": 357.92/100000,
             "20-24": 389.97/100000,
             "25-29": 410.65/100000,
             "30-34": 421.19/100000,
             "35-39": 420.72/100000,
             "40-44": 414.28/100000,
             "45-49": 406.95/100000,
             "50-54": 405.5/100000,
             "55-59": 426.42/100000,
             "60-64": 471.44/100000,
             "65-69": 518.82/100000,
             "70-74": 558.18/100000,
             "75-79": 606.17/100000,
             "80-84": 661.24/100000,
             "85-89": 713.16/100000,
             "90-94": 764.64/100000,
             "95-99": 834.57/100000}
    }
   
