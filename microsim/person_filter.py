from microsim.cv_model_repository import CVModelRepository
from microsim.dementia_model_repository import DementiaModelRepository
from microsim.treatment import DefaultTreatmentsType
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType

class PersonFilter:
    '''This class holds filters for the dataframe with the information used in creating Person objects and filters 
    for Person objects themselves.
    The filters dictionary has two keys, one for the dataframe, df,  one for the Person object, person. 
    The df filters will be applied directly on the dataframe where the information for the Person objects comes from.
    The person filters will be applied once the Person object is created, which is at a later point than the df filters.
    The goal with having filters that can be applied at the df level is to save resources (it takes both memory and time to create Person objects).
    Some times though, we need to wait until after we create the Person object in order to apply the filter (eg when the filter is based
    on a model that can only be applied to a Person object).
    The values of the filters dictionary are dictionaries themselves.
    Each dictionary, has values of the format: filterName, filterFunction.
    filterName can be any string, unique to that filter.
    For a person filter, filterFunction must be a Person class function that returns True/False.
    For a df filter, filterFunction must be a function that can be applied on a dataframe row and return True/False.
    filters["df"]: keys are filter names, values are functions that can be applied on a dataframe row
    filters["person"]: keys are filter names, values are functions that can be applied on Person objects'''
    def __init__(self):
        self.filters = {"df": dict(),
                        "person": dict()}
        
    def add_filter(self, filterType = "person", filterName = "all", filterFunction = lambda x: True):
        self.filters[filterType][filterName] = filterFunction
    
    def rm_filter(self, filterType = "person", filterName = "all"):
        del self.filters[filterType][filterName]
    
    def __str__(self):
        rep = "Person Filters:\n"
        rep += f"\t{'filter type':>15}   {'filter name':<15}\n"
        for filterType in self.filters.keys():
            for filterName in self.filters[filterType].keys():
                rep += f"\t{filterType:>15}   {filterName:<15}\n"
        return rep
                
    def __repr__(self):
        return self.__str__()
