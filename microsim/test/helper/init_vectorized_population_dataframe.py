from typing import List

import pandas as pd

from microsim.person import Person
from microsim.population import Population
from microsim.gcp_model import GCPModel


def init_vectorized_population_dataframe(person_list: List[Person], *, with_base_gcp=False):
    if with_base_gcp:
        gcp_model = GCPModel()
        for p in person_list:
            if len(p._gcp) == 0:
                base_gcp = gcp_model.calc_linear_predictor(p)
                p._gcp.append(base_gcp)
    people = pd.Series(person_list)
    population = Population(people)
    return population.get_people_current_state_and_summary_as_dataframe()
