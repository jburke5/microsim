from typing import List

import pandas as pd

from microsim.person import Person
from microsim.population import Population


def init_vectorized_population_dataframe(person_list: List[Person]):
    people = pd.Series(person_list)
    population = Population(people)
    return population.get_people_current_state_and_summary_as_dataframe()
