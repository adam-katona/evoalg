
import numpy as np
import copy

from evoalg.algorithms import algo_utils






# Maximizing fitness
class GA():

    def __init__(self,get_random_initial_parameters_func,config):
        
        self.config = config
        self.get_rand_init = get_random_initial_parameters_func
        self.current_population = None
        self.next_population = None
        self.current_generation = 0

        self.required_config_fields = [
            "GA_popsize",
            "GA_mutation_power",
            "GA_num_elites",
            "GA_allowed_reproduce_ratio",
        ]

        if algo_utils.config_contains_required_fields(self.config,self.required_config_fields) is False:
            print("Required fields are: ",self.required_config_fields)
            raise "Error, GA missing required config"


    def _get_rand_seed(self):
        return np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max)

    def ask(self):

        if self.current_population is None:
            # this is the first generation
            self.current_population = [[rand_seed]  for rand_seed in [np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, size=config["GA_popsize"])]] 

        else:
            if self.next_population is None:
                raise "Error, are you calling ask() twice without calling tell()?"
            else:
                self.current_population = self.next_population
                self.next_population = None

        encoded_population = {
            "population" : self.current_population,
            "data_to_broadcast" : {
                "mutation_power" : config["GA_mutation_power"],
            },
            "encoding_type" : "init_seed_and_mutation_seeds",
        }

        return encoded_population


    def tell(self,results):

        if self.current_population is None:
            raise "Error, are you calling tell() without calling ask() first?"

        fitness_vec = [res["fitness"] for res in results]
        order = np.argsort(fitness_vec)
        ordered_pop = self.current_population[order]

        # get elites
        elites = ordered_pop[-config["GA_num_elites"]:]

        # get babies
        num_parents = int(config["GA_popsize"] * config["GA_allowed_reproduce_ratio"])  # turncated selection (like in deep ga paper)
        num_babies = config["GA_popsize"] - config["GA_num_elites"]
        parents = ordered_pop[-num_parents:]
        babies = [copy.deepcopy(parent).append(self._get_rand_seed()) for parent in np.random.choice(parents, size=num_babies, replace=True)]

        new_population = []
        new_population.extend(elites)
        new_population.extend(babies)

        self.next_population = new_population



