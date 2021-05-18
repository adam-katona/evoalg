
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

        self.use_population_distribution_encoding = False
        if "USE_POPULATION_DISTRIBUTION_ENCODING" in config:
            if config["USE_POPULATION_DISTRIBUTION_ENCODING"] is True:
                self.use_population_distribution_encoding = True

        if algo_utils.config_contains_required_fields(self.config,self.required_config_fields) is False:
            print("Required fields are: ",self.required_config_fields)
            raise "Error, GA missing required config"


    def _get_rand_seed(self):
        return np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max)

    def _get_initial_pop_encoded(self):
        return [[rand_seed] for rand_seed in np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, size=self.config["GA_popsize"])]

    def _get_initial_pop_direct(self):
        return [self.get_rand_init(self.config,self._get_rand_seed()) for _ in range(self.config["GA_popsize"])]

    def ask(self):

        if self.current_population is None:
            # the first generation
            if self.use_population_distribution_encoding is True:
                self.current_population = self._get_initial_pop_encoded()
            else:
                self.current_population = self._get_initial_pop_direct()
        else:
            if self.next_population is None:
                raise "Error, are you calling ask() twice without calling tell()?"
            else:
                self.current_population = self.next_population
                self.next_population = None


        if self.use_population_distribution_encoding is True:
            encoded_population = {
                "population" : self.current_population,
                "data_to_broadcast" : {
                    "mutation_power" : self.config["GA_mutation_power"],
                },
                "encoding_type" : "init_seed_and_mutation_seeds",
            }
        else:
            encoded_population = {
                "population" : self.current_population,
                "data_to_broadcast" : None,
                "encoding_type" : None,
            }

        return encoded_population


    def tell(self,results):

        if self.current_population is None:
            raise "Error, are you calling tell() without calling ask() first?"

        fitness_vec = [res["fitness"] for res in results]
        order = np.argsort(fitness_vec)
        ordered_pop = [self.current_population[i] for i in order]

        # get elites
        elites = ordered_pop[-self.config["GA_num_elites"]:]

        # get babies
        num_parents = int(self.config["GA_popsize"] * self.config["GA_allowed_reproduce_ratio"])  # turncated selection (like in deep ga paper)
        num_babies = self.config["GA_popsize"] - self.config["GA_num_elites"]
        parents = ordered_pop[-num_parents:]

        if self.use_population_distribution_encoding is True:
            babies = [copy.deepcopy(parent).append(self._get_rand_seed()) for parent in np.random.choice(parents, size=num_babies, replace=True)]
            babies = [copy.deepcopy(parents[parent_i]).append(self._get_rand_seed()) for parent_i in np.random.choice(len(parents), size=num_babies, replace=True)]
        else:
            babies = [np.copy(parents[parent_i]) + np.random.randn(parents[parent_i].size) * self.config["GA_mutation_power"] for parent_i in np.random.choice(len(parents), size=num_babies, replace=True)]

        new_population = []
        new_population.extend(elites)
        new_population.extend(babies)

        self.next_population = new_population



