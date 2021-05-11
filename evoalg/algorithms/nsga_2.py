
import torch
import numpy as np
import copy

from evoalg.algorithms import algo_utils
from evoalg.algorithms import pareto


class NSGA_II():

    def __init__(self,get_random_initial_parameters_func,config):

        self.config = config
        self.get_rand_init = get_random_initial_parameters_func
        self.current_population = None
        self.next_population = None
        self.current_generation = 0

        self.required_config_fields = [
            "NSGA_II_pop_size",
            "NSGA_II_mutation_power",
            "NSGA_II_num_elites",
            "NSGA_II_allowed_reproduce_ratio",
        ]

        if algo_utils.config_contains_required_fields(self.config,self.required_config_fields) is False:
            print("Required fields are: ",self.required_config_fields)
            raise "Error, NSGA_II missing required config"


    def ask(self):

        if self.current_population is None:
            # this is the first generation
            self.current_population = [self.get_rand_init(self.config) for _ in range(self.config["NSGA_II_pop_size"])] 

        else:
            if self.next_population is None:
                raise "Error, are you calling ask() twice without calling tell()?"
            else:
                self.current_population = self.next_population
                self.next_population = None


        encoded_population = {
            "population" : self.current_population,
            "data_to_broadcast" : None,
            "encoding_type" : None,
        }

        return encoded_population



    def tell(self,results):

        if self.current_population is None:
            raise "Error, are you calling tell() without calling ask() first?"

        fitnesses = np.array([res["fitnesses"] for res in results])

        fronts = pareto.calculate_pareto_fronts(fitnesses)
        nondomination_rank_dict = pareto.fronts_to_nondomination_rank(fronts)
        
        crowding = pareto.calculate_crowding_metrics(fitnesses,fronts)
        
        # Sort the population
        non_domiated_sorted_indicies = pareto.nondominated_sort(nondomination_rank_dict,crowding)
        ordered_pop = [self.current_population[i] for i in non_domiated_sorted_indicies]

        # get elites
        elites = ordered_pop[-self.config["NSGA_II_num_elites"]:]

        # get babies
        num_parents = int(self.config["NSGA_II_pop_size"] * self.config["NSGA_II_allowed_reproduce_ratio"])  # turncated selection 
        num_babies = self.config["NSGA_II_pop_size"] - self.config["NSGA_II_num_elites"]
        parents = ordered_pop[-num_parents:]
        babies = [parents[parent_i] + np.random.randn(parents[parent_i].size)*self.config["NSGA_II_mutation_power"]  for parent_i in np.random.choice(len(parents), size=num_babies, replace=True)]

        new_population = []
        new_population.extend(elites)
        new_population.extend(babies)

        self.next_population = new_population