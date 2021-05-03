

import cma 
from evoalg.algorithms import algo_utils


# This is just a wrapper around pycma implementation of CMA ES
# This is maximizing, compared to pycma which is minimizing.
class CMA_ES():

    def __init__(self,get_random_initial_parameters_func,config):

        self.config = config
        self.get_rand_init = get_random_initial_parameters_func
        self.last_population = None
        self.current_generation = 0

        self.required_config_fields = [
            "CMA_popsize",
            "CMA_initial_sigma",
        ]

        if algo_utils.config_contains_required_fields(self.config,self.required_config_fields) is False:
            print("Required fields are: ",self.required_config_fields)
            raise "Error, ES missing required config"


        x0 = get_random_initial_parameters_func()
        
        # TODO set option for CMA
        self.cma = cma.CMA(x0)


    def ask(self):

        if self.last_population is not None:
            raise "Error, are you calling ask() twice without calling tell()?"

        solutions = self.cma.ask()

        # no encoding, because pycma does not use encoding and there is not much point anywhay, 
        # CMA cannot really scale to large parameters anyway (because of its algorithmic complexity: N^2), so encoding would not result in large gains...
        encoded_population = {
            "population" : solutions,
            "data_to_broadcast" : None, 
            "encoding_type" : None,
        }
        self.last_population = encoded_population
        return encoded_population




    def tell(self,results):
        if self.last_population is None:
            raise "Error, are you calling tell() without calling ask() first?"
        
        # because pycma is minimizing, multiply fitnesses by -1
        fitness_vec = [-1*res["fitness"] for res in results]
        self.cma.tell(self.last_population["population"], fitness_vec)
        self.last_population = None
        self.current_generation +=1

