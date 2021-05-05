
import torch
import numpy as np


from evoalg.algorithms import algo_utils





# Maximizing fitness
class ES():

    def __init__(self,get_random_initial_parameters_func,config):

        self.config = config
        self.get_rand_init = get_random_initial_parameters_func
        self.last_population = None
        self.current_generation = 0

        self.required_config_fields = [
            "ES_popsize",
            "ES_sigma",
            "ES_lr",
        ]

        if algo_utils.config_contains_required_fields(self.config,self.required_config_fields) is False:
            print("Required fields are: ",self.required_config_fields)
            raise "Error, ES missing required config"

        initial_theta = self.get_rand_init(config)
        initial_theta = torch.from_numpy(initial_theta)

        self.num_params = initial_theta.shape[0]

        self.theta = torch.nn.Parameter(initial_theta) 
        self.optimizer = torch.optim.Adam([self.theta],lr=self.config["ES_lr"])



    def ask(self):

        if self.last_population is not None:
            raise "Error, are you calling ask() twice without calling tell()?"

        # create next population

        from evoalg import random_table
        noise_table = random_table.noise_table

        # Use half pop sized because of mirrored sampling
        HALF_POP_SIZE = self.config["ES_popsize"] // 2

        random_table_indicies = [random_table.get_random_index(self.num_params) for _ in range(HALF_POP_SIZE)]
        pop_list = [(rand_i,1) for rand_i in random_table_indicies] # (rand_index,direction)
        pop_list.extend([(rand_i,-1) for rand_i in random_table_indicies]) # mirrored sampling

        encoded_population = {
            "population" : pop_list,
            "data_to_broadcast" : {
                "center_individual" : self.theta.data,
                "sigma" : self.config["ES_sigma"],
            },  
            "encoding_type" : "rand_table_mutate",
        }
        self.last_population = encoded_population

        return encoded_population


    def tell(self,results):

        if self.last_population is None:
            raise "Error, are you calling tell() without calling ask() first?"

        from evoalg import random_table
        noise_table = random_table.noise_table

        fitness_vec = [res["fitness"] for res in results]
        sorted_indicies = np.argsort(fitness_vec) 
        # np.argsort does ascending, we want the gradient to point towards the high fitness, the first index gets the lowest score
        all_ranks = np.linspace(-0.5,0.5,len(sorted_indicies)) 
        perturbation_ranks = np.zeros(len(sorted_indicies))
        perturbation_ranks[sorted_indicies] = all_ranks
        perturbation_ranks = torch.from_numpy(perturbation_ranks).float()

        perturbation_array = [noise_table[rand_i:rand_i+self.num_params]*direction for rand_i,direction in self.last_population["population"]]
        perturbation_array = torch.stack(perturbation_array)

        grad = torch.matmul(perturbation_ranks,perturbation_array)  # ES update, calculate the weighted sum of the perturbations
        grad = grad / self.config["ES_popsize"] / self.config["ES_sigma"]

        self.theta.grad = -grad # we are maximizing, but torch optimizer steps in the opposite direction of the gradient, multiply by -1 so we can maximize.
        self.optimizer.step()


        self.last_population = None
        self.current_generation += 1



    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
    