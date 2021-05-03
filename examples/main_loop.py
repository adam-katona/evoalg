

import numpy as np
import time

from evoalg.parralelization import dask_parallel
from evoalg.algorithms import ES


def main():

    config = {}

    # 1. Choose a task
    #evaluate_func = 
    #get_random_parameter_func = 


    # 2. Choose a parralelization strategy and init cluster
    parallel_evaluate = dask_parallel.dask_evaluate

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=20)
    client = Client(cluster)


    # 3. Choose and configure an algorithm
    config["ES_popsize"] = 500
    config["ES_sigma"] = 0.1
    config["ES_lr"] = 0.01
    algo = ES.ES(get_random_parameter_func,config)

    # 4. Run evolution
    MAX_GENERATION = 1000
    for generation_num in range(MAX_GENERATION):
        now = time.time()

        pop_enocded = algo.ask()
        results = parallel_evaluate(pop_enocded,evaluate_func,get_random_parameter_func,config,client)
        algo.tell(results)

        fitnesses = np.array([res["fitness"] for res in results])
        print(generation_num," mean: ",np.mean(fitnesses)," max: ",np.max(fitnesses)," time: ",time.time()-now)

















