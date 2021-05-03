

import dask 
import itertools

from evoalg import decode_pop


# the function run by the worker
def _dask_run(individual,shared_data):
    decoded_individual = decode_pop.decode_individual(individual = individual,
                                                      broadcasted_data = shared_data["data_to_broadcast"],
                                                      encoding_type = shared_data["encoding_type"],
                                                      get_random_initial_parameters_func = shared_data["get_random_initial_parameters_func"])
    
    evaluate_task_func = shared_data["evaluate_task_func"]
    result = evaluate_task_func(decoded_individual,shared_data["config"])
    return result


def dask_evaluate(encoded_pop,evaluate_task_func,get_random_initial_parameters_func,config,client):

    if client is None:
        raise "Error, when using dask_evaluate, you must provide a dask client"

    # this is shared for all workers, will be serialized once and broadcasted
    shared_data = {
        "data_to_broadcast" : encoded_pop["data_to_broadcast"],
        "encoding_type" : encoded_pop["encoding_type"],
        "evaluate_task_func" : evaluate_task_func,
        "get_random_initial_parameters_func" : get_random_initial_parameters_func,
        "config" : config,
    }

    shared_data_future = client.scatter(shared_data,broadcast=True)
    result_futures = client.map(_dask_run,encoded_pop["population"],itertools.repeat(shared_data_future))
    results = client.gather(result_futures)

    return results





