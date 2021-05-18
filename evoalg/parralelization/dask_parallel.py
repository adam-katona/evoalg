

import dask 
import itertools

from evoalg import decode_pop
import torch

import sys
sys.path.append("/users/ak1774/scratch/NCA/evocraft-cellular-automata")
sys.path.append("/users/ak1774/scratch/NCA/brain-tokyo-workshop/WANNRelease/WANN")

# the function run by the worker
def _dask_run(individual,shared_data):
    torch.set_num_threads(1)

    from pydoc import locate
    evaluate_func = locate(shared_data["config"]["TASK_EVALUATE_FUN"])
    get_random_parameter_func = locate(shared_data["config"]["TASK_INIT_FUN"])

    decoded_individual = decode_pop.decode_individual(individual = individual,
                                                      broadcasted_data = shared_data["data_to_broadcast"],
                                                      encoding_type = shared_data["encoding_type"],
                                                      get_random_initial_parameters_func = get_random_parameter_func,
                                                      config = shared_data["config"])

    result = evaluate_func(decoded_individual,shared_data["config"])
    
    return result

# DEBUG ###########################
#def _dask_run(individual,shared_data,config):
#    torch.set_num_threads(1)
#    
#    from pydoc import locate
#    import sys
#    sys.path.append("/users/ak1774/scratch/NCA/evocraft-cellular-automata")
#    evaluate_func = locate(config["TASK_EVALUATE_FUN"])
#    get_random_parameter_func = locate(config["TASK_INIT_FUN"])
#
#    decoded_individual = decode_pop.decode_individual(individual = individual,
#                                                      broadcasted_data = shared_data,
#                                                      encoding_type = "rand_table_mutate",
#                                                      get_random_initial_parameters_func = get_random_parameter_func)
#
#    result = evaluate_func(decoded_individual,config)
#    return result





# TODO currently this does not work, there are mysterious dask warnings, and the code is stuck somewhere
# Need to figure out why
def dask_evaluate(encoded_pop,evaluate_task_func,get_random_initial_parameters_func,config,client):

    if client is None:
        raise "Error, when using dask_evaluate, you must provide a dask client"

    # this is shared for all workers, will be serialized once and broadcasted
    shared_data = {
        "data_to_broadcast" : encoded_pop["data_to_broadcast"],
        "encoding_type" : encoded_pop["encoding_type"],
        #"evaluate_task_func" : evaluate_task_func,
        #"get_random_initial_parameters_func" : get_random_initial_parameters_func,
        "config" : config,
    }

    #shared_data_future = client.scatter(encoded_pop["data_to_broadcast"],broadcast=True)
    #result_futures = client.map(_dask_run,encoded_pop["population"],itertools.repeat(shared_data_future),itertools.repeat(config))

    shared_data_future = client.scatter(shared_data,broadcast=True)
    result_futures = client.map(_dask_run,encoded_pop["population"],itertools.repeat(shared_data_future))
    results = client.gather(result_futures)

    return results


# to debug simple case
def dask_evaluate_simple(encoded_pop,evaluate_task_func,config,client):
    result_futures = client.map(evaluate_task_func,encoded_pop["population"],itertools.repeat(config))
    results = client.gather(result_futures)

    return results


