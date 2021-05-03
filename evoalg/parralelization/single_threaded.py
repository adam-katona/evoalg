

from evoalg import decode_pop

def single_threaded_evaluate(encoded_pop,evaluate_task_func,get_random_initial_parameters_func,config,client=None):

    results = []
    for individual in encoded_pop["population"]:
        decoded_individual = decode_pop.decode_individual(individual = individual,
                                                          broadcasted_data = encoded_pop["data_to_broadcast"],
                                                          encoding_type = encoded_pop["encoding_type"],
                                                          get_initial_parameters_func = get_random_initial_parameters_func)

        result = evaluate_task_func(decoded_individual,config)
        results.append(result)

    return results


