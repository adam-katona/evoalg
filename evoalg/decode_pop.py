

import numpy as np
import torch

# NOTE: there are 2 ways to encode random parameters,
# 1: is to send a random seed (no memory cost, some cpu cost)
# 2: is to send random table indicies (high memory cost (each worker needs a copy of the random table, no cpu cost)

def decode_rand_table_mutate_individual(individual,broadcasted_data):
    rand_table_i,sign = individual
    center_individual = broadcasted_data["center_individual"]
    sigma = broadcasted_data["sigma"]

    parameter_size = center_individual.numel()

    from evoalg import random_table
    noise_table = random_table.noise_table

    mutation_noise = sign * noise_table[rand_table_i:rand_table_i+parameter_size] * sigma
    decoded_individual = center_individual + mutation_noise

    return decoded_individual


def get_deterministic_random_noise(seed,size):
    np.random.seed(seed)
    noise = torch.from_numpy(np.random.randn(size).astype(np.float32))
    np.random.seed()
    return noise

def decode_init_seed_and_mutation_seeds_individual(individual,broadcasted_data,get_random_initial_parameters_func):

    mutation_power = broadcasted_data["mutation_power"]

    initial_seed = individual[0]
    decoded_individual = get_random_initial_parameters_func(seed=initial_seed)
    size = decoded_individual.numel()

    for mutation_seed in individual[1:]:
        noise = get_deterministic_random_noise(mutation_seed,size)
        decoded_individual += noise * mutation_power

    return decoded_individual




def decode_individual(individual,broadcasted_data,encoding_type,get_random_initial_parameters_func):
    if encoding_type is None:
        # the individual is not encoded, return as it is
        return individual
    elif encoding_type == "init_seed_and_mutation_seeds":
        return decode_init_seed_and_mutation_seeds_individual(individual,broadcasted_data,get_random_initial_parameters_func)
        
    elif encoding_type == "rand_table_mutate":
        return decode_rand_table_mutate_individual(individual,broadcasted_data)
    else:
        raise "Unkown encoding_type!"


# ususally this function is not called, since the point of encoding the population is that we can distribute it efficiently, by letting the workers decode them individually.
# decoding them all in one place defeats the purpuse...
def decode_population(encoded_pop):
    return [decode_individual(individual,encoded_pop["data_to_broadcast"],encoded_pop["encoding_type"]) for individual in encoded_pop["population"] ]