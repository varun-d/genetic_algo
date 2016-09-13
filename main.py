import random
import string
from math import sqrt

TARGET = 'hospital'    # Target word or let's call it sequence of characters. For now only ASCII lowercase, no space or uppercase :)
POP_SIZE = 500     # Population size
MUTATION = 0.01     # Mutation proportion
population = []     # Our world as an array of member dictionaries, with 'member' and 'fit_score'. 'member' is a char seq (a word)
SD_MULTIPLIER = 1
ELITE = 0.2     # Elitism: keep top 20% in existing generation for next generation.
TARGET_BUILD = string.ascii_lowercase

# Manually calculating Standard Deviation. Don't want to import any library.
def calc_std(num_list):
    _sum = sum(num_list)
    _N = len(num_list)
    _mean = _sum / float(_N)

    ss = 0  # Sum of squares
    for xi in num_list:
        deviation = xi - _mean
        ss += (deviation * deviation)

    return sqrt (ss / float(_N))

# Manually calculating Average. Don't want to import any library.
def calc_mean(num_list):
    _sum = sum(num_list)
    _N = len(num_list)
    _mean = _sum / float(_N)
    return _mean


# Inception of time. Generate random words using only lowercase ASCII.
def generate_population(pop_list, size=len(TARGET), chars=TARGET_BUILD):
    for _ in range(0, POP_SIZE):
        pop_list.append( {'_name': ''.join(random.choice(chars) for _ in range(size)), '_fit_score': 0} )
    return pop_list

# Calculates fitness score based on the following
# 1. If a char is in the TARGET
# 2. If there is a match -> if position is same as TARGET position add 0.5 more to fitness score.

def calc_fitness_score(word):
    char_list = list(word)
    target_list = list(TARGET)
    fit_score = 0

    for i in range(len(TARGET)):
        if char_list[i] == target_list[i]:
            fit_score += 1

    return fit_score / float(len(TARGET))


def natural_selection(population_list, elite):
    """
    Takes in current population generation and sends back new generation with children and previous population.
    :param population_list:
    :return:
    """
    _mate_pool = []     # The array holding members, the number of times their probability, used for random selection.
    _natural_selection = []     # The new generation, which will be returned
    all_fitness_scores = [m['_fit_score'] for m in population_list]

    fmax = max(all_fitness_scores)

    for member in population_list:
        f_prob = (member['_fit_score']/fmax) * 100  # fi/fmax * 100 gives us the probability. Dump it all in a bucket for easy selection.
        for i in range(int(f_prob)): # Duplicate each member the number of times their f_prob. I'm sure there's a better way to do this.
            _mate_pool.append(member)

    # Get top ELITE members into _natural_selection before doing random baby making
    sorted_population = sorted(population_list, key=lambda k: k['_fit_score'])
    elite_range = int(len(sorted_population) * elite)
    _natural_selection =  sorted_population[-elite_range:] # Get from last sorted awesome members, these are in asc order.


    # Start mating process here. Put on some baby making music.
    for _ in range(POP_SIZE - len(_natural_selection)): # To keep our new population under control, conduct control mating.
        parent_member_A = random.choice(_mate_pool)
        parent_member_B = random.choice(_mate_pool)
        child_name = get_mate_child(parent_member_A['_name'], parent_member_B['_name'], 2)
        child_fitness = calc_fitness_score(child_name)
        _natural_selection.append({'_name': child_name, '_fit_score': child_fitness})

    return _natural_selection

# A two parent mating process. With a mutation option if you select type == 2
def get_mate_child (mem1, mem2, type):
    _half_target = len(TARGET)/2    # First half of our member
    _remaining_target = len(TARGET) - len(TARGET)/2     # Second half of our member

    if type == 1: # Try Algo 1: First two chars + last 3 chars with NO mutation
        parent_1 = list(mem1)[:_half_target] # First half of our TARGET word
        parent_2 = list(mem2)[-_remaining_target:] # Remaining half of our TARGET
        new_member_chars = parent_1 + parent_2
        return ''.join(new_member_chars)    # Return a combined word

    elif type == 2: # This two parent algo uses mutation. How do I introduce mutation proportion?

        parent_1 = list(mem1)[:_half_target]  # First half of our TARGET word
        parent_2 = list(mem2)[-_remaining_target:]  # Remaining half of our TARGET

        new_member_chars = parent_1 + parent_2
        new_member_chars.remove(random.choice(new_member_chars))  # Remove random char. todo, can this be a problem?
        new_member_chars.append(random.choice(TARGET_BUILD))    # Add random random char todo, we aren't adding a new char to the same location
        return ''.join(new_member_chars)  # Return a combined word


if __name__ == '__main__':

    generation = 0      # Variable to keep track of generations
    population = generate_population(population)        # Generate first random population

    # Generate fitness score for members in our population
    pop_with_score = []
    for member in population:
        pop_with_score.append(
            {'_name': member['_name'],
             '_fit_score': calc_fitness_score(member['_name'])
             }
        )

    population = pop_with_score     # Rename our population with fit scores back to 'population'

    # Uncomment below to print your initial population with fitness scores
    # for member in population:
    #     print member['_name'] + "  " + str(member['_fit_score'])
    #
    # exit("Check Fitness")

    # Try-Except block to print out population data if we have to stop our algorithm because of non-convergence
    try:
        while not any(member['_name'] == TARGET for member in population):
            # Important! The above statement means stop when *any* population member matches our target

            # The main function that does selection, mating and returns new population generation
            population = natural_selection(population, ELITE)

            # Get *list* of all fitness scores of current generation
            all_fitness_scores = [ m['_fit_score'] for m in population ]

            # Increment our generation number
            generation += 1

            # Uncomment following statements for debug
            # print ("Running generation number: %d" % generation)
            # print ("Mean: %f, SD: %f." % (calc_mean(all_fitness_scores), calc_std(all_fitness_scores)))
            print ("%f" % calc_mean(all_fitness_scores))        # Print total fitness mean for each generation
    except KeyboardInterrupt:
        # In stopping a non-converging GA, print the last known population with fitness scores
        for member in population:
            print member['_name'] + "  " + str(member['_fit_score'])

    # On termination condition (success) print last population with fitness scores
    for member in population:
        print member['_name'] + "  " + str(member['_fit_score'])

