import random
import string
from math import sqrt

TARGET = 'varun'    # Target word or let's call it sequence of characters. For now only ASCII lowercase, no space or uppercase :)
POP_SIZE = 1000     # Population size
MUTATION = 0.01     # Mutation proportion
population = []     # Our world as an array of member dictionaries, with 'member' and 'fit_score'. 'member' is a char seq (a word)
SD_MULTIPLIER = 1
TARGET_BUILD = string.ascii_lowercase

# Manually calculating Standard Deviation. Don't want to import any library.
def calc_std(num_list):
    """
    Accepts list of numbers and returns a single number
    :param num_list: list of numbers, [1,2,3,4]
    :return: Population Standard Dev (/N)
    """
    _sum = sum(num_list)
    _N = len(num_list)
    _mean = _sum / float(_N)

    ss = 0  # Sum of squares
    for xi in num_list:
        deviation = xi - _mean
        ss += (deviation * deviation)

    return sqrt (ss / float(_N))


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


def natural_selection(population_list):
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
        if f_prob > 0:  # Add all members with prob > 0 back to the pool. todo I'm sure this will create problems.
            _natural_selection.append(member)

        for i in range(int(f_prob)): # Duplicate each member the number of times their f_prob. I'm sure there's a better way to do this.
            _mate_pool.append(member)

    # Start mating process here
    for _ in range(POP_SIZE - len(_natural_selection)): # To keep our new population under control, conduct control mating.
        parent_member_A = random.choice(_mate_pool)
        parent_member_B = random.choice(_mate_pool)
        child_name = get_mate_child(parent_member_A['_name'], parent_member_B['_name'], 2)
        child_fitness = calc_fitness_score(child_name)
        _natural_selection.append({'_name': child_name, '_fit_score': child_fitness})

    return _natural_selection

def get_best_population(population_list):
    """
    Returns truncated population
    :param population_list: main population
    :return: new truncated population
    """
    new_trunc_pop = []
    all_fitness_scores = [ m['_fit_score'] for m in population_list ]
    std_dev = calc_std(all_fitness_scores)
    for member in population_list:
        if member['_fit_score'] > (std_dev * SD_MULTIPLIER):
            new_trunc_pop.append(member)
    return new_trunc_pop


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


def mating_season (pop_to_mate):
    _new_population = []
    for i in range(len(pop_to_mate)-1):
        _new_population.append(pop_to_mate[i]) # Add the first parent to our new population too
        _new_population.append(pop_to_mate[i+1])    # Add the second parent to our new population too
        better_baby = get_mate_child(pop_to_mate[i]['_name'], pop_to_mate[i + 1]['_name'], 2)
        _new_population.append(
            { '_name': better_baby,
            '_fit_score': calc_fitness_score(better_baby) # Get new fitness score for this
              }
        )

    newlist = sorted(_new_population, key=lambda k: k['_fit_score'])
    return newlist[:5000]

def get_child(mating_pool):
    """
    Randomly select two members from mating_pool, mate and add mutation and send back child to population
    :param mating_pool: is the list of members repeated number of times of their probability to be picked up
    :return: child member to be appended to population
    """
    parent_member_A = random.choice(mating_pool)
    parent_member_B = random.choice(mating_pool)

    return get_mate_child(parent_member_A['_name'], parent_member_B['_name'], 2)

def get_children(mating_pool):
    _children = []
    for i in range(POP_SIZE):
        parent_member_A = random.choice(mating_pool)
        parent_member_B = random.choice(mating_pool)
        child_name = get_mate_child(parent_member_A['_name'], parent_member_B['_name'], 1)
        child_fitness = calc_fitness_score(child_name)
        _children.append({'_name': child_name, '_fit_score': child_fitness})

    return _children




def get_unique(mating_pool):
    """
    Reverse process to get unique members back into the population that were selected for mating.
    :param mating_poop: Mating pool list with repeating members as per their probability
    :return: population list with uniques
    """
    return [dict(y) for y in set(tuple(x.items()) for x in mating_pool)]

if __name__ == '__main__':

    # Generate First Random population
    population = generate_population(population)

    # Get first fitness score
    pop_with_score = []
    for member in population:
        pop_with_score.append(
            {'_name': member['_name'],
             '_fit_score': calc_fitness_score(member['_name'])
             }
        )

    population = pop_with_score

    # for member in population:
    #     print member['_name'] + "  " + str(member['_fit_score'])
    #
    # exit("Check Fitness")

    generation = 0

    try:
        # while TARGET not in population:
        while generation < 10:
            population = natural_selection(population)

            # Show Health (Mean) and Generation number
            all_fitness_scores = [ m['_fit_score'] for m in population ]
            print calc_mean(all_fitness_scores)

            generation += 1
            print ("Running generation number: %d" % generation)
            print ("Population size is: %d" % len(population) )
    except KeyboardInterrupt:
        for member in population:
            print member['_name'] + "  " + str(member['_fit_score'])

    for member in population:
        print member['_name'] + "  " + str(member['_fit_score'])




"""
# Adding one child at a time ...

        while TARGET not in population:

            mating_pool_ranking = natural_selection(population)

            population = get_unique(mating_pool_ranking)

            # NEXT: Randomly mate two from mating_pool and add to population
            new_member = get_child(mating_pool_ranking)
            population.append({
                '_name': new_member,
                '_fit_score':calc_fitness_score(new_member)
            })

            # Show Health (Mean) and Generation number
            all_fitness_scores = [m['_fit_score'] for m in population]
            print calc_mean(all_fitness_scores)

            generation += 1
            print ("Running generation number: %d" % generation)
            print ("Population size is: %d" % len(population) )
    except KeyboardInterrupt:
        for member in population:
            print member['_name'] + "  " + str(member['_fit_score'])

"""