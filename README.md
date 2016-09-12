# genetic_algo
Genetic Algorithm Playground

First ever GA written after a 20 minute crash course by Daniel Shiffman. Spoiler alert, the following code does't work:

```python
import urllib2
import random
import string
from math import sqrt

TARGET = 'varun'    # Target word or let's call it sequence of characters. For now only ASCII lowercase, no space or uppercase :)
POP_SIZE = 1000     # Population size
MUTATION = 0.01     # Mutation proportion
population = []     # Our world as an array of member dictionaries, with 'member' and 'fit_score'. 'member' is a char seq (a word)
SD_MULTIPLIER = 2
TARGET_BUILD = string.ascii_lowercase

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
    targ_list = list(TARGET)
    fit_score = 0

    for c in char_list:
        if c in targ_list: # if this char is in our target member
            if char_list.index(c) == targ_list.index(c):
                fit_score += 1
    return fit_score / float(len(TARGET))


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
def get_mate_child (mem1, mem2, type=1):
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
        parent_1.pop()
        parent_1.append(random.choice(TARGET_BUILD))
        parent_2.pop()
        parent_2.append(random.choice(TARGET_BUILD))
        new_member_chars = parent_1 + parent_2
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
    return _new_population[:5000]



if __name__ == '__main__':

    # Generate First Random population
    population = generate_population(population)
    print len(population)

    # Get first fitness score
    pop_with_score = []
    for member in population:
        pop_with_score.append(
            {'_name': member['_name'],
             '_fit_score': calc_fitness_score(member['_name'])
             }
        )

    population = pop_with_score

    generation = 0

    try:
        while TARGET not in population:
            pop_to_breed = get_best_population(population) # Use truncation method to get members with fitness score > 1 std deviation
            population = mating_season(pop_to_breed)

            # Show Health (Mean) and Generation number
            all_fitness_scores = [m['_fit_score'] for m in population]
            print calc_mean(all_fitness_scores)

            generation += 1
            print ("Running generation number: %d" % generation)
            print ("Population size is: %d" % len(population) )
    except KeyboardInterrupt:
        for member in population:
            print member['_name'] + "  " + str(member['_fit_score'])
  ```
  
  
