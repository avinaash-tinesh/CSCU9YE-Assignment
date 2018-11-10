import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy
import time
import math as maths
from AITest import *

# Display the colours in the order of the permutation in a pyplot window
# Input, list of colours, and ordering  of colours.
# They need to be of the same length


def plot_colours(col, perm, plt_name):
    assert len(col) == len(perm)

    ratio = 20  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
    img = np.zeros((ratio, len(col), 3))
    for i in range(0, len(col)):
        img[:, i, :] = colours[perm[i]]

    fig, axes = plt.subplots(1, figsize=(8, 4))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.savefig("Graphs/{}.png".format(plt_name), bbox_inches="tight")
    plt.title("Visualization for {}".format(plt_name))
    plt.show(block=False)
    plt.pause(5)
    plt.close()


'''
    Euclidean distance function
'''


def calc_distance(col1, col2):  # calculates difference between two colours
    distance = 0
    for x in range(len(col1)):
        distance += (float(col2[x]) - float(col1[x])) ** 2
    return maths.sqrt(distance)


'''
    Get Permutation of ordered color list for greedy heuristic
'''


def getPerm(colOrder, testCols):
    perm = []
    for i in range(len(colOrder)):
        for j in range(len(testCols)):
            if(colOrder[i] == testCols[j]):
                perm.append(j)
    return perm


'''
    Evaluate function, compares euclidean distances between colors to evaluate results from algorithm
'''


def evaluate(perm, testCols):
    total_distance = 0
    for i in range(len(perm) - 1):
        x = perm[i]
        y = perm[i + 1]
        total_distance += calc_distance(testCols[perm[i]], testCols[perm[i + 1]])
    return total_distance


'''
    Get random neighbor function
'''


def get_random_neighbor(perm):
    random_limits = [0, 0]
    random_limits[0] = random.randint(0, len(perm))
    random_limits[1] = random.randint(0, len(perm))
    random_limits.sort()
    reverse_portion = perm[random_limits[0]:random_limits[1]]
    perm = [x for x in perm if x not in reverse_portion]
    reverse_portion = list(reversed(reverse_portion))
    for i in range(len(reverse_portion)):
        perm.insert(random_limits[0] + i, reverse_portion[i])
    return perm


'''
    Greedy Heuristic
'''


def greedy(testCols):  # nearest neighbor algorithm
    colours = testCols.copy()  # init list of unordered colours
    orderedColours = []         # init list of ordered colours
    colour = colours[random.randint(0, len(colours))]  # random starting colour
    # colour = colours[s0]
    orderedColours.append(colour)       # init first colour
    colours.remove(colour)
    for x in range(len(testCols) - 1):  # loop takes colour and finds nearest neighbor from unordered list
        closestDistance = 100           # nearest neighbor is added to ordered colour list and taken away from
        closestColour = [0, 0, 0]       # unordered list so it can't be picked again
        for colournum_sims in colours:      # I ordered the colours instead of making a permutation so its converted later
            distance = calc_distance(colour, colournum_sims)
            if distance < closestDistance and distance > 0:
                closestColour = copy.deepcopy(colournum_sims)
                closestDistance = copy.deepcopy(distance)
        orderedColours.append(closestColour)

        colours.remove(closestColour)
        colour = copy.deepcopy(closestColour)

    return orderedColours


'''
    Hill climbing 
'''


def hill_climbing(num_sims, testCols):
    # Init time
    time_start = time.time()
    # Init permutation
    starting_perm = random.sample(range(len(testCols)), len(testCols))
    solution_distances = []
    best_objective_value = evaluate(starting_perm, testCols)
    solution_distances.append(best_objective_value)

    for i in range(num_sims):
        perm_neighbor = get_random_neighbor(starting_perm)
        neighbor_dist = evaluate(perm_neighbor, testCols)

        if neighbor_dist < best_objective_value:
            starting_perm, best_objective_value = perm_neighbor, neighbor_dist

        solution_distances.append(best_objective_value)

    time_end = time.time() - time_start

    return starting_perm, best_objective_value, solution_distances, time_end


'''
    Multi Start Hill Climbing
'''


def multi_hill_climb(testCols, tries, num_sims):
    # Init time
    time_start = time.time()

    best_perm, best_objective_value = hill_climbing(num_sims, testCols)[0:2]
    best_distances = []
    best_distances.append(best_objective_value)

    for i in range(tries - 1):
        perm, distance = hill_climbing(num_sims, testCols)[0:2]

        if distance < best_objective_value:
            best_perm, best_objective_value = perm, distance

        best_distances.append(best_objective_value)

    time_end = time.time() - time_start

    return best_perm, best_distances, best_objective_value, np.mean(best_distances), np.median(best_distances), np.std(best_distances), time_end


'''
    Simulated Annealing
'''


def simulated_annealing(num_sims, testCols):
    # Init time
    time_start = time.time()

    # Initialize random solution and calculate cost
    perm = random.sample(range(len(testCols)), len(testCols))
    best_objective_value = evaluate(perm, testCols)
    solution_distances = []
    solution_distances.append(best_objective_value)

    # Set start temperature, finishing temperature and alpha values
    temp = 1.0
    temp_fin = 0.00001
    alpha = 0.9

    while temp > temp_fin:

        for i in range(num_sims):
            new_perm = get_random_neighbor(perm)
            new_sol = evaluate(new_perm, testCols)

            if new_sol < best_objective_value:
                perm, best_objective_value = new_perm, new_sol
            else:
                p_acceptance = np.exp(-(new_sol - best_objective_value) / temp)
                if p_acceptance > random.random():
                    perm, best_objective_value = new_perm, new_sol

            solution_distances.append(best_objective_value)

        temp = temp * alpha

    time_end = time.time() - time_start
    return perm, best_objective_value, solution_distances, time_end


'''
    Multi Start Simulated Annealing
'''


def multi_simulated_annealing(testCols, tries, num_sims):
    # Init time
    time_start = time.time()

    best_distances = []
    best_perm, best_objective_value = simulated_annealing(num_sims, testCols)[0:2]
    best_distances.append(best_objective_value)

    for i in range(tries - 1):
        perm, distance = simulated_annealing(num_sims, testCols)[0:2]

        if distance < best_objective_value:
            best_perm, best_objective_value = perm, distance

        best_distances.append(best_objective_value)

    time_end = time.time() - time_start

    return best_perm, best_distances, best_objective_value, np.mean(best_distances), np.median(best_distances), np.std(best_distances), time_end


# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 500  # Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing
