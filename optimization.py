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


def plot_colours(col, perm):
    assert len(col) == len(perm)

    ratio = 20  # ratio of line height/width, e.g. colour lines will have height 10 and width 1
    img = np.zeros((ratio, len(col), 3))
    for i in range(0, len(col)):
        img[:, i, :] = colours[perm[i]]

    fig, axes = plt.subplots(1, figsize=(8, 4))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.show()


'''
    Euclidean distance function
'''


def calc_distance(col1, col2):  # calculates difference between two colours
    distance = 0
    for x in range(len(col1)):
        distance += (float(col2[x]) - float(col1[x])) ** 2
    return maths.sqrt(distance)


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


def getRGB(colour):  # gets actual RGB value of colour, only used for debugging
    rgbcol = [round(float(i) * 255) for i in colour]
    return rgbcol


def getPerm(colOrder, testCols):  # gets the permutation of the ordered colour list from greedy function
    perm = []
    for i in range(len(colOrder)):
        for j in range(len(testCols)):
            if(colOrder[i] == testCols[j]):
                perm.append(j)
    return perm


'''
    Hill climbing logic
'''


def hill_climber_body(starting_sol, testCols, num_sims):
    perm = random.sample(range(len(testCols)), len(testCols))
    solution_distances = []
    for i in range(num_sims):
        perm_neighbor = get_random_neighbor(perm)
        if evaluate(perm, testCols) > evaluate(perm_neighbor, testCols):
            perm = copy.deepcopy(perm_neighbor)
        solution_distances.append(perm)
    return perm, solution_distances


'''
    Hill climber method
'''


def hill_climber(testCols, num_sims):
    # Init time
    time_start = time.time()

    starting_perm = random.sample(range(len(testCols)), len(testCols))
    perm, solution_list = hill_climber_body(starting_perm, testCols, num_sims)
    solution_gradient = []
    for sol in solution_list:
        solution_gradient.append(evaluate(sol, test_colours))
    time_end = time.time() - time_start
    return perm, solution_gradient, time_end


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
    Evaluate function, compares euclidean distances between colors to evaluate results from algorithm
'''


def evaluate(perm, testCols):  # adds up the difference between all of the colours to measure
    total_distance = 0          # how good the permutation is
    for i in range(len(perm) - 1):
        x = perm[i]
        y = perm[i + 1]
        total_distance += calc_distance(testCols[perm[i]], testCols[perm[i + 1]])
    return total_distance


'''
    Multi Start Hill Climbing
'''


def multi_hill_climb(testCols, tries, num_sims):        # calls hill climb a bunch of times with different starting colours
    # Init time
    time_start = time.time()

    best_perms = []                 # tracks the massive amounts of data that is produced
    perm_gradients = []             # even small amounts of num_simsations on this one take a while
    for i in range(tries):              # fucking worst one haha
        best_perm, perm_gradient = hill_climber(testCols, num_sims)[0:2]
        best_perms.append(best_perm)
        perm_gradients.append(perm_gradient)
    time_end = time.time() - time_start

    return best_perms, perm_gradients, time_end


'''
    Simulated Annealing
'''


def simulated_annealing(num_sims, testCols):
    # Init time
    time_start = time.time()

    # Initialize random solution and calculate cost
    perm = random.sample(range(len(testCols)), len(testCols))
    starting_sol = evaluate(perm, testCols)
    solution_distances = []
    solution_distances.append(starting_sol)

    # Set start temperature, finishing temperature and alpha values
    temp = 1.0
    temp_fin = 0.00001
    alpha = 0.9

    while temp > temp_fin:
        for i in range(num_sims):
            new_perm = get_random_neighbor(perm)
            new_sol = evaluate(new_perm, testCols)

            if new_sol < starting_sol:
                perm = new_perm
                starting_sol = new_sol
            else:
                p_acceptance = np.exp(-(new_sol - starting_sol) / temp)
                if p_acceptance > random.random():
                    perm = new_perm
                    starting_sol = new_sol
            solution_distances.append(starting_sol)
        temp = temp * alpha
    time_end = time.time() - time_start
    return perm, solution_distances, time_end


'''
    Multi Start Simulated Annealing
'''


def multi_simulated_annealing(testCols, tries, num_sims):
    # Init time
    time_start = time.time()

    best_distances = []
    best_perm = simulated_annealing(num_sims, testCols)[0]
    best_objective_value = evaluate(best_perm, testCols)
    best_distances.append(best_objective_value)

    for i in range(tries - 1):
        perm = simulated_annealing(num_sims, testCols)[0]
        distance = evaluate(perm, testCols)
        if distance < best_objective_value:
            best_perm, best_objective_value = perm, distance
        best_distances.append(best_objective_value)

    time_end = time.time() - time_start

    return best_perms, best_distances, best_objective_value, np.mean(best_objective_value), np.median(best_objective_value), np.std(best_objective_value), time_end


#####_______main_____######

# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 100  # Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing

permutation = random.sample(range(test_size), test_size)
print(str(evaluate(permutation, test_colours)) + " random perm")
# produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

# Lewis, Sandy, Daniels and I checking if our evaluate and euclid distance is fucked or not (Spoiler: its not fucked)
# perm = [61, 23, 91, 33, 76, 58, 60, 82, 48, 34, 70, 2, 71, 49, 22, 42, 74, 75, 31, 85,
#         68, 44, 98, 20, 27, 10, 95, 19, 16, 25, 57, 69, 80, 77, 59, 93, 32, 8, 41, 7, 97, 73, 65, 40, 67, 83, 30, 47, 99, 90, 94, 86, 55, 4, 38, 64, 66, 15, 78, 35, 51,
#         39, 13, 24, 14, 50, 63, 56, 43, 92, 26, 12, 21, 6, 5, 72, 17, 81, 53, 0, 87, 45, 84, 36, 29, 62, 89, 11, 18, 96, 54, 37, 88, 1, 79, 3, 46, 28, 9, 52]
# print(str(evaluate(perm, test_colours)) + " Lewis test")

# uncomment this block for greedy
# greedycols = greedy(test_colours)
# greedyPerm = getPerm(greedycols, test_colours)
# greedyPerm = list(map(int, greedyPerm))
# print(str(evaluate(greedyPerm, test_colours)) + " greedy perm")
# plot_colours(test_colours, greedyPerm)

# uncomment this block for single start hill climber
# hill_perm, sol_gradient, hill_run_time = hill_climber(test_colours, 1000)
# print("Hill climbing: " + str(evaluate(hill_perm, test_colours)))
# print("Completed in %.5f seconds" % hill_run_time)
# plot_colours(test_colours, hill_perm)
# plt.figure()
# plt.plot(sol_gradient)
# plt.show()

# uncomment this block for multi hill start, still need to do some work to sort through the fucking
# mountain of data it returns
# multi_hill_perms, perm_gradients, multi_hill_run_time = multi_hill_climb(test_colours, 30, 2000)
# lowest_score = 100
# lowest_perm = []
# for x in multi_hill_perms:
#     perm_score = evaluate(x, test_colours)
#     print(perm_score)
#     if perm_score < lowest_score:
#         lowest_score = perm_score
#         lowest_perm = x

# print("Multi-start hill climbing: " + str(evaluate(lowest_perm, test_colours)))
# print("Completed in %.5f seconds" % multi_hill_run_time)

# Uncomment this block for simulated annealing
# sa_perm, sa_gradient, sa_run_time = simulated_annealing(300, test_colours)
# print("Simulated Annealing: " + str(evaluate(sa_perm, test_colours)))
# print("Completed in %.5f seconds" % sa_run_time)
# plot_colours(test_colours, sa_perm)
# plt.figure()
# plt.plot(sa_gradient)
# plt.show()

# multi sa
multi_sa_perms, multi_sa_gradients, multi_sa_obj_val, multi_sa_mean, multi_sa_median, multi_sa_std, multi_sa_run_time = multi_simulated_annealing(test_colours, 30, 600)
print("Simulated Annealing (Multi-Start): {}".format(multi_sa_obj_val))
print("Mean: {}".format(multi_sa_mean))
print("Median: {}".format(multi_sa_median))
print("Standard Deviation: {}".format(multi_sa_std))
print("Completed in %.5f seconds" % multi_sa_run_time)
plot_colours(test_colours, multi_sa_obj_val)
plt.figure()
plt.plot(multi_sa_gradients)
plt.show()


# plot_colours(test_colours, permutation) # this is the random permutation plot
