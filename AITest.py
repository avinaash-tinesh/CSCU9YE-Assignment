import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math as maths
import copy


# Reads the file  of colours
# Returns the number of colours in the file and a list with the colours (RGB) values

def read_file(fname):
	with open(fname, 'r') as afile:
		lines = afile.readlines()
	n = int(lines[3])  # number of colours  in the file
	col = []
	lines = lines[4:]  # colors as rgb values
	for l in lines:
		rgb = l.split()
		col.append(rgb)
	return n, col


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


def calcDistance(col1, col2):
	distance = 0
	for x in range(len(col1)):
		distance += (float(col2[x]) - float(col1[x])) ** 2
	return maths.sqrt(distance)


def greedy(testCols):
	colours = testCols.copy()
	orderedColours = []
	colour = colours[random.randint(0, len(colours))]
	# colour = colours[s0]
	orderedColours.append(colour)
	colours.remove(colour)
	for x in range(len(testCols) - 1):
		closestDistance = 100
		closestColour = [0, 0, 0]
		for colourIter in colours:
			distance = calcDistance(colour, colourIter)
			if distance < closestDistance and distance > 0:
				closestColour = copy.deepcopy(colourIter)
				closestDistance = copy.deepcopy(distance)
		orderedColours.append(closestColour)


		colours.remove(closestColour)
		colour = copy.deepcopy(closestColour)

	return orderedColours


def getRGB(colour):
	rgbcol = [round(float(i) * 255) for i in colour]
	return rgbcol

def getPerm(colOrder, testCols):
	perm = []
	for i in range(len(colOrder)):
		for j in range(len(testCols)):
			if(colOrder[i] == testCols[j]):
				perm.append(j)

	return perm

def hill_climber_body(starting_sol, testCols):
	perm = random.sample(range(len(testCols)), len(testCols))
	solution_gradient = []
	for i in range(100):
		perm_neighbor = get_random_neighbor(perm)
		if evaluate(perm, testCols) > evaluate(perm_neighbor, testCols):
			perm = copy.deepcopy(perm_neighbor)
		solution_gradient.append(perm)
	return perm, solution_gradient

def hill_climber(testCols):
	starting_perm = random.sample(range(len(testCols)), len(testCols))
	perm, solution_list = hill_climber_body(starting_perm, testCols)
	solution_gradient = []
	for sol in solution_list:
		solution_gradient.append(evaluate(sol, test_colours))
	return perm, solution_gradient


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


def evaluate(perm, testCols):
	total_distance = 0
	for i in range(len(perm)-1):
		x = perm[i]
		y= perm[i+1]
		total_distance += calcDistance(testCols[perm[i]], testCols[perm[i+1]])
	return total_distance

def multi_hill_climb(testCols):
	best_perms = []
	perm_gradients = []
	for i in range(30):
		best_perm, perm_gradient = hill_climber(testCols)
		best_perms.append(best_perm)
		perm_gradients.append(perm_gradient)

	return best_perms, perm_gradients






#####_______main_____######

# Get the directory where the file is located
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)  # Change the working directory so we can read the file

ncolors, colours = read_file('colours.txt')  # Total number of colours and list of colours

test_size = 100# Size of the subset of colours for testing
test_colours = colours[0:test_size]  # list of colours for testing

permutation = random.sample(range(test_size), test_size)
print(str(evaluate(permutation, test_colours)) + " random perm")
# produces random pemutation of lenght test_size, from the numbers 0 to test_size -1

# greedycols = greedy(test_colours)
# greedyPerm = getPerm(greedycols, test_colours)
# greedyPerm = list(map(int, greedyPerm))
# print(str(evaluate(greedyPerm, test_colours)) + " greedy perm")


# hill_perm, sol_gradient = hill_climber(test_colours)
# print(str(evaluate(hill_perm, test_colours)) + " hill climber perm")
# plt.figure()
# plt.plot(sol_gradient)
# plt.show()

multi_hill_perms, perm_gradients = multi_hill_climb(test_colours)
lowest_score = 100
lowest_perm = []
for x in multi_hill_perms:
	perm_score = evaluate(x, test_colours)
	print(perm_score)
	if  perm_score < lowest_score:
		lowest_score = perm_score
		lowest_perm = x

print(str(evaluate(lowest_perm, test_colours)) + " multi")
# plot_colours(test_colours, permutation)
# plot_colours(test_colours, greedyPerm)
# plot_colours(test_colours, hill_perm)



