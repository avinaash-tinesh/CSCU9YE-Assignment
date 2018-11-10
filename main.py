from optimization import *


#####_______main_____######


''' Run Function test'''


def run_test(test_size, i):

    # Get the directory where the file is located
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)  # Change the working directory so we can read the file

    # Total number of colours and list of colours
    ncolors, colours = read_file('colours.txt')

    # List of colours for testing
    test_colours = colours[0:test_size]

    '''------------Random Permutation------------'''
    permutation = random.sample(range(test_size), test_size)
    print("---------------------Random Permutation {} Colors---------------------".format(test_size))
    print("Random Permutation {}: {}".format(test_size, str(evaluate(permutation, test_colours))))
    print("\n")
    plot_colours(test_colours, permutation, "Random Permutation - {} Colors".format(test_size))

    if i == 1:
        '''------------Greedy Constuctive Heuristic------------'''
        greedycols = greedy(test_colours)
        greedyPerm = getPerm(greedycols, test_colours)
        greedyPerm = list(map(int, greedyPerm))
        print("---------------------Greedy Constructive Heuristic {} Colors---------------------".format(test_size))
        print("Greedy Constructive Heuristic: {}".format(str(evaluate(greedyPerm, test_colours))))
        plot_colours(test_colours, greedyPerm, "Greedy Constructive Heuristic - {} Colors".format(test_size))

    elif i == 2:
        '''---------------Single Start Hill Climber---------------'''
        hill_perm, hill_obj_val, hill_distances, hill_run_time = hill_climbing(9900, test_colours)
        print("---------------------Hill Climbing {} Colors---------------------")
        print("Hill climbing: {}".format(hill_obj_val))
        print("Completed in %.5f seconds" % hill_run_time)
        plot_colours(test_colours, hill_perm, "Hill Climbing - {} Colors".format(test_size))
        plt.figure()
        plt.plot(hill_distances)
        plt.savefig("Graphs/Hill Climbing ({} Colors).png".format(test_size), bbox_inches="tight")
        plt.title("Hill Climbing, Num Colors: {}".format(test_size))
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    elif i == 3:
        '''---------------Multi Start Hill Climber---------------'''
        multi_hill_perm, multi_hill__distances, multi_hill_obj_val, multi_hill_mean, multi_hill_median, multi_hill_std, multi_hill_run_time = multi_hill_climb(test_colours, 30, 110000)
        print("---------------------Multi Start Hill Climbing {} Colors---------------------".format(test_size))
        print("Multi-start hill climbing: {}".format(multi_hill_obj_val))
        print("Mean: {}".format(multi_hill_mean))
        print("Median: {}".format(multi_hill_median))
        print("Standard Deviation: {}".format(multi_hill_std))
        print("Completed in %.5f seconds" % multi_hill_run_time)
        plot_colours(test_colours, multi_hill_perm, "Multi-Start Hill Climbing - {} Colors".format(test_size))
        plt.figure()
        plt.plot(multi_hill__distances)
        plt.savefig("Graphs/Multi-Start Hill Climbing ({} Colors).png".format(test_size), bbox_inches="tight")
        plt.title("Multi-Start Hill Climbing, Num Colors: {}".format(test_size))
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    elif i == 4:
        '''---------------Simulated Annealing---------------'''
        sa_perm, sa_obj_val, sa_distances, sa_run_time = simulated_annealing(750, test_colours)
        print("---------------------Simulated Annealing {} Colors---------------------".format(test_size))
        print("Simulated Annealing: {}".format(sa_obj_val))
        print("Completed in %.5f seconds" % sa_run_time)
        plot_colours(test_colours, sa_perm, "Simulated Annealing - {} Colors".format(test_size))
        plt.figure()
        plt.plot(sa_distances)
        plt.savefig("Graphs/Simulated Annealing ({} Colors).png".format(test_size), bbox_inches="tight")
        plt.title("Simulated Annealing, Num Colors: {}".format(test_size))
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    elif i == 5:
        '''---------------Multi Start Simulated Annealing---------------'''
        multi_sa_perm, multi_sa_distances, multi_sa_obj_val, multi_sa_mean, multi_sa_median, multi_sa_std, multi_sa_run_time = multi_simulated_annealing(test_colours, 30, 750)
        print("---------------------Multi Start Simulated Annealing {} Colors---------------------".format(test_size))
        print("Simulated Annealing (Multi-Start): {}".format(multi_sa_obj_val))
        print("Mean: {}".format(multi_sa_mean))
        print("Median: {}".format(multi_sa_median))
        print("Standard Deviation: {}".format(multi_sa_std))
        print("Completed in %.5f seconds" % multi_sa_run_time)
        plot_colours(test_colours, multi_sa_perm, "Multi-Start Simulated Annealing - {} Colors".format(test_size))
        plt.figure()
        plt.boxplot(multi_sa_distances)
        plt.savefig("Graphs/Multi-Start Simulated Annealing ({} Colors).png".format(test_size), bbox_inches="tight")
        plt.title("Multi-Start Simulated Annealing, Num Colors: {}".format(test_size))
        plt.show(block=False)
        plt.pause(5)
        plt.close()


run_test(100, 1)
