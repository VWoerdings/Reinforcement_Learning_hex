import matplotlib.pyplot as plt
import numpy as np
import trueskill as ts
import time

import TerminatorHex
from HexBoard import HexBoard
from Experiment import play_1v1

"""This script calculates the rating of three Hex algorithms by playing them against each other and visualizes the 
evolution of their ratings. Each round, each algorithm plays two games against the other two.
This script rates the following AIs:
    AI with random evaluation
    AI with search depth 3 our improved heuristic
    AI with search depth 4 our improved heuristic
"""

if __name__ == '__main__':
    # Initialize AI
    terminator_depth_3 = TerminatorHex.TerminatorHex(3, use_suggested_heuristic=True,
                                                     heuristic_evaluator=None, depth_weighting=0, random_seed='random',
                                                     do_iterative_deepening=False, max_time=None, do_transposition=False
                                                     )
    terminator_depth_4 = TerminatorHex.TerminatorHex(4, use_suggested_heuristic=True,
                                                     heuristic_evaluator=None, depth_weighting=0, random_seed='random',
                                                     do_iterative_deepening=False, max_time=None, do_transposition=False
                                                     )

    # Assign move generators
    random_player_move = terminator_depth_3.random_move
    dijkstra3_move = terminator_depth_3.terminator_move
    dijkstra4_move = terminator_depth_4.terminator_move

    # Initialize ratings
    random_player_rating = ts.Rating()
    dijkstra3_rating = ts.Rating()
    dijkstra4_rating = ts.Rating()

    random_player_desc = "Random AI"
    dijkstra3_desc = "Search depth 3"
    dijkstra4_desc = "Search depth 4"

    # Initialize lists to keep track of rating history
    random_player_mu = [random_player_rating.mu]
    dijkstra3_mu = [dijkstra3_rating.mu]
    dijkstra4_mu = [dijkstra4_rating.mu]
    random_player_sigma = [random_player_rating.sigma]
    dijkstra3_sigma = [dijkstra3_rating.sigma]
    dijkstra4_sigma = [dijkstra4_rating.sigma]

    max_rounds = 50
    minimum_sigma = 1.0  # Requirement for convergence
    round_number = 0
    max_sigma = max(random_player_rating.sigma, dijkstra3_rating.sigma, dijkstra4_rating.sigma)
    start = time.time()
    while round_number < max_rounds and max_sigma >= minimum_sigma:
        print("Currently playing round number %d of %d" % (round_number + 1, max_rounds))
        print("Highest sigma is %.3f" % max_sigma)

        # Random vs dijkstra3
        print("Playing", random_player_desc, "vs", dijkstra3_desc)
        random_player_rating, dijkstra3_rating = play_1v1(random_player_move, random_player_rating,
                                                          dijkstra3_move, dijkstra3_rating, round_number)

        # Dijkstra3 vs dijkstra4
        print("Playing", dijkstra3_desc, "vs", dijkstra4_desc)
        dijkstra3_rating, dijkstra4_rating = play_1v1(dijkstra3_move, dijkstra3_rating,
                                                      dijkstra4_move, dijkstra4_rating, round_number)

        # Random vs dijkstra4
        print("Playing", random_player_desc, "vs", dijkstra4_desc)
        random_player_rating, dijkstra4_rating = play_1v1(random_player_move, random_player_rating,
                                                          dijkstra4_move, dijkstra4_rating, round_number)

        random_player_mu.append(random_player_rating.mu)
        random_player_sigma.append(random_player_rating.sigma)
        dijkstra3_mu.append(dijkstra3_rating.mu)
        dijkstra3_sigma.append(dijkstra3_rating.sigma)
        dijkstra4_mu.append(dijkstra4_rating.mu)
        dijkstra4_sigma.append(dijkstra4_rating.sigma)

        round_number += 1
        max_sigma = max(random_player_rating.sigma, dijkstra3_rating.sigma, dijkstra4_rating.sigma)
    end = time.time()

    print("Final ratings are:")
    print(random_player_desc, ": ", random_player_rating, sep="")
    print(dijkstra3_desc, ": ", dijkstra3_rating, sep="")
    print(dijkstra4_desc, ": ", dijkstra4_rating, sep="")
    print("Total time elapsed is %.3f" % (end - start))

    # Plot rating evolution
    random_half_sigma = np.array(random_player_sigma) / 2
    dijkstra3_half_sigma = np.array(dijkstra3_sigma) / 2
    dijkstra4_half_sigma = np.array(dijkstra4_sigma) / 2

    plt.figure()
    plt.errorbar(range(max_rounds + 1), random_player_mu, yerr=random_half_sigma, label=random_player_desc, fmt='o')
    plt.errorbar(range(max_rounds + 1), dijkstra3_mu, yerr=dijkstra3_half_sigma, label=dijkstra3_desc, fmt='o')
    plt.errorbar(range(max_rounds + 1), dijkstra4_mu, yerr=dijkstra4_half_sigma, label=dijkstra4_desc, fmt='o')
    plt.title("Skill rating vs number of rounds played (2 games per round)")
    plt.xlabel("Round number")
    plt.ylabel("Rating")
    plt.ylim((0, 50))
    plt.legend()
    plt.show()
