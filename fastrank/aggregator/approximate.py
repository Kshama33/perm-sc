__all__ = ['BordaRankAggregator', 'LocalSearchRefiner', 'KemenyLocalSearchRefiner', 'KemenyOptimalAggregator']

import logging
from itertools import combinations, permutations
from typing import Callable

import numba
import numpy as np
from scipy.optimize import linprog

from .base import RankAggregator, AggregateRefiner
from ..utils import ranks_from_preferences, sum_kendall_tau, sum_spearmanr, sample_random_preferences, \
    cdk_graph_from_preferences, cdk_graph_distance, cdk_graph_vertex_swap, preferences_from_cdk_graph, \
    preferences_from_ranks


class KemenyOptimalAggregator(RankAggregator):
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        pass


class BordaRankAggregator(RankAggregator):
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        def _compute_borda(preferences: np.ndarray) -> np.ndarray:
            positions = ranks_from_preferences(preferences)
            return positions.mean(axis=0)

        borda_counts = _compute_borda(preferences)
        return np.argsort(borda_counts)


@numba.njit
def _ls_refine(preferences, candidate, fn, method='min', max_iter=-1):
    # type: (np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], float], str, int) -> np.ndarray
    improved = True
    other = candidate.copy()
    best_dist = fn(preferences, candidate)

    is_max = method == 'max'

    if is_max:
        best_dist = -best_dist

    rand_indices = np.arange(len(candidate))

    while improved and max_iter != 0:
        improved = False
        np.random.shuffle(rand_indices)
        max_iter -= 1

        for i in rand_indices:
            for j in np.random.permutation(len(candidate) - i - 1):
                j += i + 1
                other[i], other[j] = other[j], other[i]
                curr_dist = fn(preferences, other)

                if is_max:
                    curr_dist = -curr_dist

                if curr_dist < best_dist:
                    improved = True
                    best_dist = curr_dist
                    candidate, other = other, other.copy()
                else:
                    other[i], other[j] = other[j], other[i]

    return candidate


class LocalSearchRefiner(AggregateRefiner):
    def __init__(self, objective: str = 'kendalltau', max_iter: int = -1):
        self.objective = objective
        self.max_iter = max_iter

    def refine(self, preferences: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        match self.objective:
            case 'kendalltau':
                logging.warning('The `KemenyLocalSearchRefiner` refiner is generally much faster.')
                X_ranks = ranks_from_preferences(preferences)
                y_rank = ranks_from_preferences(candidate)
                y_rank = _ls_refine(X_ranks, y_rank, sum_kendall_tau, max_iter=self.max_iter)

                return preferences_from_ranks(y_rank)
            case 'spearmanr':
                return _ls_refine(preferences, candidate, sum_spearmanr, 'max', max_iter=self.max_iter)
            case _:
                raise ValueError(f'Objective {self.objective} is not supported.')


@numba.njit
def _kemeny_ls_refine(X_graph, y_graph, max_iter: int = -1) -> np.ndarray:
    improved = True
    best_dist = cdk_graph_distance(X_graph, y_graph)
    n = y_graph.shape[0]
    rand_indices = np.arange(n)

    while improved and max_iter != 0:
        improved = False
        np.random.shuffle(rand_indices)
        max_iter -= 1

        for i in rand_indices:
            for j in np.random.permutation(n - i - 1):
                j += i + 1
                cdk_graph_vertex_swap(y_graph, i, j)
                curr_dist = cdk_graph_distance(X_graph, y_graph)

                if curr_dist < best_dist:
                    improved = True
                    best_dist = curr_dist
                else:
                    cdk_graph_vertex_swap(y_graph, i, j)

    return y_graph


class KemenyLocalSearchRefiner(AggregateRefiner):
    def __init__(self, max_iter: int = -1):
        self.max_iter = max_iter

    def refine(self, preferences: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        X_graph = cdk_graph_from_preferences(preferences)
        y_graph = cdk_graph_from_preferences(candidate)
        return preferences_from_cdk_graph(_kemeny_ls_refine(X_graph, y_graph, max_iter=self.max_iter))


if __name__ == '__main__':
    real_prefs = np.array([[1, 2, 0], [1, 2, 0], [1, 0, 2], [0, 1, 2], [1, 2, 0]])
    real_proposal = BordaRankAggregator().aggregate(real_prefs)
    refined_proposal = LocalSearchRefiner().refine(real_prefs, real_proposal)
    print(real_proposal, refined_proposal)
    print(sum_kendall_tau(real_prefs, real_proposal), sum_kendall_tau(real_prefs, refined_proposal))
    real_proposal = BordaRankAggregator().aggregate(real_prefs)
    # print(real_proposal, LocalSearchRefiner('spearmanr').refine(real_prefs, real_proposal))

    import time
    a = time.time()
    prefs = sample_random_preferences(10, 50)  # warmup
    proposal = BordaRankAggregator().aggregate(prefs)
    print(time.time() - a)

    print(np.sum(LocalSearchRefiner().refine(prefs, proposal)))
    print(np.sum(KemenyLocalSearchRefiner().refine(prefs, proposal)))

    prefs = sample_random_preferences(2, 100)
    proposal = BordaRankAggregator().aggregate(prefs)
    print(sum_kendall_tau(prefs, proposal))

    a = time.time()
    print(sum_kendall_tau(prefs, LocalSearchRefiner().refine(prefs, proposal)))
    print(time.time() - a)

    a = time.time()
    print(sum_kendall_tau(prefs, KemenyLocalSearchRefiner().refine(prefs, proposal)))
    print(time.time() - a)
