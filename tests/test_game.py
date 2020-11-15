#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
from pympler import asizeof
from timeit import Timer
import unittest

import spacechem
from spacechem.tests import test_data

LAST_TEST_RESULTS = {}


def iter_test_data(data_dict):
    for level_code in data_dict:
        for solution_code in data_dict[level_code]:
            yield level_code, solution_code


def get_percent_diff_str(last_metric, cur_metric):
    '''Provide a % indicator, colored red or green if the given metric significantly changed.'''
    percent_diff = (cur_metric - last_metric) / last_metric
    percent_formatting = ''
    if percent_diff <= -0.05:
        percent_formatting = '\033[92m'  # green
    elif percent_diff >= 0.05:
        percent_formatting = '\033[91m'  # red

    if percent_diff >= 0:
        percent_formatting += '+'

    format_end = '\033[0m'

    return f'{percent_formatting}{percent_diff:.0%}{format_end}'


class TestGame(unittest.TestCase):
    def test_valid_solutions(self):
        for level_code, solution_code in iter_test_data(test_data.valid):
            level = spacechem.level.ResearchLevel(level_code)
            solution = spacechem.solution.Solution(level, solution_code)
            test_id = f'{level.get_name()} {solution.name}'
            with self.subTest(msg=test_id):
                reactor = spacechem.game.Reactor(solution)

                self.assertEqual(reactor.run(),
                                 solution.expected_score)

                # TODO: This is only measuring the final object, we don't know the max runtime mem usage
                mem_usage = asizeof.asizeof(reactor)

                # Check the time performance of the solver
                timer = Timer(('l=spacechem.level.ResearchLevel(level_code)'
                               ';s=spacechem.solution.Solution(l, solution_code)'
                               ';spacechem.game.score_solution(s)'),
                              globals={'level_code': level_code,
                                       'solution_code': solution_code,
                                       'spacechem': spacechem})

                if test_id not in LAST_TEST_RESULTS:
                    # autorange() measures how many loops of the solver needed to exceed 0.2 seconds. Ensures the slower
                    # solutions don't bloat the test times. We'll divide that down to 0.04s
                    # Pretty meta, but since this number will stick for the lifetime of the test, I'm also measuring it
                    # twice and maxing it to be sure that a randomly extra-slow solve can't force us to under-sample a
                    # test forever. Running tests for the first time will take about an extra half second per test.
                    loops = max(max(timer.autorange()[0] for _ in range(2)) // 5, 1)
                else:
                    loops = LAST_TEST_RESULTS[test_id]['loops']

                # repeat = # of timeit calls, number = # of code executions per timeit call
                # Given our above loops measurement this should run in about 0.2s per test
                min_time = min(timer.repeat(repeat=loops, number=5)) / 5

                if test_id in LAST_TEST_RESULTS:
                    print(f"Min {min_time:.5f}s ({get_percent_diff_str(LAST_TEST_RESULTS[test_id]['min_time'], min_time)}) to run {test_id}")
                    print(f"Mem usage: {mem_usage} ({get_percent_diff_str(LAST_TEST_RESULTS[test_id]['mem_usage'], mem_usage)})")
                else:
                    print(f'Setting # of loops to: {loops * 5}')
                    print(f'Min {min_time:.5f}s (NEW) to run {test_id}')
                    print(f'Mem usage: {mem_usage} (NEW)')

                LAST_TEST_RESULTS[test_id] = {'loops': loops,
                                              'min_time': min_time,
                                              'mem_usage': mem_usage}

    def test_infinite_loops(self):
        for level_code, solution_code in iter_test_data(test_data.infinite_loops):
            level = spacechem.level.ResearchLevel(level_code)
            solution = spacechem.solution.Solution(level, solution_code)
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                with self.assertRaises(spacechem.exceptions.InfiniteLoopError):
                    spacechem.game.score_solution(solution)

    def test_invalid_outputs(self):
        for level_code, solution_code in iter_test_data(test_data.invalid_outputs):
            level = spacechem.level.ResearchLevel(level_code)
            solution = spacechem.solution.Solution(level, solution_code)
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                with self.assertRaises(spacechem.exceptions.InvalidOutputError):
                    spacechem.game.score_solution(solution)

    def test_collisions(self):
        for level_code, solution_code in iter_test_data(test_data.collisions):
            level = spacechem.level.ResearchLevel(level_code)
            solution = spacechem.solution.Solution(level, solution_code)
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                with self.assertRaises(spacechem.exceptions.ReactionError):
                    spacechem.game.score_solution(solution)


if __name__ == '__main__':
    # Use pickle to load the last test results and display a user-friendly summary of what tests performances went
    # up/down
    test_results_file = Path(__file__).parent / 'test_results.pickle'
    if test_results_file.exists():
        with test_results_file.open('rb') as f:
            LAST_TEST_RESULTS = pickle.load(f)

    unittest.main(verbosity=0, exit=False)

    # Write the current times/mem usage to file
    with test_results_file.open('wb') as f:
        pickle.dump(LAST_TEST_RESULTS, f)
