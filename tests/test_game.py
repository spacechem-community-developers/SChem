#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
from pympler import asizeof
from timeit import timeit
import unittest

import spacechem
from spacechem.tests import test_data

LAST_TEST_TIMES = {}
LAST_MEM_USAGE = {}


def iter_test_data(data_dict):
    for level_code in data_dict:
        for solution_code in data_dict[level_code]:
            yield level_code, solution_code


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

                def get_percent_diff_str(last_metric, cur_metric):
                    '''Provide a % indicator, colored red or green if the given metric significantly changed.'''
                    percent_diff = (cur_metric - last_metric) / last_metric
                    percent_formatting = ''
                    if percent_diff <= -0.1:
                        percent_formatting = '\033[92m'
                    elif percent_diff >= 0.1:
                        percent_formatting = '\033[91m'

                    if percent_diff >= 0:
                        percent_formatting += '+'

                    format_end = '\033[0m'

                    return f'{percent_formatting}{percent_diff:.0%}{format_end}'

                # TODO: This is only measuring the final object, we don't know if it was bigger at runtime
                mem_usage = asizeof.asizeof(reactor)
                if test_id in LAST_MEM_USAGE:
                    last_mem_usage = LAST_MEM_USAGE[test_id]

                # Check the time performance of the solver
                avg_time = timeit(('l=spacechem.level.ResearchLevel(level_code)'
                                   ';s=spacechem.solution.Solution(l, solution_code)'
                                   ';spacechem.game.score_solution(s)'),
                                  number=100, globals={'level_code': level_code,
                                                       'solution_code': solution_code,
                                                       'spacechem': spacechem}) / 100

                if test_id in LAST_TEST_TIMES:
                    print(f'Avg {avg_time:.5f}s ({get_percent_diff_str(LAST_TEST_TIMES[test_id], avg_time)}) to run {test_id}')
                else:
                    print(f'Avg {avg_time:.5f}s (NEW) to run {test_id}')

                if test_id in LAST_MEM_USAGE:
                    print(f'Mem usage: {mem_usage} ({get_percent_diff_str(LAST_MEM_USAGE[test_id], mem_usage)})')
                else:
                    print(f'Mem usage: {mem_usage} (NEW)')

                LAST_TEST_TIMES[test_id] = avg_time
                LAST_MEM_USAGE[test_id] = mem_usage

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
    test_times_file = Path(__file__).parent / 'test_times.pickle'
    if test_times_file.exists():
        with test_times_file.open('rb') as f:
            LAST_TEST_TIMES = pickle.load(f)

    test_mem_usage_file = Path(__file__).parent / 'test_mem_usage.pickle'
    if test_mem_usage_file.exists():
        with test_mem_usage_file.open('rb') as f:
            LAST_MEM_USAGE = pickle.load(f)

    unittest.main(verbosity=0, exit=False)

    # Write the current times/mem usage to file
    with test_times_file.open('wb') as f:
        pickle.dump(LAST_TEST_TIMES, f)
    with test_mem_usage_file.open('wb') as f:
        pickle.dump(LAST_MEM_USAGE, f)
