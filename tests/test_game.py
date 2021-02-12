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

num_subtests = 0


def iter_test_data(solution_codes):
    global num_subtests
    for solution_code in solution_codes:
        num_subtests += 1

        # Parse only the metadata line so we can error out from the appropriate subTest if the full parse fails
        level_name, _, _, solution_name = spacechem.solution.Solution.parse_metadata(solution_code)
        test_id = f'{level_name} - {solution_name}'
        level_code = spacechem.levels.levels[level_name] if level_name in spacechem.levels.levels else test_data.test_levels[level_name]

        yield test_id, level_code, solution_code


def percent_diff_str(last_metric, cur_metric):
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
        '''Tests for solutions that should run to completion and match the expected score.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.valid_solutions):
            with self.subTest(msg=test_id):
                level = spacechem.level.Level(level_code)
                solution = spacechem.solution.Solution(level, solution_code)
                self.assertEqual(solution.run(), solution.expected_score)

                # TODO: This is only measuring the final object, we don't know the max runtime mem usage
                mem_usage = asizeof.asizeof(solution)

                # Check the time performance of the solver
                timer = Timer(('l=spacechem.level.Level(level_code)'
                               ';s=spacechem.solution.Solution(l, solution_code)'
                               ';s.run()'),
                              globals={'level_code': level_code,
                                       'solution_code': solution_code,
                                       'spacechem': spacechem})

                if test_id not in LAST_TEST_RESULTS:
                    # autorange() measures how many loops of the solver needed to exceed 0.2 seconds. Ensures the slower
                    # solutions don't bloat the test times. We'll divide that down to 0.1s for expediency, or at least
                    # 5 runs per test, whichever is higher.
                    # Pretty meta, but since this number will stick for the lifetime of the test, I'm also measuring it
                    # twice and maxing it to be sure that a randomly extra-slow solve can't force us to under-sample a
                    # test forever. Running tests for the first time will take about an extra half second per new test.
                    loops = max(max(timer.autorange()[0] for _ in range(2)) // 2, 5)
                else:
                    loops = LAST_TEST_RESULTS[test_id]['loops']

                # repeat = # of timeit calls, number = # of code executions per timeit call
                # I think they expect people to use autorange for `number` but the volatility seems to be better when
                # it's used for `repeat`. Probably repeat is better for large times, while number may help avoid the
                # measurement error of timeit outweighing short times.
                # Given our above loops measurement this should run in about 0.1s for lightweight tests
                min_time = min(timer.repeat(repeat=loops, number=1))

                if test_id in LAST_TEST_RESULTS:
                    metrics_str = f"{min_time:.4f}s (b. of {loops}) ({percent_diff_str(LAST_TEST_RESULTS[test_id]['min_time'], min_time)})"
                    metrics_str += f" | Mem usage: {mem_usage} B ({percent_diff_str(LAST_TEST_RESULTS[test_id]['mem_usage'], mem_usage)})"
                    metrics_str += f" | {test_id}"
                else:
                    metrics_str = f"{min_time:.4f}s (b. of {loops}) (NEW) | Mem usage: {mem_usage} B (NEW) | {test_id}"
                print(metrics_str)

                LAST_TEST_RESULTS[test_id] = {'loops': loops,
                                              'min_time': min_time,
                                              'mem_usage': mem_usage}

    def test_infinite_loops(self):
        '''Tests for solutions that should exceed run()'s timeout.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.infinite_loops):
            with self.subTest(msg=test_id):
                level = spacechem.level.Level(level_code)
                solution = spacechem.solution.Solution(level, solution_code)
                with self.assertRaises(TimeoutError):  # TODO: spacechem.exceptions.InfiniteLoopError
                    solution.run()

    def test_invalid_outputs(self):
        '''Tests for solutions that should produce an InvalidOutput error.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.invalid_outputs):
            with self.subTest(msg=test_id):
                level = spacechem.level.Level(level_code)
                solution = spacechem.solution.Solution(level, solution_code)
                with self.assertRaises(spacechem.exceptions.InvalidOutputError):
                    solution.run()

    def test_runtime_collisions(self):
        '''Tests for solutions that should encounter errors when run.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.runtime_collisions):
            with self.subTest(msg=test_id):
                level = spacechem.level.Level(level_code)
                solution = spacechem.solution.Solution(level, solution_code)
                with self.assertRaises(Exception):
                    solution.run()

    def test_import_errors(self):
        '''Tests for solutions that shouldn't import successfully.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.import_errors):
            with self.subTest(msg=test_id):
                level = spacechem.level.Level(level_code)
                with self.assertRaises(Exception):
                    spacechem.solution.Solution(level, solution_code)


if __name__ == '__main__':
    # Use pickle to load the last test results and display a user-friendly summary of what tests performances went
    # up/down
    test_results_file = Path(__file__).parent / 'test_results.pickle'
    if test_results_file.exists():
        with test_results_file.open('rb') as f:
            LAST_TEST_RESULTS = pickle.load(f)

    unittest.main(verbosity=0, exit=False)
    print(f"Ran {num_subtests} subtests")

    # If successful, write the current times/mem usage to file
    with test_results_file.open('wb') as f:
        pickle.dump(LAST_TEST_RESULTS, f)
