#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import unittest
import sys

# Insert the parent directory to sys path so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

import schem
import test_data

num_subtests = 0


def iter_test_data(solution_codes):
    global num_subtests
    for solution_code in solution_codes:
        num_subtests += 1

        # Parse only the metadata line so we can error out from the appropriate subTest if the full parse fails
        level_name, _, _, solution_name = schem.Solution.parse_metadata(solution_code)
        test_id = f'{level_name} - {solution_name}'
        level_code = schem.levels[level_name] if level_name in schem.levels else test_data.test_levels[level_name]

        yield test_id, level_code, solution_code


class TestSolution(unittest.TestCase):
    def test_init_errors(self):
        '''Tests for solutions that shouldn't import successfully.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.import_errors):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                with self.assertRaises(Exception):
                    schem.Solution(level, solution_code)
                print(f"✅  {test_id}")

    def test_run_missing_score(self):
        """Test that run() does not require an expected score."""
        for test_id, level_code, solution_code in iter_test_data(test_data.missing_score):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                solution.run()
                print(f"✅  {test_id}")

    def test_run_wrong_score(self):
        """Test that run() ignores whether the score does not match expected."""
        for test_id, level_code, solution_code in iter_test_data(test_data.wrong_score):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                solution.run()
                print(f"✅  {test_id}")

    def test_run_runtime_collisions(self):
        '''Tests for solutions that should encounter errors when run.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.runtime_collisions):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(Exception):
                    solution.run()
                print(f"✅  {test_id}")

    def test_run_wall_collisions(self):
        '''Tests for solutions that should collide with a wall when run.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.wall_collisions):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(Exception) as context:
                    solution.run()
                self.assertTrue(' wall' in str(context.exception).lower())

                print(f"✅  {test_id}")

    def test_run_invalid_outputs(self):
        '''Tests for solutions that should produce an InvalidOutput error.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.invalid_outputs):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(schem.exceptions.InvalidOutputError):
                    solution.run()
                    print(f"✅  {test_id}")

    def test_run_infinite_loops(self):
        '''Tests for solutions that should exceed run()'s timeout.'''
        for test_id, level_code, solution_code in iter_test_data(test_data.infinite_loops):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(TimeoutError):  # TODO: schem.exceptions.InfiniteLoopError
                    solution.run()
                print(f"✅  {test_id}")

    def test_run_pause(self):
        for test_id, level_code, solution_code in iter_test_data(test_data.pauses):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(schem.exceptions.PauseException):
                    solution.run()
                print(f"✅  {test_id}")

    def test_validate_missing_score(self):
        """Test that validate() requires an expected score."""
        for test_id, level_code, solution_code in iter_test_data(test_data.missing_score):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(ValueError):
                    solution.validate()
                print(f"✅  {test_id}")

    def test_validate_wrong_score(self):
        """Test that validate() rejects successful solutions if the wrong score is specified."""
        for test_id, level_code, solution_code in iter_test_data(test_data.wrong_score):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                with self.assertRaises(schem.ScoreError):
                    solution.validate()
                print(f"✅  {test_id}")

    def test_validate_valid_solutions(self):
        '''Tests for solutions that should run to completion and match the expected score.
        Also outputs runtime performance stats.
        '''
        for test_id, level_code, solution_code in iter_test_data(test_data.valid_solutions):
            with self.subTest(msg=test_id):
                level = schem.Level(level_code)
                solution = schem.Solution(level, solution_code)
                solution.validate()
                print(f"✅  {test_id}")


if __name__ == '__main__':
    unittest.main(verbosity=0, exit=False)
    print(f"Ran {num_subtests} subtests")
