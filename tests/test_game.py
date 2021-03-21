#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import unittest
import sys

# Insert the parent directory so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

import schem
import test_data

num_subtests = 0


def iter_game_test_data(soln_strs):
    global num_subtests
    for soln_str in soln_strs:
        num_subtests += 1

        # Parse only the metadata line so we can error out from the appropriate subTest if the full parse fails
        level_name, _, _, solution_name = schem.Solution.parse_metadata(soln_str)
        test_id = f'{level_name} - {solution_name}' if solution_name is not None else level_name

        yield test_id, soln_str


class TestGame(unittest.TestCase):
    def test_run_missing_score(self):
        """Test that run() does not require an expected score."""
        for test_id, solution_code in iter_game_test_data(test_data.missing_score):
            with self.subTest(msg=test_id):
                schem.run(solution_code)
                print(f"✅  {test_id}")

    def test_run_wrong_score(self):
        """Test that run() ignores whether the score does not match expected."""
        for test_id, solution_code in iter_game_test_data(test_data.wrong_score):
            with self.subTest(msg=test_id):
                schem.run(solution_code)
                print(f"✅  {test_id}")

    def test_validate_missing_score(self):
        """Test that validate() requires an expected score."""
        for test_id, solution_code in iter_game_test_data(test_data.missing_score):
            with self.subTest(msg=test_id):
                with self.assertRaises(ValueError):
                    schem.validate(solution_code)
                print(f"✅  {test_id}")

    def test_validate_wrong_score(self):
        """Test that validate() rejects successful solutions if the wrong score is specified."""
        for test_id, solution_code in iter_game_test_data(test_data.wrong_score):
            with self.subTest(msg=test_id):
                with self.assertRaises(schem.ScoreError):
                    schem.validate(solution_code)
                print(f"✅  {test_id}")

    def test_duplicate_level_name(self):
        '''Tests for solutions to levels of the same name.'''
        for test_id, soln_str in iter_game_test_data(test_data.duplicate_level_name_solutions):
            with self.subTest(msg=test_id):
                schem.validate(soln_str, verbose=False)
                print(f"✅  {test_id}")


if __name__ == '__main__':
    unittest.main(verbosity=0, exit=False)
    print(f"Ran {num_subtests} subtests")
