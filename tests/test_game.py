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

    def test_validate_duplicate_level_name(self):
        """Tests for solutions to levels of the same name."""
        for test_id, soln_str in iter_game_test_data(test_data.duplicate_level_name_solutions):
            with self.subTest(msg=test_id):
                schem.validate(soln_str, verbose=False)
                print(f"✅  {test_id}")

    def test_run_json_duplicate_levels_import_error(self):
        """Test game.run with return_json=True and a solution which crashes on import to two same-name levels."""
        with self.assertRaises(Exception):
            soln_str = """SOLUTION:Sulfuric Acid,Zig,0-0-0,Unnamed Solution
COMPONENT:'custom-research-reactor',2,0,''"""
            schem.run(soln_str, return_json=True, verbose=False)

    def test_run_json_duplicate_levels_timeout_error(self):
        """Test game.run with return_json=True and a solution which imports successfully into one of two same-name
        levels, but times out.
        """
        soln_str = """SOLUTION:Sulfuric Acid,Zig,0-0-0
COMPONENT:'custom-research-reactor',2,0,''
MEMBER:'instr-start',0,0,128,0,7,0,0
MEMBER:'instr-start',180,0,32,1,7,0,0
MEMBER:'feature-bonder',-1,0,1,1,4,0,0
MEMBER:'feature-bonder',-1,0,1,2,4,0,0
MEMBER:'feature-bonder',-1,0,1,2,5,0,0
MEMBER:'feature-bonder',-1,0,1,1,5,0,0
MEMBER:'feature-splitter',-1,0,1,0,1,0,0"""

        with self.assertRaises(Exception):
            schem.run(soln_str, return_json=True, max_cycles=10, verbose=False)

    def test_run_json_duplicate_levels_success(self):
        """Test game.run with return_json=True and a solution which runs successfully in one of two same-name levels."""
        expected_json = {'level_name': "Sulfuric Acid",
                         'resnet_id': (3, 7, 1),
                         'cycles': 7208,
                         'reactors': 1,
                         'symbols': 36,
                         'author': "Zig",
                         'solution_name': "ResNet 3-7-1"}

        json = schem.run(test_data.duplicate_level_name_solutions[1], return_json=True, verbose=False)

        assert json == expected_json, f"Expected:\n{expected_json}\nbut got\n{json}"

    def test_validate_json_duplicate_levels_import_error(self):
        """Test game.validate with return_json=True and a solution which crashes on import to two same-name levels."""
        with self.assertRaises(Exception):
            soln_str = """SOLUTION:Sulfuric Acid,Zig,0-0-0,Unnamed Solution
COMPONENT:'custom-research-reactor',2,0,''"""
            schem.validate(soln_str, return_json=True, verbose=False)

    def test_validate_json_duplicate_levels_timeout_error(self):
        """Test game.validate with return_json=True and a solution which imports successfully into one of two same-name
        levels, but times out.
        """
        soln_str = """SOLUTION:Sulfuric Acid,Zig,0-0-0
COMPONENT:'custom-research-reactor',2,0,''
MEMBER:'instr-start',0,0,128,0,7,0,0
MEMBER:'instr-start',180,0,32,1,7,0,0
MEMBER:'feature-bonder',-1,0,1,1,4,0,0
MEMBER:'feature-bonder',-1,0,1,2,4,0,0
MEMBER:'feature-bonder',-1,0,1,2,5,0,0
MEMBER:'feature-bonder',-1,0,1,1,5,0,0
MEMBER:'feature-splitter',-1,0,1,0,1,0,0"""

        with self.assertRaises(Exception):
            schem.validate(soln_str, return_json=True, max_cycles=10, verbose=False)

    def test_validate_json_duplicate_levels_success(self):
        """Test game.validate with return_json=True and a solution which runs successfully in one of two same-name
        levels.
        """
        expected_json = {'level_name': "Sulfuric Acid",
                         'resnet_id': (3, 7, 1),
                         'cycles': 7208,
                         'reactors': 1,
                         'symbols': 36,
                         'author': "Zig",
                         'solution_name': "ResNet 3-7-1"}

        json = schem.run(test_data.duplicate_level_name_solutions[1], return_json=True, verbose=False)

        assert json == expected_json, f"Expected:\n{expected_json}\nbut got\n{json}"


if __name__ == '__main__':
    unittest.main(verbosity=0, exit=False)
    print(f"Ran {num_subtests} subtests")
