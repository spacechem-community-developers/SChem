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
        test_id = f'{level_name} - {solution_name}'

        yield test_id, soln_str


class TestGame(unittest.TestCase):
    def test_duplicate_level_name(self):
        '''Tests for solutions to levels of the same name.'''
        for test_id, soln_str in iter_game_test_data(test_data.duplicate_level_name_solutions):
            with self.subTest(msg=test_id):
                schem.validate(soln_str, verbose=False)


if __name__ == '__main__':
    unittest.main(verbosity=0, exit=False)
    print(f"Ran {num_subtests} subtests")
