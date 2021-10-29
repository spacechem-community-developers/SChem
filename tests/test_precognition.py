#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from pathlib import Path
import unittest
import sys

# Insert the parent directory to sys path so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

import schem
from schem import Level, Solution
import test_data

num_subtests = 0


def iter_test_data(solution_codes):
    global num_subtests
    for solution_code in solution_codes:
        num_subtests += 1

        # Parse only the metadata line so we can error out from the appropriate subTest if the full parse fails
        level_name, _, _, solution_name = schem.Solution.parse_metadata(solution_code)
        test_id = f'{level_name} - {solution_name}' if solution_name is not None else level_name
        level_code = schem.levels[level_name] if level_name in schem.levels else test_data.test_levels[level_name]

        yield test_id, level_code, solution_code


class TestPrecognition(unittest.TestCase):
    def test_is_precognitive_positive(self):
        """Test that run() does not require an expected score."""
        for test_id, level_code, solution_code in iter_test_data(test_data.precog_solutions):
            with self.subTest(msg=test_id):
                level = Level(level_code)
                solution = Solution(level, soln_export_str=solution_code)
                start = time.time()
                self.assertTrue(schem.is_precognitive(solution))
                print(f"✅ {time.time() - start:.3f}s - {test_id}")

    def test_is_precognitive_negative(self):
        """Test that run() does not require an expected score."""
        for test_id, level_code, solution_code in iter_test_data(test_data.non_precog_solutions):
            with self.subTest(msg=test_id):
                level = Level(level_code)
                solution = Solution(level, soln_export_str=solution_code)
                start = time.time()
                self.assertFalse(schem.is_precognitive(solution))
                print(f"✅ {time.time() - start:.3f}s - {test_id}")


if __name__ == '__main__':
    unittest.main(verbosity=0, exit=False)
    print(f"Ran {num_subtests} subtests")
