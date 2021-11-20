#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from pathlib import Path
import unittest
import sys

# Insert the parent directory to sys path so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

import schem
from schem import Solution
import test_data

num_subtests = 0


def iter_test_data(solution_codes):
    global num_subtests
    for solution_code in solution_codes:
        num_subtests += 1

        # Parse only the metadata line so we can error out from the appropriate subTest if the full parse fails
        level_name, _, _, solution_name = schem.Solution.parse_metadata(solution_code)
        test_id = f'{level_name} - {solution_name}' if solution_name is not None else level_name
        # Leave the level selection to Solution's constructor unless it's a custom level
        level_code = test_data.test_levels[level_name] if level_name in test_data.test_levels else None

        yield test_id, level_code, solution_code


class TestPrecognition(unittest.TestCase):
    def test_is_precognitive_positive(self):
        """Test solutions that are precognitive."""
        for test_id, level_code, solution_code in iter_test_data(test_data.precog_solutions):
            with self.subTest(msg=test_id):
                solution = Solution(solution_code, level=level_code)
                start = time.time()
                self.assertTrue(solution.is_precognitive())
                print(f"✅ {time.time() - start:.3f}s - {test_id}")

    def test_is_precognitive_negative(self):
        """Test solutions that are non-precognitive."""
        for test_id, level_code, solution_code in iter_test_data(test_data.non_precog_solutions):
            with self.subTest(msg=test_id):
                solution = Solution(solution_code, level=level_code)
                start = time.time()
                self.assertFalse(solution.is_precognitive())
                print(f"✅ {time.time() - start:.3f}s - {test_id}")


if __name__ == '__main__':
    unittest.main(verbosity=0, failfast=True, exit=False)
    print(f"Ran {num_subtests} subtests")
