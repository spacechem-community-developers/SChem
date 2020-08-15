#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import timeit
import unittest

import spacechem
import test_data


def iter_test_data(data_dict):
    for level_code in data_dict:
        level = spacechem.level.ResearchLevel(level_code)

        for solution_code in data_dict[level_code]:
            solution = spacechem.solution.Solution(level, solution_code)
            yield level, solution

class TestGame(unittest.TestCase):
    def test_valid_solutions(self):
        for level, solution in iter_test_data(test_data.valid_levels_and_solutions):
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                self.assertEqual(spacechem.game.score_solution(solution),
                                 solution.expected_score)

                # Check the time performance of the solver
                avg_time = timeit(lambda: spacechem.game.score_solution(solution),
                                  number=100, globals=globals()) / 100
                print(f'Avg {avg_time:.6f} seconds to run {level.get_name()} {solution.name}')

    def test_infinite_loops(self):
        for level, solution in iter_test_data(test_data.infinite_loops):
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                with self.assertRaises(spacechem.game.InfiniteLoopError):
                    spacechem.game.score_solution(solution)

    def test_invalid_outputs(self):
        for level, solution in iter_test_data(test_data.invalid_outputs):
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                with self.assertRaises(spacechem.game.InvalidOutputError):
                    spacechem.game.score_solution(solution)

    def test_collisions(self):
        for level, solution in iter_test_data(test_data.collisions):
            with self.subTest(msg=f'{level.get_name()} {solution.name}'):
                with self.assertRaises(spacechem.game.ReactionError):
                    spacechem.game.score_solution(solution)


if __name__ == '__main__':
    unittest.main(verbosity=0)
