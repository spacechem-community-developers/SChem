#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
from timeit import Timer
import unittest
import sys

# Insert the parent directory to sys path so schem is accessible even if it's not available system-wide
sys.path.insert(1, str(Path(__file__).parent.parent))

from schem.molecule import Molecule

num_subtests = 0
LAST_TEST_RESULTS = {}


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


class TestMolecule(unittest.TestCase):
    def test_isomorphism_valid(self):
        '''Tests for molecules that should be valid isomorphisms of each other.'''
        test_mol_pairs = (
            ("Wall of carbon",
             """Carbon;C;00611;10611;20611;30601;01611;12611;11611;21611;31601;32601;22611;02611;03611;13611;23611;33601;34601;24611;14611;04611;05611;15611;25611;35601;36601;26611;16611;06611;07610;17610;27610;37600""",
             "Carbon;C;00611;10611;20611;30601;01611;12611;11611;21611;31601;32601;22611;02611;03611;13611;23611;33601;34601;24611;14611;04611;05611;15611;25611;35601;36601;26611;16611;06611;07610;17610;27610;37600"),
            ("Doping rotated",
             """Silicon;Si;001411;111511;101411;011411;201411;301401;211411;311401;021411;121411;221411;321401;031410;131410;231410;331400""",
             """Silicon;Si;001411;111411;101411;011411;201411;301401;211411;311401;021411;121411;221511;321401;031410;131410;231410;331400"""),
            ("Pyramid",
             "Pyramid;C~01~06;12611;13611;14610;35601;32601;03610;34601;25610;36600;33601;22611;23611;24611;31601;30601;21611",
             "Pyramid;C~01~06;33601;22611;23611;24611;31601;30601;21611;12611;13611;14610;35601;32601;03610;34601;25610;36600"),)
        global num_subtests

        for test_id, mol_A_str, mol_B_str in test_mol_pairs:
            test_id = 'test_isomorphism_valid: ' + test_id
            with self.subTest(msg=test_id):
                num_subtests += 1
                mol_A = Molecule.from_json_string(mol_A_str)
                mol_B = Molecule.from_json_string(mol_B_str)

                self.assertTrue(mol_A.isomorphic(mol_B), "Molecule.isomorphic() unexpectedly returned False")

                # Check the time performance of the isomorphism algorithm
                timer = Timer('mol_A.isomorphic(mol_B)', globals={'mol_A': mol_A, 'mol_B': mol_B})

                if test_id not in LAST_TEST_RESULTS:
                    loops = max(max(timer.autorange()[0] for _ in range(2)) // 2, 5)
                else:
                    loops = LAST_TEST_RESULTS[test_id]['loops']

                min_time = min(timer.repeat(repeat=loops, number=1))

                if test_id in LAST_TEST_RESULTS:
                    metrics_str = f"{min_time:.5f}s (b. of {loops}) ({percent_diff_str(LAST_TEST_RESULTS[test_id]['min_time'], min_time)})"
                    metrics_str += f" | {test_id}"
                else:
                    metrics_str = f"{min_time:.5f}s (b. of {loops}) (NEW) | {test_id}"
                print(metrics_str)

                LAST_TEST_RESULTS[test_id] = {'loops': loops, 'min_time': min_time}


if __name__ == '__main__':
    test_results_file = Path(__file__).parent / 'last_performance_results.pickle'
    if test_results_file.exists():
        with test_results_file.open('rb') as f:
            LAST_TEST_RESULTS = pickle.load(f)

    unittest.main(verbosity=0, failfast=True, exit=False)
    print(f"Ran {num_subtests} subtests")

    # If successful, write the current times/mem usage to file
    with test_results_file.open('wb') as f:
        pickle.dump(LAST_TEST_RESULTS, f)
