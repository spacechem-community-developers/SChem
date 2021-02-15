#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple
from pathlib import Path

import clipboard

from .level import Level
from . import levels
from .solution import Solution
from .components import *


def run(soln_str, level_code=None, verbose=False, debug=False):
    """Given a solution string, run it against the specified level and return a Score."""
    level_name, _, expected_score, _ = Solution.parse_metadata(soln_str)

    matching_levels = []
    if level_code is not None:
        level = Level(level_code)

        if verbose and level_name != level.name:
            print(f"Warning: Running solution against level {repr(level.name)} that was originally"
                  + f" constructed for level {repr(level_name)}.")

        matching_levels.append(level)
    else:
        # Determine the built-in game level to run the solution against based on the level name in its metadata
        if level_name in levels.levels:
            if isinstance(levels.levels[level_name], str):
                matching_levels.append(Level(levels.levels[level_name]))
            else:
                matching_levels.extend(Level(export_str) for export_str in levels.levels[level_name])

            if verbose and len(matching_levels) > 1:
                print(f"Warning: Multiple levels with name {level_name} found, checking solution against all of them.")
        else:
            raise Exception(f"No known level `{level_name}`")

    score = None
    exceptions = []
    # TODO: Differentiate import error vs runtime error

    for level in matching_levels:
        try:
            solution = Solution(level=level, soln_export_str=soln_str)
            score = solution.run(debug=debug)

            # Exit early if the first level we checked matched the expected score
            if score == expected_score:
                return score
        except Exception as e:
            exceptions.append(e)

    # If the solution ran successfully in any level, return that score. Otherwise, return the first failure
    if score is not None:
        return score
    else:
        raise exceptions[0]

def validate(soln_str, level_code=None, verbose=True, debug=False):
    level_name, author, expected_score, soln_name = Solution.parse_metadata(soln_str)
    # TODO: Should use level_code's name if conflicting
    soln_descr = Solution.describe(level_name, author, expected_score, soln_name)

    try:
        score = run(soln_str, level_code=level_code, verbose=verbose, debug=debug)
    except Exception as e:
        # Mention the invalid solution via a chained exception of the same type
        raise type(e)(f"Error while validating {soln_descr}: {e}") from e

    assert score == expected_score, (f"Expected score {'-'.join(str(x) for x in expected_score)}"
                                     f" but got {'-'.join(str(x) for x in score)}")
    if verbose:
        print(f"Validated {soln_descr}")

def main():
    parser = argparse.ArgumentParser(description="Validate the solution copied to the clipboard or in the given file",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--debug', nargs='?', const='', type=str,
                        help="Print an updating view of the solution while it runs."
                             + "\nCan accept a comma-separated string with any of the following options:"
                             + "\nrN: Debug the reactor with idx N (if unspecified, overworld is shown in production lvls)."
                             + "\ncM: Start debugging from cycle M. Default 0."
                             + "\nE.g. --debug=r0,c1000 will start debugging the first reactor on cycle 1000")
    parser.add_argument('solution_file', type=Path, nargs='?',
                        help="File containing the solution(s) to execute."
                             + " If not provided, attempts to use the contents of the clipboard.")
    parser.add_argument('--level_file', type=Path,
                        help="Optional file containing the puzzle to check the solution(s) against")
    args = parser.parse_args()

    debug = False
    if args.debug is not None:
        DebugOptions = namedtuple("DebugOptions", ('reactor', 'cycle', 'speed'))
        reactor = None
        cycle = 0
        speed = 1
        for s in args.debug.split(','):
            if s and s[0] == 'r':
                reactor = int(s[1:])
            elif s and s[0] == 'c':
                cycle = int(s[1:])
            elif s and s[0] == 's':
                speed = float(s[1:])
        debug = DebugOptions(reactor, cycle, speed)

    if args.solution_file:
        if not args.solution_file.is_file():
            raise FileNotFoundError("Solution file not found")

        with args.solution_file.open() as f:
            solutions_str = f.read()
    else:
        solutions_str = clipboard.paste().replace('\r', '')  # Make sure windows doesn't crap in our string

    level_code = None
    if args.level_file:
        if not args.level_file.is_file():
            raise FileNotFoundError("Solution file not found")

        if args.level_file.suffix != '.puzzle':
            print("Warning: Parsing file without extension .puzzle as a SpaceChem level")

        with args.level_file.open() as f:
            level_code = f.read().decode('utf-8')

    solutions = list(Solution.split_solutions(solutions_str))
    for solution_str in solutions:
        try:
            validate(solution_str, level_code=level_code, verbose=True, debug=debug)
        except Exception as e:
            if len(solutions) == 1:
                raise e
            else:
                print(repr(e))


if __name__ == '__main__':
    main()
