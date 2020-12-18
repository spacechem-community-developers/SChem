#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple
from pathlib import Path

from spacechem.level import Level
from spacechem import levels
from spacechem.solution import Solution
from spacechem.tests import test_data
from spacechem.components import *
from spacechem.reactor import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', nargs='?', const='', type=str,
                        help="Print an updating view of the solution while it runs."
                             + 'Can accept a comma-separated string with any of the following options:'
                             + "rN: Debug the reactor with idx N (if unspecified, overworld is shown in production lvls)."
                             + "cM: Start debugging from cycle M. Default 0."
                             + "E.g. --debug=r0,c1000 will start debugging the first reactor on cycle 1000")
    # TODO: Accept files with multiple solutions
    parser.add_argument('--level_file', type=Path, help="Optional file containing the puzzle definition")
    parser.add_argument('--solution_file', type=Path, required=True, help="File containing the solution to execute")
    args = parser.parse_args()

    if not args.solution_file.is_file():
        raise FileNotFoundError("Solution file not found")

    with args.solution_file.open() as f:
        solution_export_str = f.read()

    checked_levels = []
    if args.level_file:
        if not args.level_file.is_file():
            raise FileNotFoundError("Solution file not found")

        if args.level_file.suffix != '.puzzle':
            print("Warning: Parsing file without extension .puzzle as a SpaceChem level")

        with args.level_file.open() as f:
            checked_levels.append(Level(f.read().decode('utf-8')))
    else:
        # Determine the built-in game level to run the solution against based on its metadata
        level_name = Solution.get_level_name(solution_export_str)
        if level_name in levels.levels:
            if isinstance(levels.levels[level_name], str):
                checked_levels.append(Level(levels.levels[level_name]))
            else:
                checked_levels.extend(Level(export_str) for export_str in levels.levels[level_name])

            if len(checked_levels) > 1:
                print(f"Warning: Multiple levels with name {level_name} found, checking solution against all of them.")
        else:
            raise Exception(f"No known level {level_name}")

    for level in checked_levels:
        try:
            solution = Solution(level=level, soln_export_str=solution_export_str)

            debug = False
            DebugOptions = namedtuple("DebugOptions", ('reactor', 'cycle'))
            if args.debug is not None:
                # Default --debug with no args to the first reactor in research levels
                reactor = 0 if level['type'].startswith('research') else None
                cycle = 0
                for s in args.debug.split(','):
                    if s and s[0] == 'r':
                        reactor = int(s[1:])
                    elif s and s[0] == 'c':
                        cycle = int(s[1:])
                debug = DebugOptions(reactor, cycle)

            solution.validate(debug=debug, verbose=True)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
