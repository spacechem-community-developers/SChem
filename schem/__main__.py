#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple
from pathlib import Path

import clipboard

from .game import validate
from .solution import Solution

def main():
    parser = argparse.ArgumentParser(description="Validate the solution(s) copied to the clipboard or in the given file.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('solution_file', type=Path, nargs='?',
                        help="File containing the solution(s) to execute."
                             + " If not provided, attempts to use the contents of the clipboard.")
    parser.add_argument('--level_file', type=Path,
                        help="Optional file containing the puzzle to check the solution(s) against")
    parser.add_argument('--debug', nargs='?', const='', type=str,
                        help="Print an updating view of the solution while it runs."
                             + "\nCan accept a comma-separated string with any of the following options:"
                             + "\nrR: Debug the reactor with idx R (if unspecified, overworld is shown in production lvls)."
                             + "\ncC: Start debugging from cycle C. Default 0."
                             + "\nsS: Change the default debug cycles/second by a factor of S."
                             + "\nE.g. --debug=r0,c1000,s0.5 will start debugging the first reactor on cycle 1000, at 0.5x speed")
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
        solutions_str = clipboard.paste().replace('\r\n', '\n')  # Make sure windows doesn't crap in our string

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
