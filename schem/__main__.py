#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path

import clipboard

from .game import run, validate
from .solution import Solution, DebugOptions


def main():
    parser = argparse.ArgumentParser(description="Validate the solution(s) copied to the clipboard or in the given file.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('solution_file', type=Path, nargs='?',
                        help="File containing the solution(s) to execute."
                             + " If not provided, attempts to use the contents of the clipboard.")
    parser.add_argument('--level_files', type=Path, nargs='+',
                        help="Optional file(s) containing the puzzle to check the solution(s) against")
    parser.add_argument('--max_cycles', type=int, default=None,
                        help="Maximum cycle count solutions may be run to. Default double the expected score, or"
                             " 1,000,000 if incomplete score metadata. Pass -1 to run infinitely.")
    parser.add_argument('--debug', nargs='?', const='', type=str,
                        help="Print an updating view of the solution while it runs."
                             + "\nCan accept a comma-separated string with any of the following options:"
                             + "\nrR: Debug the reactor with idx R (if unspecified, overworld is shown in production lvls)."
                             + "\ncC: Start debugging from cycle C. Default 0."
                             + "\nsS: Speed of debug in cycles/s. Default 10."
                             + "\ni: Show instructions. Default False since this mode can actually reduce readability."
                             + "\nE.g. --debug=r0,c1000,s0.5 will start debugging the first reactor on cycle 1000, at half a cycle/s")
    args = parser.parse_args()

    debug = False
    if args.debug is not None:
        reactor = None
        cycle = 0
        speed = 10
        show_instructions = False
        for s in args.debug.split(','):
            if s and s[0] == 'r':
                reactor = int(s[1:])
            elif s and s[0] == 'c':
                cycle = int(s[1:])
            elif s and s[0] == 's':
                speed = float(s[1:])
            elif s == 'i':
                show_instructions = True
        debug = DebugOptions(reactor=reactor, cycle=cycle, speed=speed, show_instructions=show_instructions)

    if args.solution_file:
        if not args.solution_file.is_file():
            raise FileNotFoundError("Solution file not found")

        with args.solution_file.open(encoding='utf-8') as f:
            solutions_str = f.read()
        solutions_src = 'Solution file'  # For more helpful error message
    else:
        solutions_str = clipboard.paste()
        solutions_src = 'Clipboard'  # For more helpful error message

    level_codes = []
    if args.level_files:
        if any(level_file.suffix != '.puzzle' for level_file in args.level_files):
            print("Warning: Parsing file(s) without extension .puzzle as SpaceChem level(s)")

        for level_file in args.level_files:
            if not level_file.is_file():
                raise FileNotFoundError(f"{level_file} not found")

            with level_file.open(encoding='utf-8') as f:
                level_codes.append(f.read())

    solutions = list(Solution.split_solutions(solutions_str))
    if not solutions:
        raise ValueError(f"{solutions_src} contents are empty.")

    if args.max_cycles == -1:
        args.max_cycles = math.inf

    for solution_str in solutions:
        try:
            # Call validate if the solution has an expected score, else run
            expected_score = Solution.parse_metadata(solution_str)[2]
            if expected_score is not None:
                validate(solution_str, level_codes=level_codes, max_cycles=args.max_cycles, verbose=True, debug=debug)
            else:
                run(solution_str, level_codes=level_codes, max_cycles=args.max_cycles, verbose=True, debug=debug)
        except Exception as e:
            if len(solutions) == 1:
                raise e
            else:
                print(f"{type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
