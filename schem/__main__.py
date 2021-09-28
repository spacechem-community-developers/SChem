#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
import sys
import time

import clipboard

from ._version import __version__
from .game import run, validate
from .solution import Solution, DebugOptions


def main(args: argparse.Namespace):
    with args.solution_file:  # args.solution_file is already open but `with` will close it for us
        if not args.solution_file.isatty():
            solutions_str = args.solution_file.read()
            solutions_src = 'Solution file'  # For more helpful error message
        else:  # If no STDIN input provided, instead of waiting on user input, use clipboard contents
            solutions_str = clipboard.paste()
            solutions_src = 'Clipboard'  # For more helpful error message

    level_codes = None
    if args.level_files:
        level_codes = []
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

    jsons = []
    for solution_str in solutions:
        try:
            # Call validate if the solution has an expected score, else run
            # Also disable verbosity in case of --json, since we'll be printing the json
            level_name, author, expected_score, soln_name = Solution.parse_metadata(solution_str)
            if expected_score is not None:
                ret_val = validate(solution_str, level_codes=level_codes, max_cycles=args.max_cycles,
                                   return_json=args.json, check_precog=args.check_precog,
                                   max_precog_check_cycles=args.max_precog_check_cycles,
                                   verbose=(not args.quiet), stderr_on_precog=args.verbose,
                                   debug=debug)
            else:
                ret_val = run(solution_str, level_codes=level_codes, max_cycles=args.max_cycles,
                              return_json=args.json, check_precog=args.check_precog,
                              max_precog_check_cycles=args.max_precog_check_cycles,
                              verbose=(not args.quiet), stderr_on_precog=args.verbose,
                              debug=debug)

                if not args.quiet:
                    print(f"Validated {Solution.describe(level_name, author, ret_val, soln_name)}")

            if args.json:
                jsons.append(ret_val)

        except Exception as e:
            # If not in --json mode, print all errors instead of exiting early in the case of multiple solutions
            # In --json mode users are relying on our STDOUT output so we have to make sure we exit properly with an
            # error code instead of printing errors
            if len(solutions) == 1 or args.json:
                raise e
            else:
                print(f"{type(e).__name__}: {e}")

    if args.json:
        # If a single solution is provided, output only its json, if multiple are provided, include them all in an array
        print(json.dumps(jsons[0] if len(jsons) == 1 else jsons,
                         indent=4))


if __name__ == '__main__':
    sys.tracebacklimit = 0  # Suppress traceback in STDERR output

    parser = argparse.ArgumentParser(description="Validate the solution(s) copied to the clipboard or in the given file.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='store_true', help="Print program version and exit")
    parser.add_argument('solution_file', type=argparse.FileType('r'), nargs='?',  # Accept either path arg or stdin pipe
                        default=sys.stdin,
                        help="File containing the solution(s) to execute."
                             + " If not provided, attempts to use the contents of the clipboard.")
    parser.add_argument('-l', '--level-file', '--puzzle-file', type=Path, action='append', dest='level_files',
                        metavar='LEVEL_FILE',  # otherwise LEVEL_FILES will be shown which is misleading
                        help="File containing the puzzle to check the solution(s) against.\n"
                             "If not provided, solution is checked against any official level with a matching title.\n"
                             "If flag is used multiple times, it will be checked that the solution validates for at"
                             " least one of the levels.")
    parser.add_argument('--max-cycles', type=int, default=None,
                        help="Maximum cycle count solutions may be run to. Default 1.1x the expected score, or\n"
                             "1,000,000 if incomplete score metadata.\n"
                             "Pass -1 to run infinitely.")
    parser.add_argument('--check-precog', action='store_true',
                        help="Check if the given solution(s) are precognitive, per the current community definition.\n"
                             "\nA solution is considered precognitive if either it fails for >= 50%% of random seeds,\n"
                             "or it assumes knowledge of a *particular* input molecule other than the first.\n"
                             "In other words, if, for some n >= 2, there is a choice of the nth input\n"
                             "for which the solution will always fail regardless of the rest of the input sequence.\n"
                             "\nNote that this is calculated by running the solution a dozen to hundreds of times, so\n"
                             "the runtime will increase accordingly (for typical sub-10k cycle solutions, this will\n"
                             "be anywhere from sub-second up to ~15 seconds).\n"
                             "\nIf called with the --json field, adds a 'precog' field to the output JSON.\n"
                             "Otherwise, prints check result human readably.")
    parser.add_argument('--max-precog-check-cycles', type=int, default=None,
                        help="The maximum total cycle count that may be used by all precognition-check runs;\n"
                             "if this value is exceeded before sufficient confidence in an answer is obtained, an\n"
                             "error will be raised, or in the case of --json, the 'precog' field will be set to\n"
                             "null. Pass -1 to take as many runs as determined to be needed.\n"
                             "Default 2,000,000 cycles (this is sufficient for basically any sub-100k solution).")
    stdout_args = parser.add_mutually_exclusive_group()
    stdout_args.add_argument('--json', action='store_true',
                             help="Print JSON containing the run data, including level and solution metadata.\n"
                                  "If multiple solutions are provided, instead print an array of JSON objects.\n"
                                  "--json also suppresses default validation STDOUT messages.")
    stdout_args.add_argument('--quiet', action='store_true',
                             help="Suppress default STDOUT messages/warnings.")
    parser.add_argument('--verbose', action='store_true',
                        help="In addition to standard STDOUT messages/warnings, report the time schem takes to run,\n"
                              "and if running with --check_precog, also report to STDERR explanations for why a\n"
                              "solution is precognitive. Note that --json/--quiet only suppress STDOUT and will keep\n"
                              "the latter STDERR messages.")
    parser.add_argument('--debug', nargs='?', const='', type=str,
                        help="Print an updating view of the solution while it runs.\n"
                             "Can accept a comma-separated string with any of the following options:\n"
                             "rR: Debug the reactor with idx R (if unspecified, overworld is shown in production lvls).\n"
                             "cC: Start debugging from cycle C. Default 0.\n"
                             "sS: Speed of debug in cycles/s. Default 10.\n"
                             "i: Show instructions. Default False since this mode can actually reduce readability.\n"
                             "E.g. --debug=r0,c1000,s0.5 starts debugging the first reactor on cycle 1000, at half a cycle/s")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit()

    if args.max_cycles == -1:
        args.max_cycles = math.inf

    if args.max_precog_check_cycles == -1:
        args.max_precog_check_cycles = math.inf

    # Suppress STDOUT if we're outputting JSON
    if args.json:
        args.quiet = True

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

    start = time.time()

    # Raise any caught errors from None to also suppress python's implicit exception chaining
    try:
        main(args)
    except Exception as e:
        raise e from None
    finally:
        if args.verbose and not args.quiet:  # Sorry
            end = time.time()
            print(f"{end - start:.3f}s elapsed")
