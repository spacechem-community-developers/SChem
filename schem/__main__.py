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
from .level import Level
from .schem_random import SChemRandom
from .components import RandomInput
from .solution import Solution, DebugOptions
from .exceptions import ScoreError


def elapsed_readable(seconds, decimals=0):
    """Given an elapsed time in seconds and optionally number of decimal places to round seconds to,
    return a human-readable string describing it.
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    s = f"{round(seconds, decimals)}s"

    if minutes != 0:
        s = f"{minutes}m " + s

    if hours != 0:
        s = f"{hours}h " + s

    return s


def main(args: argparse.Namespace):
    total_start = time.time()
    with args.solution_file:  # args.solution_file is already open but `with` will close it for us
        if not args.solution_file.isatty():
            solutions_str = args.solution_file.read()
            solutions_src = 'Solution file'  # For more helpful error message
        else:  # If no STDIN input provided, instead of waiting on user input, use clipboard contents
            solutions_str = clipboard.paste()
            solutions_src = 'Clipboard'  # For more helpful error message

    # If multiple custom levels were provided, store them by name so we can grab the right one for each solution
    levels = {}
    if args.level_files:
        if args.verbose and any(level_file.suffix != '.puzzle' for level_file in args.level_files):
            print("Warning: Parsing file(s) without extension .puzzle as SpaceChem level(s)")

        for level_file in args.level_files:
            if not level_file.is_file():
                raise FileNotFoundError(f"{level_file} not found")

            with level_file.open(encoding='utf-8') as f:
                level = Level(f.read())
                if level.name in levels:
                    raise ValueError(f"Multiple levels named `{level.name}` provided.")

                levels[level.name] = level

    solutions = list(Solution.split_solutions(solutions_str))
    if not solutions:
        raise ValueError(f"{solutions_src} is empty.")

    jsons = []
    for solution_str in solutions:
        if args.verbose:
            start = time.time()

        try:
            # Parse metadata
            level_name, author, expected_score, soln_name = Solution.parse_metadata(solution_str)

            level = None
            # If custom level(s) were provided, pick the corresponding level to load this solution into
            if levels:
                if level_name in levels:
                    level = levels[level_name]
                elif len(levels) == 1:
                    # If only one level was given, we'll accept a solution with mismatched name (but warn the user)
                    level = next(iter(levels.values()))
                    if not args.json:
                        print(f"Warning: Validating solution against level `{level.name}` that was originally"
                              f" constructed for level `{level_name}`.")
                else:
                    raise ValueError(f"{Solution.describe(level_name, author, expected_score, soln_name)}:"
                                     f" no level named `{level_name}` provided")

            solution = Solution(solution_str, level=level)

            # Update the random input seed(s) if requested
            if args.seed is not None:
                random_inputs = [input_component for input_component in solution.inputs
                                 if isinstance(input_component, RandomInput)]
                # Do nothing for non-random levels
                if random_inputs:
                    # Set the first random input to the given seed, while maintaining the difference between it
                    # and the other random inputs' seeds
                    input_seed_increments = [random_input.seed - random_inputs[0].seed
                                             for random_input in random_inputs]
                    for random_input, increment in zip(random_inputs, input_seed_increments):
                        random_input.seed = (args.seed + increment) % (SChemRandom.MAX_SEED + 1)
                        random_input.reset()  # Reset to pick up the seed change

                    solution.expected_score = None  # Ignore expected score if we messed with the seed

            # Call validate if the solution has an expected score, else run
            # Also disable verbosity for --json or --export to avoid interfering with their STDOUT outputs
            result = solution.evaluate(max_cycles=args.max_cycles,
                                       strict=args.strict,
                                       check_precog=args.check_precog,
                                       max_precog_check_cycles=args.max_precog_check_cycles,
                                       verbosity=0 if args.json or args.export else 1 if not args.verbose else 2,
                                       debug=debug,
                                       _run=not args.no_run)

            # Handle verbosity 0 flags, finalizing JSON or printing the export respectively
            if args.json:
                if args.export:
                    result['export'] = solution.export_str()

                if 'error' in result:  # Make exceptions serializable
                    result['error'] = f"{type(result['error']).__name__}: {result['error']}"

                jsons.append(result)
            elif args.export:
                # Don't print the export if we encountered a validation error (note that we suppressed printouts
                # above so we have to print the error here)
                if 'error' in result:
                    print(f"{type(result['error']).__name__}: {result['error']}", file=sys.stderr)
                else:
                    print(solution.export_str())
            # Separate precog explanation messages from subsequent validation messages, to make things more readable and
            # make it clear which solution the message pertains to.
            # Avoids needing the full solution description in already-long precog messages.
            # Not needed in verbose mode since the subsequent elapsed time printout includes a separator.
            # Note that we have to cover both confirmed precog case and error in the precog check
            elif (args.check_precog and not args.verbose
                  # evaluate() doesn't print explanations for non-precog solutions, if not using --verbose
                  and (('precog' in result and result['precog'])
                       # Also add a separator after precog check error messages
                       # Make sure the run succeeded by checking cycles was set, and ignore ScoreError
                       or ('cycles' in result and 'error' in result and not isinstance(result['error'], ScoreError)))):
                print("")
        except Exception as e:
            if args.json:
                jsons.append({'error': f"{type(e).__name__}: {e}"})
            # Send all errors to STDERR in the case of more than 1 solution; in the single case fully raise it
            elif len(solutions) > 1:
                print(f"{type(e).__name__}: {e}", file=sys.stderr)
            else:
                raise e
        finally:
            if args.verbose:
                print(f"{elapsed_readable(time.time() - start, decimals=3)} elapsed\n")

    if args.json:
        # If a single solution is provided, output only its json, if multiple are provided, include them all in an array
        print(json.dumps(jsons[0] if len(jsons) == 1 else jsons,
                         indent=4))

    # If there were multiple solutions provided, report the total time elapsed
    if len(solutions) >= 2 and args.verbose:
        print(f"Total elapsed time: {elapsed_readable(time.time() - total_start, decimals=1)}")


if __name__ == '__main__':
    sys.tracebacklimit = 0  # Suppress traceback in STDERR output

    parser = argparse.ArgumentParser(prog='python -m schem',  # Don't show Usage: __main__.py
                                     description="Validate the solution(s) copied to the clipboard or in the given file.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', action='store_true', help="Print program version and exit")
    parser.add_argument('solution_file', type=argparse.FileType('r', encoding='utf-8'),  # Accept path arg or stdin pipe
                        nargs='?', default=sys.stdin,
                        help="File containing the solution(s) to execute.\n"
                             "If not provided, attempts to use the contents of the clipboard.")
    parser.add_argument('-l', '--level-file', '--puzzle-file', type=Path, action='append', dest='level_files',
                        metavar='LEVEL_FILE',  # otherwise LEVEL_FILES will be shown which is misleading
                        help="File containing the puzzle to check the solution(s) against.\n"
                             "If not provided, solution is checked against any official level with a matching title.\n"
                             "If flag is used multiple times, it will be checked that the solution validates for at\n"
                             "least one of the levels.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override the seed of the level's random input.\n"
                             "Expected score is ignored when this flag is used.\n"
                             "If multiple random inputs are present, sets the first input to the given seed, and\n"
                             "keeps the relative difference between it and other inputs' seeds the same.\n"
                             "E.g. if a level had two same-seed inputs, both will use the given seed.")
    parser.add_argument('--max-cycles', type=int, default=None,
                        help="Maximum cycle count solutions may be run to. Default 1.1x the expected score, or\n"
                             "1,000,000 if incomplete score metadata.\n"
                             "Pass -1 to run infinitely.")
    parser.add_argument('--check-precog', action='store_true',
                        help="Check if the given solution(s) are precognitive, per the community definition.\n"
                             "\nA solution is considered precognitive if either it fails for > 80%% of random seeds,\n"
                             "or it assumes knowledge of a *particular* input molecule other than the first.\n"
                             "In other words, if, for some n >= 2, there is a choice of the nth input\n"
                             "for which the solution will always fail regardless of the rest of the input sequence.\n"
                             "\nNote that this is calculated by running the solution a dozen to hundreds of times, so\n"
                             "the runtime will increase accordingly (for typical sub-10k cycle solutions, this will\n"
                             "be anywhere from sub-second up to ~15 seconds).\n"
                             "\nIf a solution is precog, prints a report of why, with similar reports for non-precog\n"
                             "solutions being omitted unless --verbose is used.\n"
                             "If --json is used, instead adds 'precog' (boolean) and 'precog_explanation' fields to\n"
                             "the JSON output, with the latter populated regardless of result.\n")
    parser.add_argument('--max-precog-check-cycles', type=int, default=None,
                        help="The maximum total cycle count that may be used by all precognition-check runs;\n"
                             "if this value is exceeded before sufficient confidence in an answer is obtained, an\n"
                             "error will be raised, or in the case of --json, the 'precog' field will be set to\n"
                             "null. Pass -1 to take as many runs as determined to be needed.\n"
                             "Default 2,000,000 cycles (this is sufficient for most sub-100k solutions).")
    parser.add_argument('--export', action='store_true',
                        help="Re-export the given solution export so its lines are in schem-standardized order, and\n"
                             "print it to STDOUT, suppressing default validation STDOUT messages.\n"
                             "If using --json, instead add an 'export' field to JSON output.")
    parser.add_argument('--no-run', action='store_true',
                        help="Do not run/validate the given solution(s), displaying only their basic metadata.\n"
                             "When used with --json, the cycles field will be set to the expected cycles, or if there\n"
                             "is no expected score, the cycle field will be omitted. reactors/symbols will still\n"
                             "be returned/validated since the latter don't require running the solution.")
    parser.add_argument('--strict', action='store_true',
                        help="Require that the solution(s) have an expected score to validate (not 0-0-0).")
    stdout_args = parser.add_mutually_exclusive_group()
    stdout_args.add_argument('--json', action='store_true',
                             help="Print JSON containing the run data, including level and solution metadata.\n"
                                  "If multiple solutions are provided, instead print an array of JSON objects.\n"
                                  "Suppresses default validation STDOUT messages.\n"
                                  "Fields: level_name, resnet_id (if ResNet level), author, cycles, reactors,\n"
                                  "        symbols, solution_name, precog (if --check-precog), precog_explanation\n"
                                  "        (ditto), export (if --export), and error (if the solution couldn't be\n"
                                  "        imported, crashed, didn't match expected score, or the precog check timed\n"
                                  "        out).")
    stdout_args.add_argument('--verbose', action='store_true',
                        help="In addition to default STDOUT messages/warnings, report the time schem takes to run,\n"
                             "and if running with --check-precog, report extra info when a solution is non-precog,\n"
                             "not just when it's precog.")
    parser.add_argument('--debug', nargs='?', const='', type=str,
                        help="Print an updating view of the solution while it runs.\n"
                             "Can accept a comma-separated string with any of the following options:\n"
                             "rR: Debug reactor R (0-indexed). If unspecified, overworld is shown in production lvls.\n"
                             "cC: Start debugging from cycle C. Default 0.\n"
                             "sS: Speed of debug in cycles/s. Default 10.\n"
                             "i: Show instructions. Default False since this mode can actually reduce readability.\n"
                             "E.g. --debug=r0,c1000,s0.5 starts debugging the first reactor on cycle 1000, at half a\n"
                             "cycle per second.")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit()

    # Flag sanity checks
    if args.no_run:
        if args.seed:
            raise parser.error("Cannot combine --no-run and --seed.")
        if args.max_cycles:
            raise parser.error("Cannot combine --no-run and --max-cycles.")
        if args.check_precog:
            raise parser.error("Cannot combine --no-run and --check-precog.")

    if args.max_precog_check_cycles and not args.check_precog:
        raise parser.error("--max-precog-check-cycles requires --check-precog")

    if args.export and not args.json:
        if args.check_precog:
            raise parser.error("--export overrides --check-precog STDOUT output.")
        if args.verbose:
            raise parser.error("--export overrides --verbose STDOUT output.")

    if args.seed and not (0 <= args.seed <= SChemRandom.MAX_SEED):
        raise ValueError(f"Seed must be from 0 to {SChemRandom.MAX_SEED}.")

    if args.max_cycles == -1:
        args.max_cycles = math.inf

    if args.max_precog_check_cycles == -1:
        args.max_precog_check_cycles = math.inf

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

    # Raise any caught errors from None to also suppress python's implicit exception chaining
    try:
        main(args)
    except Exception as e:
        raise e from None
