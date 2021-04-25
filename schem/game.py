#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .level import Level
from .levels import levels as built_in_levels, defense_names
from .exceptions import ScoreError
from .solution import Solution


def run(soln_str, level_code=None, level_codes=None, max_cycles=None, verbose=False, debug=False):
    """Given a solution string, run it against the given level(s). If none are provided, use the level name from the
    solution metadata to look for and use a built-in game level. Return the score as (cycles, reactors, symbols) or
    raise an exception if the solution does not run to completion.
    """
    assert level_code is None or level_codes is None, "Only one of level_code or level_codes may be specified"
    level_name, _, expected_score, _ = Solution.parse_metadata(soln_str)

    # Convert level_code convenience arg to same format as level_codes
    if level_code is not None:
        level_codes = [level_code]

    matching_levels = []
    if level_codes:
        levels = [Level(s) for s in level_codes]
        matching_levels = [level for level in levels if level.name == level_name]

        if not matching_levels:
            if len(levels) == 1:
                # If only a single level was provided, run against it anyway but warn of the mismatch
                if verbose:
                    print(f"Warning: Running solution against level {repr(levels[0].name)} that was originally"
                          + f" constructed for level {repr(level_name)}.")
                matching_levels.append(levels[0])
            else:
                raise Exception(f"No level `{level_name}` provided")
        elif len(matching_levels) > 1 and verbose:
            print(f"Warning: Multiple levels with name {level_name} given, checking solution against all of them.")
    else:
        # Determine the built-in game level to run the solution against based on the level name in its metadata
        if level_name in built_in_levels:
            if isinstance(built_in_levels[level_name], str):
                matching_levels.append(Level(built_in_levels[level_name]))
            else:
                matching_levels.extend(Level(export_str) for export_str in built_in_levels[level_name])

            if verbose and len(matching_levels) > 1:
                print(f"Warning: Multiple levels with name {level_name} found, checking solution against all of them.")
        elif level_name in defense_names:
            raise Exception("Defense levels unsupported")
        else:
            raise Exception(f"No known level `{level_name}`")

    score = None
    exceptions = []
    # TODO: Differentiate import error vs runtime error

    for level in matching_levels:
        try:
            solution = Solution(level=level, soln_export_str=soln_str)
            cur_score = solution.run(max_cycles=max_cycles, debug=debug)

            # Return the successful score if there was no expected score or it matched
            if expected_score is None or cur_score == expected_score:
                return cur_score

            # If the expected score is never found, preserve and return the first successful score
            if score is None:
                score = cur_score
        except Exception as e:
            exceptions.append(e)

    # If the solution ran successfully in any level, return that score. Otherwise, return the first failure
    if score is not None:
        return score
    else:
        raise exceptions[0]


def validate(soln_str, level_code=None, level_codes=None, max_cycles=None, verbose=False, debug=False):
    """Given a solution string, run it against the given level. If none is provided, use the level name from the
    solution metadata to look for and use a built-in game level.
    Raise an exception if the score does not match that indicated in the solution metadata.
    """
    level_name, author, expected_score, soln_name = Solution.parse_metadata(soln_str)
    if expected_score is None:
        raise ValueError("validate() requires a valid expected score in the first solution line (currently 0-0-0);"
                         + " please update it or use run() instead.")

    # TODO: Should use level_code's name if conflicting
    soln_descr = Solution.describe(level_name, author, expected_score, soln_name)

    if max_cycles is None:
        max_cycles = 2 * expected_score.cycles
    elif expected_score.cycles > max_cycles:
        raise ValueError(f"{soln_descr}: Cannot validate; expected cycles > max cycles ({max_cycles})")

    score = run(soln_str, level_code=level_code, level_codes=level_codes, max_cycles=max_cycles,
                verbose=verbose, debug=debug)

    if score != expected_score:
        raise ScoreError(f"{soln_descr}: Expected score {expected_score} but got {score}")

    if verbose:
        print(f"Validated {soln_descr}")
