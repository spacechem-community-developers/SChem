#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .level import Level
from . import levels
from .solution import Solution


def run(soln_str, level_code=None, verbose=False, debug=False):
    """Given a solution string, run it against the given level. If none is provided, use the level name from the
    solution metadata to look for and use a built-in game level. Return the score as (cycles, reactors, symbols) or
    raise an exception if the solution does not run to completion.
    """
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
            cur_score = solution.run(debug=debug)

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

def validate(soln_str, level_code=None, verbose=False, debug=False):
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

    try:
        score = run(soln_str, level_code=level_code, verbose=verbose, debug=debug)
    except Exception as e:
        # Mention the invalid solution via a chained exception of the same type
        raise type(e)(f"Error while validating {soln_descr}: {e}") from e

    if score != expected_score:
        raise Exception(f"Expected score {expected_score} but got {score}")

    if verbose:
        print(f"Validated {soln_descr}")
