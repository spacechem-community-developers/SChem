#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from .level import Level
from .levels import levels as built_in_levels, defense_names, resnet_ids
from .exceptions import ScoreError
from .solution import Score, Solution, DebugOptions


def run(soln_str: str, level_code=None, level_codes=None, max_cycles=None, return_json=False, verbose=False,
        debug: Optional[DebugOptions] = False):
    """Wrapper for Solution.run which identifies the level to run in as needed, based on the solution metadata.

    Run the given solution, and return the (cycles, reactors, symbols) Score obtained. Raise an error if the solution
    cannot run to completion or exceeds max_cycles.

    Args:
        soln_str: Solution string as exported by SpaceChem Community Edition.
        level_code / level_codes: Puzzle export string or strings (only one or the other arg may be provided).
            If multiple levels are given, solution will be run against all with titles matching the level title in its
            metadata, and an exception raised if none match. If only one level is provided, the solution will be run
            against it, irrespective of title.

            If level_code(s) is not provided, the level name from the solution metadata is used to search for a matching
            built-in game level(s).

            When multiple built-in or provided level titles match the solution's metadata, the solution will be run
            against all in turn until one matches the expected score. If no level matches the expected score, the first
            successful run's score will be returned. If the solution does not complete any of the levels, an exception
            is raised.
            Note that while some built-in game levels have duplicate titles, currently all of these have conflicting
            level features (bonders etc.), and therefore a solution can't successfully load into / solve multiple
            built-in levels.
        max_cycles: Maximum cycle count to run to. Default double the expected cycle count in the solution metadata,
            or 1,000,000 cycles if no expected score (use math.inf if you don't fear infinite loop solutions).
        return_json: If True, instead of a Score return a dict including the usual score fields, but also the level
            title, ResearchNet volume-issue-puzzle tuple (None if not a ResearchNet level), and solution author/title.

            Additionally, raise an exception only if the solution cannot be imported into any level; if the solution
            encounters a reaction error or exceeds max_cycles, just return None in the dict's 'cycles' field and
            return the metadata of the first level that the solution could be successfully imported into.

            Default False.
        verbose: If True, print warnings if there is not exactly one level with title matching the solution metadata.
                 Default False.
        debug: Print an updating view of the solution while running; see DebugOptions. Default False.
    """
    assert level_code is None or level_codes is None, "Only one of level_code or level_codes may be specified"
    level_name, _, expected_score, _ = Solution.parse_metadata(soln_str)

    # Convert level_code convenience arg to same format as level_codes
    if level_code is not None:
        level_codes = [level_code]

    matching_levels = []
    matching_resnet_ids = []  # Used for reporting resnet ID of the run level when return_json is True
    if level_codes:
        levels = [Level(s) for s in level_codes]
        matching_levels = [level for level in levels if level.name == level_name]
        matching_resnet_ids = [None for _ in range(len(matching_levels))]

        if not matching_levels:
            if len(levels) == 1:
                # If only a single level was provided, run against it anyway but warn of the mismatch
                if verbose:
                    print(f"Warning: Running solution against level {repr(levels[0].name)} that was originally"
                          + f" constructed for level {repr(level_name)}.")
                matching_levels.append(levels[0])
                matching_resnet_ids.append(None)
            else:
                raise Exception(f"No level `{level_name}` provided")
        elif len(matching_levels) > 1 and verbose:
            print(f"Warning: Multiple levels with name {level_name} given, checking solution against all of them.")
    else:
        # Determine the built-in game level to run the solution against based on the level name in its metadata
        if level_name in built_in_levels:
            if isinstance(built_in_levels[level_name], str):
                matching_levels.append(Level(built_in_levels[level_name]))
                matching_resnet_ids.append(resnet_ids[level_name] if level_name in resnet_ids else None)
            else:
                matching_levels.extend(Level(export_str) for export_str in built_in_levels[level_name])
                matching_resnet_ids.extend(resnet_ids[level_name] if level_name in resnet_ids
                                           else [None for _ in range(len(built_in_levels[level_name]))])

            if verbose and len(matching_levels) > 1:
                print(f"Warning: Multiple levels with name {level_name} found, checking solution against all of them.")
        elif level_name in defense_names:
            raise Exception("Defense levels unsupported")
        else:
            raise Exception(f"No known level `{level_name}`")

    ret_val = None
    exceptions = []

    for level, resnet_id in zip(matching_levels, matching_resnet_ids):
        try:
            solution = Solution(level=level, soln_export_str=soln_str)
            cur_score = cycles, reactors, symbols = solution.run(max_cycles=max_cycles, debug=debug)

            run_data = {'level_name': level.name,
                        'resnet_id': resnet_id,
                        'cycles': cycles,
                        'reactors': reactors,
                        'symbols': symbols,
                        'author': solution.author,
                        'solution_name': solution.name}

            # Return the successful run if there was no expected score or it matched the expected cycle count
            if expected_score is None or cur_score.cycles == expected_score.cycles:
                return run_data if return_json else cur_score

            # Preserve the first successful score (in case the expected score is never found)
            if ret_val is None:
                ret_val = run_data if return_json else cur_score
        except Exception as e:
            exceptions.append(e)

    # Raise the first error if no successful run was found
    if ret_val is not None:
        return ret_val
    else:
        raise exceptions[0]


def validate(soln_str: str, level_code=None, level_codes=None, max_cycles=None, return_json=False, verbose=False,
             debug: Optional[DebugOptions] = False):
    """Sibling of Solution.validate which identifies the level to run in as needed, based on the solution metadata.

    Run the given solution, and raise an exception if the score does not match that indicated in the solution metadata.

    Args:
        soln_str: Solution string as exported by SpaceChem Community Edition.
        level_code / level_codes: Puzzle export string or strings (only one or the other arg may be provided).
            If multiple levels are given, solution will be run against all with titles matching the level title in its
            metadata, and an exception raised if none match. If only one level is provided, the solution will be run
            against it, irrespective of title.

            If level_code(s) is not provided, the level name from the solution metadata is used to search for a matching
            built-in game level(s).

            When multiple built-in or provided level titles match the solution's metadata, the solution will be run
            against all in turn until one matches the expected score, or raise an error if none do.
            Note that while some built-in game levels have duplicate titles, currently all of these have conflicting
            level features (bonders etc.), and therefore a solution can't successfully load into / solve multiple
            built-in levels.
        max_cycles: Maximum cycle count to run to. Default double the expected cycle count in the solution metadata.
        return_json: If True, instead of a Score return a dict including the usual score fields, but also the level
            title, ResearchNet volume-issue-puzzle tuple (None if not a ResearchNet level), and solution author/title.

            Additionally, raise an exception only if the solution cannot be imported into any level; if the solution
            encounters a reaction error or exceeds max_cycles, just return None in the dict's 'cycles' field and
            return the metadata of the first level that the solution could be successfully imported into.

            Default False.
        verbose: If True, print warnings if there is not exactly one level with title matching the solution metadata.
                 Default False.
        debug: Print an updating view of the solution while running; see DebugOptions. Default False.

    """
    level_name, author, expected_score, soln_name = Solution.parse_metadata(soln_str)
    if expected_score is None:
        raise ValueError("validate() requires a valid expected score in the first solution line (currently 0-0-0);"
                         + " please update it or use run() instead.")

    # TODO: Should use level_code's name if conflicting
    soln_descr = Solution.describe(level_name, author, expected_score, soln_name)

    if max_cycles is not None and expected_score.cycles > max_cycles:
        raise ValueError(f"{soln_descr}: Cannot validate; expected cycles > max cycles ({max_cycles})")

    ret_val = run(soln_str, level_code=level_code, level_codes=level_codes, max_cycles=max_cycles,
                  return_json=return_json, verbose=verbose, debug=debug)

    if not return_json:
        score = ret_val
    # run(return_json=True) only raises an error on solution load errors; runtime errors just set cycles to None
    elif ret_val['cycles'] is None:
        raise Exception(f"{soln_descr}: Expected score {expected_score} but solution crashed or timed out.")
    else:
        score = Score(ret_val['cycles'], ret_val['reactors'], ret_val['symbols'])

    # Validate the score
    if score != expected_score:
        raise ScoreError(f"{soln_descr}: Expected score {expected_score} but got {score}")

    if verbose:
        print(f"Validated {soln_descr}")

    if return_json:
        return ret_val
