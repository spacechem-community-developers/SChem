#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple

from spacechem.level import Level
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
    # TODO: accept level/solution files, including files with multiple solutions
    # parser.add_argument('--level_file'  # Maybe not needed if we just lookup the level code
    # parser.add_argument('--solution_file'
    args = parser.parse_args()

    level_code = tuple(test_data.valid.keys())[14]
    level = Level(level_code)

    # TODO: If level not specified, automatically find the relevant main or ResearchNet level to run against
    #       In the few cases of name ambiguities we can just run in both levels and see which it solves

    solution_code = tuple(test_data.valid[level_code])[0]
    solution = Solution(level=level, soln_export_str=solution_code)

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

    #print('Exporting and re-verifying the exported string:')
    #solution_reexport_str = solution.export_str()
    #Solution(level=level, soln_export_str=solution_reexport_str).validate(verbose=True)


if __name__ == '__main__':
    main()
