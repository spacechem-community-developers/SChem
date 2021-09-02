#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import copy
from dataclasses import dataclass
from itertools import product
import platform
import time
from typing import Optional

import cursor
import rich
# TODO: Hoping to avoid default number-highlighting via Console(highlight=False).print but it seems to break ANSI resets

from .components import Component, Input, Output, Reactor, Recycler, DisabledOutput
from .waldo import InstructionType
from .exceptions import ScoreError
from .grid import *
from .level import OVERWORLD_COLS, OVERWORLD_ROWS
from .terrains import terrains, MAX_TERRAIN_INT

IS_WINDOWS = platform.system() == 'Windows'


@dataclass
class DebugOptions:
    """Debug options for running a solution.

    reactor: The reactor to debug. If None, overworld is shown for production levels, or only reactor for researches.
    cycle: The cycle to start debugging at. Default 0.
    speed: The speed to run the solution at, in cycles / s. Default 10.
    show_instructions: Whether to show waldo instructions. Default False as this can actually reduce readability.
    """
    reactor: Optional[int] = None
    cycle: int = 0
    speed: float = 10
    show_instructions: bool = False


class Score(namedtuple("Score", ('cycles', 'reactors', 'symbols'))):
    '''Immutable class representing a SpaceChem solution score.'''
    __slots__ = ()

    def __str__(self):
        '''Convert to the format used in solution metadata.'''
        return '-'.join(str(i) for i in self)

    @classmethod
    def is_score_str(cls, s):
        '''Return True if the given string is formatted like a Spacechem score, e.g. 45-1-14.
        Incomplete score formats (0-0-0, and Incomplete-r-s from the old SCT) are included.
        '''
        parts = s.split('-')
        return (len(parts) == 3
                and (parts[0].isdigit() or parts[0] == 'Incomplete')  # Thinks 045-1-14 is a score but whatever
                and all(part.isdigit() for part in parts[1:]))

    @classmethod
    def from_str(cls, s):
        """Return a Score object, or None if the string represents an incomplete score."""
        parts = s.split('-')
        assert len(parts) == 3, "String is not a score"

        # Return None politely on either modern SC format or old SaveChemTool format for incomplete solution
        if s == '0-0-0' or (parts[0] == 'Incomplete' and all(part.isdigit() for part in parts[1:])):
            return None

        return cls(*(int(x) for x in s.split('-')))


class Solution:
    '''Class for constructing and running game entities from a given level object and solution code.'''
    __slots__ = ('level_name', 'author', 'expected_score', 'name',
                 'level', 'components', 'cycle')

    DEFAULT_MAX_CYCLES = 1_000_000  # Default timeout for solutions whose expected cycle count is unknown

    # Convenience properties
    @property
    def inputs(self):
        return (component for component in self.components if isinstance(component, Input))

    @property
    def reactors(self):
        return (component for component in self.components if isinstance(component, Reactor))

    @property
    def outputs(self):
        return (component for component in self.components if isinstance(component, Output))

    @property
    def symbols(self):
        return sum(sum(len(waldo) for waldo in reactor.waldos) for reactor in self.reactors)

    @property
    def description(self):
        return self.describe(self.level.name, self.author, self.expected_score, self.name)

    @classmethod
    def parse_metadata(cls, s):
        '''Given a solution export string or its SOLUTION line, return the level name, author, expected score,
        and solution name (or None if either of the latter are not set).
        Due to the use of comma-separated values in export lines, commas in the level or author name may cause
        errors in the parsing. To combat this, it is assumed that author names contain no commas, and that the first
        comma-separated string (after the second comma) which is formatted like a score (e.g. 45-1-14) is the score
        field. This is sufficient to uniquely identify the other fields.
        As a consequence, if either the author name contains a comma or the author/level names look like scores,
        it is possible for some fields to get mis-parsed. This is still a little better than SC which gets
        completely messed up by any commas in the level name.
        '''
        s = s.replace('\r\n', '\n').strip('\n').split('\n', maxsplit=1)[0]  # Get first non-empty line

        assert s.startswith('SOLUTION:'), "Given string is not a SpaceChem solution"
        s = s[len('SOLUTION:'):]

        # Iteratively extract fields, accounting for quoting (which SC does for strings containing a comma)
        fields = []
        while s:
            if s[0] != "'":
                # Unquoted field, just grab everything up to next comma
                field, *rest = s.split(',', maxsplit=1)
                s = ','.join(rest)
                fields.append(field)
            else:
                # Quoted field. Search for the next un-doubled (un-escaped) quote and ensure it's at the end of the csv field
                closing_quote_indices = [i for i, c in enumerate(s) if c == "'"][1::2]  # Every second quote
                end_quote_idx = next((i for i in closing_quote_indices if s[i + 1:i + 2] != "'"), None)  # First unpaired closing quote

                if end_quote_idx is None or not (end_quote_idx == len(s) - 1  # End of string or end of csv field
                                                 or s[end_quote_idx + 1] == ','):
                    raise Exception("Invalid quoted string in solution metadata")

                fields.append(s[1:end_quote_idx].replace("''", "'"))  # Unescape internal quotes
                s = s[end_quote_idx + 2:]  # Skip ahead past the quote and comma

        assert len(fields) >= 3, "Missing fields in solution metadata"

        # Modern SC quotes all fields containing a comma so our fields should be split correctly by now.
        # However in older versions fields with commas weren't quoted, resulting in parsing errors for e.g. levels with
        # commas in the name. If the third field isn't a valid score, attempt to handle this case elegantly by finding
        # the real score field and assuming any extra fields (commas) before it were the level name's fault
        if not Score.is_score_str(fields[2]):
            # Look for the next score-like field and assume it is the expected score
            score_field_idx = next((i for i in range(3, len(fields)) if Score.is_score_str(fields[i])),
                                   None)
            assert score_field_idx is not None, "Solution metadata missing expected score"

            # Merge all excess pre-score fields into the level name, assuming the author was comma-free
            fields = [','.join(fields[:score_field_idx - 1])] + fields[score_field_idx - 1:]

        level_name = fields[0]
        author_name = fields[1]
        expected_score = Score.from_str(fields[2])  # Note that this returns None for an incomplete score
        soln_name = ','.join(fields[3:]) if len(fields) > 3 else None

        return level_name, author_name, expected_score, soln_name

    @classmethod
    def describe(cls, level_name, author, expected_score, soln_name):
        """Return a pretty-printed string describing the given solution metadata."""
        soln_descr = f"[{level_name}]"
        if expected_score is not None:
            soln_descr += f' {expected_score}'
        if soln_name is not None:
            soln_descr += f' "{soln_name}"'
        soln_descr += f" by {author}"

        return soln_descr

    @classmethod
    def split_solutions(cls, solns_str):
        """Given a string potentially containing multiple solutions, return an iterator of them.
        Returns an empty iterator if the given string has no non-empty lines.
        """
        solns_str = '\n'.join(s for s in solns_str.replace('\r\n', '\n').split('\n') if s)  # Remove empty lines

        # Ensure first non-empty line is a solution string
        if solns_str and not solns_str.startswith('SOLUTION:'):
            raise ValueError("Invalid solution string: expected SOLUTION: on line 1")

        # Split with newline prefix in case some fucker names themselves SOLUTION:
        return (f'SOLUTION:{s}' for s in ('\n' + solns_str).split('\nSOLUTION:')[1:])

    # TODO: Solution constructor should probably accept either level or level_code for convenience
    def __init__(self, level, soln_export_str=None):
        self.level = level
        self.level_name = level['name']
        self.name = None
        self.author = 'Unknown'
        self.expected_score = None
        self.cycle = 0

        # Set up the level terrain so we can look up input/output positions and any blocked terrain
        if level['type'].startswith('research'):
            terrain_id = 'research'
        elif level['type'].startswith('production'):
            # Main game levels have unique terrain which we have to hardcode D:
            # We can at least avoid name collisions if we deliberately don't add the terrain field to their JSONs
            # TODO: This is a bit dangerous; the game auto-adds a terrain field if it ever gets its hands on the json
            terrain_id = level['name'] if ('terrain' not in level
                                           and level['name'] in terrains) else level['terrain']

            # SC caps oversized terrain IDs
            if isinstance(terrain_id, int) and terrain_id > MAX_TERRAIN_INT:
                terrain_id = MAX_TERRAIN_INT
        elif level['type'] == 'sandbox':
            terrain_id = 7
        else:
            raise ValueError(f"Unrecognized level type {repr(level['type'])}")

        # Add level-defined entities - including any fixed-position reactors/pipes
        # Store components by posn for now, for convenience. We'll re-order and store them in self.components later
        posn_to_component = {}  # Note that components are referenced by their top-left corner posn

        # Inputs
        input_zone_types = None
        if self.level['type'].startswith('research'):
            input_zone_types = ('input-zones',)
            output_zone_type = 'output-zones'
        elif self.level['type'] == 'production':
            input_zone_types = ('random-input-zones', 'fixed-input-zones')
            output_zone_type = 'output-zones'
        elif self.level['type'].startswith('production-community-edition'):
            # CE Productions finally give up on the useless distinction between fixed and random zones
            input_zone_types = ('random-input-components', 'programmed-input-components')
            output_zone_type = 'output-components'
        else:
            # Sandbox
            input_zone_types = ('random-input-zones', 'fixed-input-zones')
            output_zone_type = None

        for input_zone_type in input_zone_types:
            if input_zone_type not in self.level:
                continue

            if isinstance(self.level[input_zone_type], dict):
                for i, input_dict in self.level[input_zone_type].items():
                    i = int(i)

                    # For some reason production fixed inputs are defined less flexibly than in researches
                    # TODO: Input class should probably accept raw molecule string instead of having to stuff it into a dict
                    #       One does wonder why Zach separated random from fixed inputs in productions but not researches...
                    if isinstance(input_dict, str):
                        # Convert from molecule string to input zone dict format
                        input_dict = {'inputs': [{'molecule': input_dict, 'count': 12}]}

                    # Due to a bug in SpaceChem, random-input-zones may end up populated with an empty input
                    # entry in a non-random level. Skip it if that's the case
                    # TODO: Make CE issue on github for this
                    if not input_dict['inputs']:
                        continue

                    # Fetch the component type and posn from the terrain's presets based on this input's index
                    component_type, component_posn = terrains[terrain_id][input_zone_type][i]

                    posn_to_component[component_posn] = Input(input_dict=input_dict,
                                                              _type=component_type, posn=component_posn,
                                                              is_research=self.level['type'].startswith('research'))
            else:
                # CE production levels expect input types/positions to be manually specified in the input dict
                for input_dict in self.level[input_zone_type]:
                    # Skip empty random input zones
                    if 'molecules' in input_dict and not input_dict['molecules']:
                        continue

                    new_component = Input(input_dict=input_dict, is_research=self.level['type'].startswith('research'))
                    posn_to_component[new_component.posn] = new_component

        # Sandboxes use an alternate format for specifying the programmed input
        # Convert to the standard CE-format programmed input dict
        if self.level['type'] == 'sandbox' and 'programmed-input-molecules' in self.level:
            input_dict = {'starting-molecules': [self.level['programmed-input-molecules'][str(i)]
                                                 for i in self.level['programmed-input-start']],
                          'repeating-molecules': [self.level['programmed-input-molecules'][str(i)]
                                                  for i in self.level['programmed-input-repeat']]}
            component_type, component_posn = terrains[terrain_id]['programmed-input']

            posn_to_component[component_posn] = Input(input_dict=input_dict,
                                                      _type=component_type, posn=component_posn,
                                                      is_research=False)

        # Outputs
        if output_zone_type is not None:
            if isinstance(self.level[output_zone_type], dict):
                for i, output_dict in self.level[output_zone_type].items():
                    i = int(i)

                    # I'd handle this in the Output constructor but I refuse to enable Zach's madness
                    if self.level['type'] == 'production':
                        output_dict = copy.deepcopy(output_dict)  # To avoid mutating the level
                        output_dict['count'] *= 4  # Zach pls

                    component_type, component_posn = terrains[terrain_id][output_zone_type][i]
                    posn_to_component[component_posn] = Output(output_dict=output_dict,
                                                               _type=component_type, posn=component_posn)
            else:
                for output_dict in self.level[output_zone_type]:
                    new_component = Output(output_dict=output_dict)
                    posn_to_component[new_component.posn] = new_component

        # Add disabled output components for the unused outputs of research levels (crash if given a molecule)
        if self.level['type'].startswith('research') and not ('has-large-output' in self.level
                                                              and self.level['has-large-output']):
            for i in range(2):
                if i not in set(int(x) for x in self.level[output_zone_type]):
                    _, component_posn = terrains[terrain_id][output_zone_type][i]
                    posn_to_component[component_posn] = DisabledOutput(_type='disabled-output', posn=component_posn)

        # Preset reactors
        if self.level['type'].startswith('research'):
            # Preset one reactor, treating the level dict as its component dict
            # TODO: _type='reactor' is hacky but ensures Component.__init__ knows what it's dealing with
            new_component = Reactor(self.level.dict, _type='reactor', posn=Position(col=2, row=0))
            posn_to_component[new_component.posn] = new_component
        else:
            # Recycler
            if 'has-recycler' in self.level and self.level['has-recycler']:
                component_type, component_posn = terrains[terrain_id]['recycler']
                posn_to_component[component_posn] = Recycler(_type=component_type, posn=component_posn)

            # Preset components
            for component_list_key in ('pass-through-counters', 'components-with-output', 'other-components'):
                if component_list_key in self.level:
                    for component_dict in self.level[component_list_key]:
                        # Add custom reactor attributes
                        if component_dict['type'].startswith('freeform-custom-reactor-'):
                            i = int(component_dict['type'].split('-')[-1]) - 1
                            # TODO: Use py3.9's dict union operator
                            component_dict = {**self.level['custom-reactors'][i], **component_dict}

                        # As a convenience, propagate top-level "disallowed-instructions" into each component if defined
                        if 'disallowed-instructions' in self.level and 'disallowed-instructions' not in component_dict:
                            component_dict['disallowed-instructions'] = self.level['disallowed-instructions']

                        new_component = Component(component_dict)

                        if component_dict['type'] == 'drag-qpipe-in':
                            # TODO: Bit hacky but since user levels can't use this, just going to make sure the
                            #       Teleporter output is defined before its input in "Teleporters"
                            new_component.destination = posn_to_component[Position(col=component_dict['destination-x'],
                                                                                   row=component_dict['destination-y'])]

                        posn_to_component[new_component.posn] = new_component

        # Add solution-defined components and update preset level components (e.g. inputs' pipes, preset reactor contents)
        if soln_export_str is not None:
            # Get the first non-empty line
            soln_metadata_str, *split_remainder = soln_export_str.replace('\r\n', '\n').strip('\n').split('\n', maxsplit=1)
            components_str = '' if not split_remainder else split_remainder[0].strip('\n')
            self.level_name, self.author, self.expected_score, self.name = self.parse_metadata(soln_metadata_str)

            if components_str:
                assert components_str.startswith('COMPONENT:'), "Unexpected data on line 1"

            soln_defined_component_posns = set()  # Used to check that the solution doesn't doubly-define a component

            for component_str in ('COMPONENT:' + s for s in ('\n' + components_str).split('\nCOMPONENT:')[1:]):
                component_type, component_posn = Component.parse_metadata(component_str)

                # Ensure this component hasn't already been created/updated by the solution
                if component_posn in soln_defined_component_posns:
                    raise ValueError(f"Solution defines component at {component_posn} twice")
                soln_defined_component_posns.add(component_posn)

                # Create a raw instance of the component if it doesn't exist yet
                if component_posn not in posn_to_component:
                    # Ensure this is a legal component type for the level (e.g. drag-starter-reactor -> has-starter)
                    reactor_type_flag = f'has-{component_type.split("-")[1]}'
                    assert ((reactor_type_flag in self.level and self.level[reactor_type_flag])
                            # Misleadingly, allowed-reactor-types includes storage tanks
                            or ('allowed-reactor-types' in self.level
                                and component_type in self.level['allowed-reactor-types'])
                            # Sandbox levels always allow sandbox reactor, infinite storage tank, and printers
                            or (self.level['type'] == 'sandbox' and component_type in ('drag-sandbox-reactor',
                                                                                       'drag-storage-tank-infinite',
                                                                                       'drag-printer-passthrough',
                                                                                       'drag-printer-output'))), \
                        f"New component type {component_type} (at {component_posn}) is not legal in this level"

                    component_dict = {}
                    if component_type.startswith('freeform-custom-reactor-'):
                        # Add custom reactor attributes if needed
                        i = int(component_type.split('-')[-1]) - 1
                        component_dict = self.level['custom-reactors'][i]

                    # As a convenience for defining early game levels without needing custom reactors, propagate
                    # any top-level "disallowed-instructions" value from the level into each reactor
                    if 'disallowed-instructions' in self.level and 'disallowed-instructions' not in component_dict:
                        component_dict['disallowed-instructions'] = self.level['disallowed-instructions']

                    component = Component(component_dict, _type=component_type, posn=component_posn)

                    # Ensure the component is within the overworld bounds (note that presets don't have this
                    #       restriction, e.g. Going Green). Its pipes will be bounds-checked later since even pipes of
                    #       preset components should have bounds checks.
                    # Note: It looks like other than in Going Green, SC deletes out-of-bound preset components
                    #       However, so long as its at the puzzle level and not the solution, being a little more
                    #       permissive than SC should be okay
                    component_posns = set(product(range(component.posn[0], component.posn[0] + component.dimensions[0]),
                                                  range(component.posn[1], component.posn[1] + component.dimensions[1])))
                    assert all(0 <= p[0] < OVERWORLD_COLS and 0 <= p[1] < OVERWORLD_ROWS
                               for p in component_posns), f"Component {component_type} is out of bounds"

                    posn_to_component[component_posn] = component

                # Update the existing component (e.g. its pipes or reactor internals)
                # 'mutable-pipes' is not used by SC, but is added to handle the fact that Ω-Pseudoethyne disallows
                # mutating a preset reactor's 1-long pipe whereas custom levels allow it.
                # The custom level code for Ω-Pseudoethyne is the only one to set this property (and sets it to false)
                update_pipes = (not self.level['type'].startswith('research')
                                and ('mutable-pipes' not in self.level
                                     or self.level['mutable-pipes']))
                component = posn_to_component[component_posn]
                try:
                    component.update_from_export_str(component_str, update_pipes=update_pipes)
                except Exception as e:
                    if not self.level['type'].startswith('research'):
                        raise type(e)(f"{component.type} at {component.posn}: {e}") from e

                    raise e

                # TODO: Ensure the updated component pipes are within the overworld bounds

        # Now that we're done updating components, check that all components/pipes are validly placed
        # TODO: Should probably just be a method validating self.components, and also called at start of run()
        blocked_posns = set()
        # Tracks which directions pipes may cross through each other, and implicitly also which cells contain pipes
        pipe_posns_to_dirns = {}

        # Add impassable terrain
        if not self.level['type'].startswith('research'):
            blocked_posns.update(terrains[terrain_id]['obstructed'])

        # Check for component/pipe collisions
        for component in posn_to_component.values():
            # Add this component's cells to the blacklist
            # Check for collisions and add this component's main body (non-pipe) posns to the blacklist
            component_posns = set(product(range(component.posn[0], component.posn[0] + component.dimensions[0]),
                                          range(component.posn[1], component.posn[1] + component.dimensions[1])))
            assert (component_posns.isdisjoint(blocked_posns)
                    and component_posns.isdisjoint(pipe_posns_to_dirns.keys())), \
                f"Component at {component.posn} colliding with terrain or another component"
            blocked_posns.update(component_posns)

            # Check pipes are valid
            for i, pipe in enumerate(component.out_pipes):
                if not pipe:  # TODO: Should maybe yell if the pipe is empty instead of None
                    continue

                assert len(pipe.posns) == len(set(pipe.posns)), "Pipe overlaps with itself"

                # Make sure the pipe is properly connected to this component
                # TODO: this code would break if any component had 3 output pipes but none do...
                assert pipe.posns[0] == (component.dimensions[0], ((component.dimensions[1] - 1) // 2) + i), \
                    f"Pipe {i} is not connected to parent component {component.type} at {component.posn}"

                # Ensure this pipe doesn't collide with a component, terrain, or the overworld edges
                # Recall that pipe posns are defined relative to their parent component's posn
                cur_pipe_posns = set((component.posn[0] + pipe_posn[0], component.posn[1] + pipe_posn[1])
                                     for pipe_posn in pipe.posns)
                assert all(0 <= p[0] < OVERWORLD_COLS and 0 <= p[1] < OVERWORLD_ROWS
                           for p in cur_pipe_posns), f"Component {component.type} pipe {i} is out of bounds"
                assert cur_pipe_posns.isdisjoint(blocked_posns), \
                    f"Collision(s) between pipe and terrain/component at {cur_pipe_posns & blocked_posns}"

                # Identify each pipe segment as vertical, horizontal, or a turn, and ensure all overlaps are legal.
                # We can do the latter by tracking which pipe directions have already been 'occupied' by a pipe in any
                # given cell - with a turn occupying both directions. We'll use RIGHT/DOWN for horizontal/vertical.

                # To find whether a pipe is straight or not, we need to iterate over each pipe posn with its neighbors
                # Ugly edge case: When a pipe starts by moving vertically, it prevents a horizontal pipe from
                #                 overlapping it because it 'turns' into its reactor.
                #                 But when a vertical pipe ends next to a reactor, it does not prevent another
                #                 pipe from crossing through horizontally - in fact, the horizontal pipe will be the
                #                 one to connect to the reactor. Therefore, the 'neighbor' of the last posn should
                #                 be itself, in order to count the pipe as straight either vertically or horizontally.
                #                 However, the 'neighbor' of the first posn should be itself with 1 subtracted
                #                 from the column (the connection to the reactor), to ensure it prevents horizontal
                #                 crossovers (note that no components have pipes on their top/bottom edges so this is
                #                 safe).
                #                 This also ensures that a 1-long pipe will count as horizontal and not vertical.
                for prev, cur, next_ in zip([pipe.posns[0] + LEFT] + pipe.posns[:-1],
                                            pipe.posns,
                                            pipe.posns[1:] + [pipe.posns[-1]]):
                    real_posn = component.posn + cur

                    if real_posn not in pipe_posns_to_dirns:
                        pipe_posns_to_dirns[real_posn] = set()

                    # Pipe is not vertical (blocks the horizontal direction)
                    if not prev.col == cur.col == next_.col:
                        assert RIGHT not in pipe_posns_to_dirns[real_posn], f"Illegal pipe overlap at {real_posn}"
                        pipe_posns_to_dirns[real_posn].add(RIGHT)

                    # Pipe is not horizontal (blocks the vertical direction)
                    if not prev.row == cur.row == next_.row:
                        assert UP not in pipe_posns_to_dirns[real_posn], f"Illegal pipe overlap at {real_posn}"
                        pipe_posns_to_dirns[real_posn].add(UP)

        # Store all components, sorting them left-to-right then top-to-bottom to ensure correct I/O priorities
        # (since posns are col first then row, this is just a regular sort on the posn tuples)
        self.components = [component for posn, component in sorted(posn_to_component.items())]

        # Connect the ends of all pipes to available component inputs
        for component in self.components:
            for pipe in component.out_pipes:
                if pipe is None:
                    continue

                pipe_end = component.posn + pipe.posns[-1]

                # Ugly edge case:
                # If there are two pipe ends in the same spot, the vertical one should not connect to a component.
                # We can tell when this is happening thanks to the posn_dirn dict we built up earlier while checking for
                # pipe collisions. The end of a pipe is always 'straight', so if there are two directions occupied in
                # the cell this pipe ends in, we know there are two pipes here. We must ignore this pipe if its second
                # last segment is not to the left of its last segment.
                if (pipe_posns_to_dirns[pipe_end] == {UP, RIGHT}
                        and len(pipe) >= 2 and pipe.posns[-2] != pipe.posns[-1] + LEFT):
                    continue

                # This is a bit of a hack, but by virtue of output/reactor/etc. shapes, the top-left corner of a
                # component is always 1 up and right of its input pipe, plus one more row for lower inputs (up to 3 for
                # recycler). Blindly check the 3 possible positions for components that could connect to this pipe

                # Exception: Teleporter component from Corvi's "Teleporters"
                # TODO: Calculate components' actual in pipe locations same way we do for out pipes
                component_posn = pipe_end + (1, 0)
                if component_posn in posn_to_component:
                    other_component = posn_to_component[component_posn]
                    if other_component.dimensions[1] == 1 and len(other_component.in_pipes) == 1:
                        other_component.in_pipes[0] = pipe
                    continue

                for i in range(3):
                    component_posn = pipe_end + (1, -1 - i)
                    if component_posn in posn_to_component:
                        other_component = posn_to_component[component_posn]
                        if other_component.dimensions[1] > 1 and len(other_component.in_pipes) >= i + 1:
                            other_component.in_pipes[i] = pipe
                        break

    def export_str(self):
        # Solution metadata
        fields = [self.level['name'], self.author, str(self.expected_score)]
        if self.name is not None:
            fields.append(self.name)

        # Quote field names that contain a comma or start with a quote (and escape their internal quotes)
        export_str = "SOLUTION:" + ','.join(f"""'{s.replace("'", "''")}'""" if s and (',' in s or s[0] == "'") else s
                                            for s in fields)

        # Components
        # Exclude inputs whose pipes are length 1 (unmodified), and out-pipeless components like outputs and recycler
        # TODO: This doesn't cover inputs with preset pipes > 1 long - which also shouldn't be included
        #       Really it's probably just every unmodified preset component.
        for component in self.components:
            if component.out_pipes and not (isinstance(component, Input) and len(component.out_pipe) == 1):
                export_str += '\n' + component.export_str()

        return export_str

    def __str__(self):
        '''Return a string representing the overworld of this solution including all components/pipes, and the cycle.'''
        # 1 character per tile
        grid = [[' ' for _ in range(OVERWORLD_COLS)] for _ in range(OVERWORLD_ROWS)]

        # Display each component with #'s
        for component in self.components:
            # Visual bounds set since components are allowed to hang outside the play area (e.g. in Going Green)
            for c, r in product(range(max(0, component.posn.col), min(OVERWORLD_COLS,
                                                                      component.posn.col + component.dimensions[0])),
                                range(max(0, component.posn.row), min(OVERWORLD_ROWS,
                                                                      component.posn.row + component.dimensions[1]))):
                grid[r][c] = '#'
            if isinstance(component, Output):
                # Display the output count in the component. All output are 2x3 so we should have room
                for i, s in enumerate(reversed(str(component.current_count))):
                    grid[component.posn.row + 1][component.posn.col + 1 - i] = s

            # Display each pipe with -, \, /, +, |, or a . for tiles containing a molecule
            for pipe in component.out_pipes:
                last_posn = Position(-1, -1)
                for relative_posn, molecule in zip(pipe.posns, pipe):
                    posn = component.posn + relative_posn

                    if molecule is not None:
                        grid[posn.row][posn.col] = '.'
                    elif posn.col == last_posn.col:
                        grid[posn.row][posn.col] = '|'
                    else:  # first pipe from component won't match last_posn
                        grid[posn.row][posn.col] = '-'

                    last_posn = posn

        result = f"_{OVERWORLD_COLS * '_'}_\n"
        for row in grid:
            result += f"|{''.join(row)}|\n"
        result += f"‾{OVERWORLD_COLS * '‾'}‾"

        result += f'\nCycle: {self.cycle}'

        return result

    def debug_print(self, duration=0.5, reactor_idx=None, flash_features=True, show_instructions=False):
        '''Print the currently running solution state then clear it from the terminal.
        Args:
            duration: Seconds before clearing the printout from the screen. Default 0.5.
            reactor_idx: If specified, only print the contents of the reactor with specified index.
            flash_features: Whether to flash activated features, inputs, and outputs.
        '''
        if reactor_idx is None:
            output = str(self)
        else:
            output = list(self.reactors)[reactor_idx].__str__(flash_features=flash_features,
                                                              show_instructions=show_instructions)
            output += f'\nCycle: {self.cycle}'

        # Print the current state
        rich.print(output)  # Could use end='' but that makes keyboard interrupt output ugly

        # TODO: this sleep is additive with the time schem takes to run each cycle so debug is always too slow
        time.sleep(duration)

        # Use the ANSI escape code for moving to the start of the previous line to reset the terminal cursor
        cursor_reset = (output.count('\n') + 1) * "\033[F"  # +1 for the implicit newline print() appends
        if IS_WINDOWS:
            rich.print(cursor_reset, end='')  # TODO: No idea why the normal print doesn't work in windows ConEmu
        else:
            print(cursor_reset, end='')

    def validate_components(self):
        '''Validate that this solution is legal (whether or not it can run to completion).
        Called at the start of each run to ensure that no trickery was done on this object post-construction.
        '''
        # Make sure the reactor limit has been respected
        if self.level['type'].startswith('production'):
            assert sum(1 if isinstance(component, Reactor) else 0
                       for component in self.components) <= self.level['max-reactors'], "Reactor limit exceeded"

    def cycle_movement(self):
        """Move contents of all components one cycle's-worth forward."""
        for component in self.components:
            try:
                component.move_contents(self.cycle)
            except Exception as e:
                # Mention the originating reactor in errors when possible for multi-reactor solutions
                if len(list(self.reactors)) != 1 and isinstance(component, Reactor):
                    for i, reactor in enumerate(self.reactors):
                        if component is reactor:
                            # Mention the reactor index via a chained exception of the same type
                            raise type(e)(f"Reactor {i}: {e}") from e
                raise e

    def run(self, max_cycles=None, debug: Optional[DebugOptions] = False):
        '''Run this solution, returning a Score or else raising an exception if the level was not solved.

        Args:
            max_cycles: Maximum cycle count to run to. Default double the expected cycle count in the solution metadata,
                        or 1,000,000 cycles if not provided (use math.inf if you don't fear infinite loop solutions).
            debug: Print an updating view of the solution while running. See DebugOptions.
        '''
        # TODO: Running the solution should not meaningfully modify it. Namely, need to reset reactor.molecules
        #       and waldo.position/waldo.direction before each run, or otherwise prevent these from persisting.
        #       Otherwise a solution will only be safely runnable once.

        # TODO: Should re-check for component/pipe collisions every time we run it; run() should be a source of truth,
        #       and shouldn't be confoundable by modifying the solution after the ctor validations

        # Set the maximum runtime if unspecified to ensure a broken solution can't infinite loop forever
        if max_cycles is None:
            if self.expected_score is not None:
                max_cycles = 2 * self.expected_score.cycles
            else:
                max_cycles = self.DEFAULT_MAX_CYCLES

        # Default debug view to the reactor's interior in research levels
        if debug and self.level['type'].startswith('research'):
            debug.reactor = 0

        self.validate_components()
        reactors = list(self.reactors)
        outputs = list(self.outputs)

        # If we are continuing running from a pause, start with a movement phase
        if any(waldo.instr_map[waldo.position][1].type == InstructionType.PAUSE
               for reactor in reactors
               for waldo in reactor.waldos
               if waldo.position in waldo.instr_map):
            self.cycle_movement()

        # Run the level
        try:
            # In debug mode the cursor can annoyingly flicker into the middle of the printed output; hide it
            if debug:
                cursor.hide()

            while self.cycle < max_cycles + 1:
                self.cycle += 1

                # Execute instant actions (entity inputs/outputs, waldo instructions)
                for component in self.components:
                    if component.do_instant_actions(self.cycle):
                        # Outputs return true the first time they reach their target count; whenever one does, check
                        # all the others for completion (we can't just count completions since 0/0 outputs should
                        # win even if never triggered). Not checking until an output completes also matches the expected
                        # behaviour of puzzles where all outputs are 0/0
                        if all(output.current_count >= output.target_count for output in outputs):
                            # TODO: Update solution expected score? That would match the game's behavior, but makes the validator
                            #       potentially misleading. Maybe run() and validate() should be the same thing.
                            # -1 looks weird but seems provably right based on output vs pause comparison
                            return Score(self.cycle - 1, len(reactors), self.symbols)

                if debug and self.cycle >= debug.cycle:
                    self.debug_print(duration=0.5 / debug.speed, reactor_idx=debug.reactor,
                                     show_instructions=debug.show_instructions)

                self.cycle_movement()

                if debug and self.cycle >= debug.cycle:
                    self.debug_print(duration=0.5 / debug.speed, reactor_idx=debug.reactor, flash_features=False,
                                     show_instructions=debug.show_instructions)

            raise TimeoutError(f"Solution exceeded {max_cycles} cycles, probably infinite looping?")
        except KeyboardInterrupt:
            # Don't persist the last debug print on keyboard interrupt since it probably happened during debug_print's
            # sleep (while the printout is still displayed)
            # We can do this by setting debug to None before the finally block, we just have to also fix the cursor here
            if debug:
                debug = None
                cursor.show()
            raise
        except Exception as e:
            # Mention the solution description and cycle number on error via a chained exception of the same type
            raise type(e)(f"{self.description}: Cycle {self.cycle}: {e}") from e
        finally:
            # Persist the last debug printout
            if debug:
                if debug.reactor is None:
                    rich.print(str(self))
                else:
                    rich.print(reactors[debug.reactor].__str__(show_instructions=debug.show_instructions))
                    rich.print(f'Cycle: {self.cycle}\n')  # Extra newline for readability of any traceback below it

                # Restore the cursor
                cursor.show()

    def validate(self, max_cycles=None, verbose=False, debug=False):
        '''Run this solution and assert that the score matches the expected score from its metadata.'''
        if self.expected_score is None:
            raise ValueError("validate() requires a valid expected score in the first solution line (currently 0-0-0);"
                             + " please update it or use run() instead.")

        if verbose and self.level_name != self.level.name:
            print(f"Warning: Validating solution against level {repr(self.level.name)} that was originally"
                  + f" constructed for level {repr(self.level_name)}.")

        if max_cycles is not None and self.expected_score.cycles > max_cycles:
            raise ValueError(f"{self.description}: Cannot validate; expected cycles > max cycles ({max_cycles})")

        score = self.run(max_cycles=max_cycles, debug=debug)

        if score != self.expected_score:
            raise ScoreError(f"{self.description}: Expected score {self.expected_score} but got {score}")

        if verbose:
            print(f"Validated {self.description}")
