#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import copy
from itertools import product
import time

from .components import COMPONENT_SHAPES, Pipe, Component, Input, Output, Reactor, Recycler, DisabledOutput
from .exceptions import ScoreError
from .grid import *
from .level import OVERWORLD_COLS, OVERWORLD_ROWS
from .terrains import terrains, MAX_TERRAIN_INT


class Score(namedtuple("Score", ('cycles', 'reactors', 'symbols'))):
    '''Immutable class representing a SpaceChem solution score.'''
    __slots__ = ()

    def __str__(self):
        '''Convert to the format used in solution metadata.'''
        return '-'.join(str(i) for i in self)

    @classmethod
    def is_score_str(cls, s):
        '''Return True if the given string is formatted like a Spacechem score, e.g. 45-1-14. 0-0-0 is included, but
        note that it is used to denote incomplete solutions.
        '''
        parts = s.split('-')
        return len(parts) == 3 and all(part.isdigit() for part in parts)  # Thinks 045-1-14 is a score but whatever

    @classmethod
    def from_str(cls, s):
        return cls(*(int(x) for x in s.split('-')))


class Solution:
    '''Class for constructing and running game entities from a given level object and solution code.'''
    __slots__ = ('level_name', 'author', 'expected_score', 'name',
                 'level', 'components')

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
        s = s.strip().split('\n', maxsplit=1)[0].strip()
        assert s.startswith('SOLUTION:'), "Given string is not a SpaceChem solution"
        fields = s[len('SOLUTION:'):].split(',')
        assert len(fields) >= 3, "Missing fields in solution metadata"

        # Starting from the third CSV value, look for a score-like string and assume it is the expected score
        score_field_idx = next((i for i in range(2, len(fields)) if Score.is_score_str(fields[i])),
                               None)
        assert score_field_idx is not None, "Solution metadata missing expected score"

        level_name = ','.join(fields[:score_field_idx - 1])
        author_name = fields[score_field_idx - 1]
        # The game stores unsolved solutions as '0-0-0'

        expected_score = Score.from_str(fields[score_field_idx]) if fields[score_field_idx] != '0-0-0' else None
        soln_name = ','.join(fields[score_field_idx + 1:]) if len(fields) > score_field_idx + 1 else None
        # Game single-quotes solution names if they contain a comma, strip this
        if soln_name is not None and ',' in soln_name and soln_name[0] == soln_name[-1] == "'":
            soln_name = soln_name[1:-1]

        return level_name, author_name, expected_score, soln_name

    @classmethod
    def describe(cls, level_name, author, expected_score, soln_name):
        """Return a pretty-printed string describing the given solution metadata."""
        soln_descr = f"[{level_name}] {expected_score}"
        if soln_name is not None:
            soln_descr += f' "{soln_name}"'
        soln_descr += f" by {author}"

        return soln_descr

    @classmethod
    def split_solutions(cls, soln_str):
        """Given a string potentially containing multiple solutions, return an iterator of them."""
        soln_str = soln_str.strip()
        assert soln_str.startswith('SOLUTION:'), "Given text is not a SpaceChem solution"

        # Split with newline prefix in case some fucker names themselves SOLUTION:
        return (f'SOLUTION:{s}'.strip() for s in ('\n' + soln_str).split('\nSOLUTION:')[1:])

    def __init__(self, level, soln_export_str=None):
        self.level = level
        self.level_name = level['name']  # Note that the level name is vestigial; a solution can run in any compatible level
        self.name = None
        self.author = 'Unknown'
        self.expected_score = None

        # Set up the level terrain so we can look up input/output positions and any blocked terrain
        if level['type'].startswith('production'):
            # Main game levels have unique terrain which we have to hardcode D:
            # We can at least avoid name collisions if we deliberately don't add the terrain field to their JSONs
            # TODO: This is a bit dangerous; the game auto-adds a terrain field if it ever gets its hands on the json
            terrain_id = level['name'] if ('terrain' not in level
                                           and level['name'] in terrains) else level['terrain']

            # SC caps oversized terrain IDs
            if isinstance(terrain_id, int) and terrain_id > MAX_TERRAIN_INT:
                terrain_id = MAX_TERRAIN_INT
        else:
            terrain_id = 'research'  # BIG HACKS

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
            raise ValueError(f"Unknown level type {self.level['type']}")

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
                    if not input_dict['molecules']:
                        continue

                    new_component = Input(input_dict=input_dict, is_research=self.level['type'].startswith('research'))
                    posn_to_component[new_component.posn] = new_component

        # Outputs
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
            # Parse solution metadata from the first line
            soln_metadata_str, *split_remainder = soln_export_str.strip().split('\n', maxsplit=1)
            components_str = '' if not split_remainder else split_remainder[0]
            self.level_name, self.author, self.expected_score, self.name = self.parse_metadata(soln_metadata_str)

            if components_str:
                assert components_str.startswith('COMPONENT:'), "Unexpected data on line 1"

            soln_defined_component_posns = set()  # Used to check that the solution doesn't doubly-define a component

            for component_str in ('COMPONENT:' + s for s in components_str.split('COMPONENT:')[1:]):
                component_type, component_posn = Component.parse_metadata(component_str)

                # Ensure this component hasn't already been created/updated by the solution
                if component_posn in soln_defined_component_posns:
                    raise ValueError(f"Solution defines component at {component_posn} twice")
                soln_defined_component_posns.add(component_posn)

                # Create a raw instance of the component if it doesn't exist yet
                if not component_posn in posn_to_component:
                    # Ensure this is a legal component type for the level (e.g. drag-starter-reactor -> has-starter)
                    reactor_type_flag = f'has-{component_type.split("-")[1]}'
                    assert ((reactor_type_flag in self.level and self.level[reactor_type_flag])
                            # Misleadingly, allowed-reactor-types includes storage tanks
                            or ('allowed-reactor-types' in self.level
                                and component_type in self.level['allowed-reactor-types'])), \
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
                posn_to_component[component_posn].update_from_export_str(component_str, update_pipes=update_pipes)

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
                    if not (prev.col == cur.col == next_.col):
                        assert RIGHT not in pipe_posns_to_dirns[real_posn], f"Illegal pipe overlap at {real_posn}"
                        pipe_posns_to_dirns[real_posn].add(RIGHT)

                    # Pipe is not horizontal (blocks the vertical direction)
                    if not (prev.row == cur.row == next_.row):
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
        export_str = f"SOLUTION:{self.level['name']},{self.author},{self.expected_score}"
        if self.name is not None:
            # SC expects single-quotes surrounding solution names that contain a comma
            if ',' in self.name:
                export_str += f",'{self.name}'"
            else:
                export_str += f",{self.name}"

        # Components
        # Exclude inputs whose pipes are length 1 (unmodified), outputs, and the recycler.
        # I'm probably forgetting another case.
        # TODO: This doesn't cover inputs with preset pipes > 1 long - which also shouldn't be included
        #       Really it's probably just every unmodified preset component
        for component in self.components:
            if not (isinstance(component, Output)
                    or (isinstance(component, Input)
                        and len(component.out_pipe) == 1)
                    or isinstance(component, Recycler)):
                export_str += '\n' + component.export_str()

        return export_str

    def __str__(self):
        ''''Return a string representing the overworld of this solution including all components and pipes.'''
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
        result += f"‾{OVERWORLD_COLS * '‾'}‾\n"

        return result

    def debug_print(self, cycle, duration=0.5):
        # Print the current state
        output = str(self)
        output += f'\nCycle: {cycle}'
        print(output)  # Could use end='' but that makes keyboard interrupt output ugly

        time.sleep(duration)

        # Use the ANSI escape code for moving to the start of the previous line to reset the terminal cursor
        cursor_reset = (output.count('\n') + 1) * "\033[F"  # +1 for the implicit newline print() appends
        print(cursor_reset, end='')

    def validate_components(self):
        '''Validate that this solution is legal (whether or not it can run to completion).
        Called at the start of each run to ensure that no trickery was done on this object post-construction.
        '''
        # Make sure the reactor limit has been respected
        if self.level['type'].startswith('production'):
            assert sum(1 if isinstance(component, Reactor) else 0
                       for component in self.components) <= self.level['max-reactors'], "Reactor limit exceeded"

    def run(self, max_cycles=None, debug=False):
        '''Run this solution, returning a score tuple or else raising an exception if the level was not solved.'''
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

        # Run the level
        symbols = sum(sum(len(waldo) for waldo in component.waldos)
                      for component in self.components
                      if hasattr(component, 'waldos'))  # hacky but saves on using a counter or reactor list in the ctor
        cycle = 0
        num_outputs = len(list(self.outputs))
        completed_outputs = 0

        try:
            while cycle < max_cycles:
                # Move molecules/waldos
                for component in self.components:
                    try:
                        component.move_contents()
                    except Exception as e:
                        # Mention the originating reactor in errors when possible
                        if isinstance(component, Reactor):
                            for i, reactor in enumerate(reactors):
                                if component is reactor:
                                    # Mention the reactor index via a chained exception of the same type
                                    raise type(e)(f"Reactor {i}: {e}") from e
                        raise e

                if debug and cycle >= debug.cycle:
                    if debug.reactor is None:
                        self.debug_print(cycle, duration=0.0625 / debug.speed)
                    else:
                        reactors[debug.reactor].debug_print(cycle, duration=0.0625 / debug.speed)

                # Execute instant actions (entity inputs/outputs, waldo instructions)
                for component in self.components:
                    if component.do_instant_actions(cycle):
                        # Outputs return true the first time they reach their target count; count these occurrences and
                        # end when they've all completed
                        completed_outputs += 1
                        if completed_outputs == num_outputs:
                            # TODO: Update solution expected score? That would match the game's behavior, but makes the validator
                            #       potentially misleading. Maybe run() and validate() should be the same thing.
                            # TODO: The cycle + 1 here is a hack since we're inconsistent with SC's displayed count. Given that
                            #       waldos move one tile before inputs populate their pipe (and the cycle is already displayed as 1
                            #       while they do that first move), I think the only way to match SC exactly is to do instant actions
                            #       before move actions, but to have inputs trigger when `(cycle - 1) % rate == 0` instead of
                            #       `cycle % rate == 0`, so that they don't input on cycle 0 (before the waldos have moved).
                            #       For now putting the +1 here looks less awkward...
                            return Score(cycle + 1, len(reactors), symbols)

                if debug and cycle >= debug.cycle:
                    if debug.reactor is None:
                        self.debug_print(cycle, duration=0.0625 / debug.speed)
                    else:
                        reactors[debug.reactor].debug_print(cycle, duration=0.0625 / debug.speed)

                cycle += 1

            raise TimeoutError(f"Solution exceeded {max_cycles} cycles, probably infinite looping?")
        except Exception as e:
            # Mention the cycle number on error via a chained exception of the same type
            raise type(e)(f"Cycle {cycle}: {e}") from e
        finally:
            # Persist the last debug printout
            if debug:
                if debug.reactor is None:
                    print(str(self))
                else:
                    print(str(reactors[debug.reactor]))

    def validate(self, verbose=False, debug=False):
        '''Run this solution and assert that the resulting score matches the expected score that was included in its
        solution code.
        '''
        if self.expected_score is None:
            raise ValueError("validate() requires a valid expected score in the first solution line (currently 0-0-0);"
                             + " please update it or use run() instead.")

        if verbose and self.level_name != self.level.name:
            print(f"Warning: Validating solution against level {repr(self.level.name)} that was originally"
                  + f" constructed for level {repr(self.level_name)}.")

        score = self.run(debug=debug)

        if score != self.expected_score:
            raise ScoreError(f"Expected score {self.expected_score} but got {score}")

        if verbose:
            print(f"Validated {self.describe(self.level.name, self.author, score, self.name)}")
