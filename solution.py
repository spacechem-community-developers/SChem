#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import copy
from itertools import product
import time

from spacechem.components import COMPONENT_SHAPES, Pipe, Input, Output, Reactor, Recycler, StorageTank
from spacechem.exceptions import RunSuccess
from spacechem.grid import Direction, Position
from spacechem.level import OVERWORLD_COLS, OVERWORLD_ROWS, TERRAIN_MAPS


class Score(namedtuple("Score", ('cycles', 'reactors', 'symbols'))):
    '''Immutable class representing a SpaceChem solution score.'''
    __slots__ = ()

    def __str__(self):
        '''Convert to the format used in solution metadata.'''
        return '-'.join(str(i) for i in self)


class Solution:
    '''Class for constructing and running game entities from a given level object and solution code.'''
    __slots__ = ('level_name', 'author', 'expected_score', 'name',
                 'level', 'components')

    @classmethod
    def get_level_name(cls, soln_export_str):
        '''Helper to extract the level name from a given solution name. Should be used to ensure the correct level is
        being passed to Solution.__init__ in case of ambiguities.
        '''
        soln_metadata_str = soln_export_str.strip().split('\n', maxsplit=1)[0]
        assert soln_metadata_str.startswith('SOLUTION:'), "Missing SOLUTION line"

        fields = soln_metadata_str.split(',', maxsplit=3)  # maxsplit since solution name may include commas
        assert len(fields) >= 3, 'SOLUTION line missing fields'

        return fields[0][len('SOLUTION:'):]

    def __init__(self, level, soln_export_str=None):
        self.level = level
        self.level_name = level['name']  # Note that the level name is vestigial; a solution can run in any compatible level
        self.name = None
        self.author = 'Unknown'
        self.expected_score = '0-0-0'

        # Set up the level terrain so we can look up input/output positions and any blocked terrain
        if level['type'] == 'production':
            # Main game levels have unique terrain which we have to hardcode D:
            # We can at least avoid name collisions if we deliberately don't add the terrain field to their JSONs
            # TODO: This is a bit dangerous; the game auto-adds a terrain field if it ever gets its hands on the json
            terrain_id = level['name'] if ('terrain' not in level
                                           and level['name'] in TERRAIN_MAPS) else level['terrain']
        else:
            terrain_id = 'research'  # BIG HACKS

        # Add level-defined entities - including any fixed-position reactors/pipes
        # Store components by posn for now, for convenience. We'll re-order and store them in self.components later
        posn_to_component = {}  # Note that components are referenced by their top-left corner posn

        # Inputs
        input_zone_types = ('random-input-zones', 'fixed-input-zones') if self.level['type'] == 'production' else ('input-zones',)
        input_rate = 10 if self.level['type'] == 'production' else 1  # TODO Support CE variable input rates
        for input_zone_type in input_zone_types:
            for i, input_dict in self.level[input_zone_type].items():
                i = int(i)
                # For some reason production fixed inputs are defined less flexibly than in researches
                # TODO: Input class should probably accept raw molecule string instead of having to stuff it into a dict
                #       One does wonder why Zach separated random from fixed inputs in productions but not researches...
                if isinstance(input_dict, str):
                    # Convert from molecule string to input zone dict format
                    input_dict = {'inputs': [{'molecule': input_dict, 'count': 12}]}

                # Due to a bug in SpaceChem, the random-input-zones may end up populated with an empty input
                # entry. Skip it if that's the case
                # TODO: Make CE issue on github for this
                if not input_dict['inputs']:
                    continue

                component_type, component_posn = TERRAIN_MAPS[terrain_id][input_zone_type][i]
                posn_to_component[component_posn] = Input(component_type=component_type, posn=component_posn,
                                                          input_dict=input_dict, input_rate=input_rate)

        # Outputs
        for i, output_dict in self.level['output-zones'].items():
            i = int(i)

            # Zach pls
            if self.level['type'] == 'production':
                output_dict = copy.deepcopy(output_dict)  # To avoid mutating the level
                output_dict['count'] *= 4

            component_type, component_posn = TERRAIN_MAPS[terrain_id]['output-zones'][i]
            posn_to_component[component_posn] = Output(component_type=component_type, posn=component_posn,
                                                       output_dict=output_dict)

        # Recycler
        if self.level['type'] == 'production' and self.level['has-recycler']:
            component_type, component_posn = TERRAIN_MAPS[terrain_id]['recycler']
            posn_to_component[component_posn] = Recycler(component_type=component_type, posn=component_posn)

        # TODO: Preset reactors - note that research levels behave like a production with a single preset reactor

        # Add solution-defined entities and update preset level components (e.g. pipe inputs, preset reactor contents)
        if soln_export_str is not None:
            # Parse solution metadata from the first line
            soln_metadata_str, components_str = soln_export_str.strip().split('\n', maxsplit=1)
            assert soln_metadata_str.startswith('SOLUTION:'), "Missing SOLUTION line"

            fields = soln_metadata_str.split(',', maxsplit=3)  # maxsplit since solution name may include commas
            assert len(fields) >= 3, 'SOLUTION line missing fields'

            self.level_name = fields[0][len('SOLUTION:'):]
            self.author = fields[1]
            # The game stores unsolved solutions as '0-0-0'
            self.expected_score = Score(*(int(i) for i in fields[2].split('-'))) if fields[2] != '0-0-0' else None
            self.name = fields[3] if len(fields) == 4 else None  # Optional field


            # TODO: Disallow mutating pipes in research levels or on preset reactors, and preset input pipes
            for component_str in ('COMPONENT:' + s for s in components_str.split('COMPONENT:') if s):
                component_metadata = component_str.split('\n', maxsplit=1)[0]
                fields = component_metadata.split(',')
                component_type = fields[0].strip('COMPONENT:').strip("'")
                component_posn = Position(int(fields[1]), int(fields[2]))

                # Check that this component is either an existing one added by the level or is a new component that the
                # level allows
                if component_posn in posn_to_component:
                    cur_component = posn_to_component[component_posn]
                    assert component_type == cur_component.type, \
                        (f'Built-in draggable of type "{cur_component.type}" cannot be overwritten with draggable'
                         + f' of type "{component_type}"')

                    # Generate the new component and update or overwrite the existing component
                    if isinstance(cur_component, Reactor):
                        if self.level['type'].startswith('research'):
                            new_component = Reactor.from_export_str(component_str, features_dict=self.level.dict)
                        else:
                            # In this case, the constructor will self-verify the features based on its component ID
                            new_component = Reactor.from_export_str(component_str)

                        # Reactor contents can be overwritten but not their pipes (any pipe changes are silently ignored)
                        new_component.out_pipes = cur_component.out_pipes

                        # Overwrite the existing component
                        posn_to_component[component_posn] = new_component
                    elif isinstance(cur_component, Input):
                        # TODO: For levels like e.g. PseudoEthyne, input pipes shouldn't be overwritable
                        # Drop the COMPONENT line and expect the remaining lines to construct a pipe
                        cur_component.out_pipe = Pipe.from_export_str(component_str[component_str.find('\n') + 1:])
                    else:
                        raise Exception(f"Unexpected modification to immutable level component {component_type}")
                else:
                    # Component is new; create and add it
                    if 'reactor' in component_type.split('-'):
                        if self.level['type'].startswith('research'):
                            posn_to_component[component_posn] = Reactor.from_export_str(component_str,
                                                                                        features_dict=self.level.dict)
                        else:
                            # Ensure this is a legal reactor type for the level (e.g. drag-starter-reactor -> has-starter)
                            assert f'has-{component_type.split("-")[1]}' in self.level, f"Unknown reactor type {component_type}"
                            assert self.level[f'has-{component_type.split("-")[1]}'], f"Illegal reactor type {component_type}"

                            # In this case, the constructor will self-verify the features based on its component type
                            posn_to_component[component_posn] = Reactor.from_export_str(component_str)
                    # TODO: Storage tanks are only available in defense levels
                    elif component_type == 'drag-storage-tank':
                        if self.level['type'] != 'defense':
                            raise Exception(f"Component {component_type} cannot be used outside Defense levels")
                        posn_to_component[component_posn] = StorageTank.from_export_str(component_str)
                    elif 'input' in component_type.split('-'):
                        raise Exception(f"Could not find level input at position {component_posn}.")
                    else:
                        raise Exception(f"Solution places unexpected component {component_type}")

        # Now that we're done updating components, check that all components/pipes are validly placed
        # TODO: Should probably just be a method validating self.components, and also called at start of run()
        blocked_posns = set()
        # Tracks which directions pipes may cross through each other, and implicitly also which cells contain pipes
        pipe_posns_to_dirns = {}

        # Add impassable terrain
        if self.level['type'] == 'production':
            blocked_posns.update(TERRAIN_MAPS[terrain_id]['obstructed'])

        # Check for component/pipe collisions
        for component in posn_to_component.values():
            # Add this component's cells to the blacklist
            if 'reactor' in component.type.split('-'):
                component_dimensions = COMPONENT_SHAPES['reactor']  # TODO: this is hacky but I'm lazy
            else:
                component_dimensions = COMPONENT_SHAPES[component.type]

            # Check for collisions and add this component's main body (non-pipe) posns to the blacklist
            component_posns = set(product(range(component.posn[0], component.posn[0] + component_dimensions[0]),
                                          range(component.posn[1], component.posn[1] + component_dimensions[1])))
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
                # Note: this code would break if any component had 3 output pipes but none do
                assert pipe.posns[0] == (component_dimensions[0], ((component_dimensions[1] - 1) // 2) + i), \
                    f"Pipe {i} is not connected to parent component {component.type} at {component.posn}"

                # Ensure this pipe doesn't collide with a component or terrain
                # Recall that pipe posns are defined relative to their parent component's posn
                cur_pipe_posns = set((component.posn[0] + pipe_posn[0], component.posn[1] + pipe_posn[1])
                                     for pipe_posn in pipe.posns)
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
                for prev, cur, next_ in zip([pipe.posns[0] + Direction.LEFT] + pipe.posns[:-1],
                                            pipe.posns,
                                            pipe.posns[1:] + [pipe.posns[-1]]):
                    real_posn = component.posn + cur

                    if real_posn not in pipe_posns_to_dirns:
                        pipe_posns_to_dirns[real_posn] = set()

                    # Pipe is not vertical (blocks the horizontal direction)
                    if not (prev.col == cur.col == next_.col):
                        assert Direction.RIGHT not in pipe_posns_to_dirns[real_posn], f"Illegal pipe overlap at {real_posn}"
                        pipe_posns_to_dirns[real_posn].add(Direction.RIGHT)

                    # Pipe is not horizontal (blocks the vertical direction)
                    if not (prev.row == cur.row == next_.row):
                        assert Direction.UP not in pipe_posns_to_dirns[real_posn], f"Illegal pipe overlap at {real_posn}"
                        pipe_posns_to_dirns[real_posn].add(Direction.UP)

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
                if (pipe_posns_to_dirns[pipe_end] == {Direction.UP, Direction.RIGHT}
                        and len(pipe) >= 2 and pipe.posns[-2] != pipe.posns[-1] + Direction.LEFT):
                    continue

                # This is a bit of a hack, but by virtue of output/reactor/etc. shapes, the top-left corner of the
                # component is always 1 up and right of the input pipe, plus one more row per extra input the object
                # accepts (up to 3 for recycler). Blindly check the 3 possible positions for the components that
                # could connect to this pipe
                component_posn = pipe_end + (1, -1)

                if component_posn in posn_to_component:
                    other_component = posn_to_component[component_posn]
                    if (isinstance(other_component, Output)
                            or isinstance(other_component, StorageTank)
                            or isinstance(other_component, Reactor)
                            or isinstance(other_component, Recycler)):
                        other_component.in_pipes[0] = pipe
                    continue

                component_posn = pipe_end + (1, -2)
                if component_posn in posn_to_component:
                    other_component = posn_to_component[component_posn]
                    if ((isinstance(other_component, Reactor) and other_component.type != 'drag-disassemby-reactor')
                            or isinstance(other_component, Recycler)):
                        other_component.in_pipes[1] = pipe
                    continue

                component_posn = pipe_end + (1, -3)
                if component_posn in posn_to_component:
                    other_component = posn_to_component[component_posn]
                    if isinstance(other_component, Recycler):
                        other_component.in_pipes[2] = pipe

    def export_str(self):
        # Solution metadata
        export_str = f"SOLUTION:{self.level['name']},{self.author},{self.expected_score}"
        if self.name is not None:
            export_str += f',{self.name}'

        # Components
        # Exclude inputs whose pipes are length 1 (unmodified), outputs, and the recycler.
        # I'm probably forgetting another case.
        # TODO: This doesn't cover inputs with preset pipes > 1 long - which also shouldn't be included
        for component in self.components:
            if not (isinstance(component, Output)
                    or (isinstance(component, Input)
                        and len(component.out_pipe) == 1)
                    or isinstance(component, Recycler)):
                export_str += '\n' + component.export_str()

        return export_str

    def __str__(self):
        ''''Return a string representing the overworld of this solution including all components and pipes.'''
        # 2 characters per tile
        grid = [['  ' for _ in range(OVERWORLD_COLS)] for _ in range(OVERWORLD_ROWS)]

        # Display each component with #'s
        for component in self.components:
            if isinstance(component, Reactor):
                dimensions = COMPONENT_SHAPES['reactor']
            else:
                dimensions = COMPONENT_SHAPES[component.type]

            # Visual bounds set since components are allowed to hang outside the play area (e.g. in Going Green)
            for posn in product(range(max(0, component.posn.col), min(OVERWORLD_COLS,
                                                                      component.posn.col + dimensions[0])),
                                range(max(0, component.posn.row), min(OVERWORLD_ROWS,
                                                                      component.posn.col + dimensions[1]))):
                grid[posn[1]][posn[0]] = '##'  # row, col

            # Display each pipe with --, |, //, ||, or a . for tiles containing a molecule
            for pipe in component.out_pipes:
                for relative_posn, molecule in zip(pipe.posns, pipe):
                    posn = component.posn + relative_posn
                    grid[posn.row][posn.col] = '==' if molecule is None else ' .'

        result = f"_{OVERWORLD_COLS * '__'}_\n"
        for row in grid:
            result += f"|{''.join(row)}|\n"
        result += f"‾{OVERWORLD_COLS * '‾‾'}‾\n"

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
        if self.level['type'] == 'production':
            assert sum(1 if isinstance(component, Reactor) else 0
                       for component in self.components) <= self.level['max-reactors'], "Reactor limit exceeded"

    def run(self, max_cycles=1000000, debug=False):
        '''Run this solution, returning a score tuple or else raising an exception if the level was not solved.'''
        # TODO: Running the solution should not meaningfully modify it. Namely, need to reset reactor.molecules
        #       and waldo.position/waldo.direction before each run, or otherwise prevent these from persisting.
        #       Otherwise a solution will only be safely runnable once.

        # TODO: Should re-check for component/pipe collisions every time we run it; run() should be a source of truth,
        #       and shouldn't be confoundable by modifying the solution after the ctor validations

        self.validate_components()

        reactors = [component for component in self.components if isinstance(component, Reactor)]

        # Set the maximum runtime to ensure a broken solution can't infinite loop forever
        if self.expected_score is not None:
            max_cycles = 2 * self.expected_score.cycles

        # Run the level
        symbols = sum(sum(len(waldo) for waldo in component.waldos)
                      for component in self.components
                      if hasattr(component, 'waldos'))  # hacky but saves on using a counter or reactor list in the ctor
        cycle = 0
        completed_outputs = 0
        try:
            while cycle < max_cycles:
                # Move molecules/waldos
                for component in self.components:
                    component.move_contents()

                if debug and cycle >= debug.cycle:
                    if debug.reactor is None:
                        self.debug_print(cycle, duration=0.0625)
                    else:
                        reactors[debug.reactor].debug_print(cycle, duration=0.0625)

                # Execute instant actions (entity inputs/outputs, waldo instructions)
                for component in self.components:
                    if component.do_instant_actions(cycle):
                        # Outputs return true the first time they reach their target count; count these occurrences and
                        # raise success when they've all completed
                        completed_outputs += 1
                        if completed_outputs == len(self.level['output-zones']):
                            raise RunSuccess()

                if debug and cycle >= debug.cycle:
                    if debug.reactor is None:
                        self.debug_print(cycle, duration=0.0625)
                    else:
                        reactors[debug.reactor].debug_print(cycle, duration=0.0625)

                cycle += 1

            raise TimeoutError(f"Solution exceeded {max_cycles} cycles, probably infinite looping?")
        except RunSuccess:
            # TODO: Update solution expected score? That would match the game's behavior, but makes the validator
            #       potentially misleading. Maybe run() and validate() should be the same thing.
            # TODO: The cycle + 1 here is a hack since we're inconsistent with SC's displayed count. Given that
            #       waldos move one tile before inputs populate their pipe (and the cycle is already displayed as 1
            #       while they do that first move), I think the only way to match SC exactly is to do instant actions
            #       before move actions, but to have inputs trigger when `(cycle - 1) % rate == 0` instead of
            #       `cycle % rate == 0`, so that they don't input on cycle 0 (before the waldos have moved).
            #       For now putting the +1 here looks less awkward...
            return Score(cycle + 1, len(reactors), symbols)
        finally:
            # Persist the last debug printout
            if debug:
                if debug.reactor is None:
                    print(str(self))
                else:
                    print(str(reactors[debug.reactor]))

    def validate(self, verbose=True, debug=False):
        '''Run this solution and assert that the resulting score matches the expected score that was included in its
        solution code.
        '''
        if self.level_name != self.level.name:
            print(f"Warning: Validating solution against level {repr(self.level.name)} that was originally"
                  + f" constructed for level {repr(self.level_name)}.")

        score = self.run(debug=debug)
        assert score == self.expected_score, (f"Expected score {'-'.join(str(x) for x in self.expected_score)}"
                                              f" but got {'-'.join(str(x) for x in score)}")
        if verbose:
            out_str = f"Validated [{self.level.name}] {score}"
            if self.name is not None:
                out_str += f' "{self.name}"'
            out_str += f" by {self.author}"

            print(out_str)
