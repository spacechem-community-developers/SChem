#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import namedtuple
import copy
from dataclasses import dataclass
from itertools import product
import platform
import time
from typing import Optional, Union
import sys

# These modules are only used for debug mode's pretty-printing, so make them optional
try:
    import cursor
    import rich
except ImportError:
    pass
# TODO: Hoping to avoid default number-highlighting via Console(highlight=False).print but it seems to break ANSI resets

from .components import Component, Input, RandomInput, Output, Weapon, Reactor, Recycler, DisabledOutput, \
                        TeleporterInput, TeleporterOutput, Boss, NuclearMissile, DEFAULT_RESEARCH_REACTOR_TYPE
from .waldo import InstructionType
from .exceptions import SolutionImportError, ScoreError, InfiniteLoopError
from .grid import *
from .level import Level, OVERWORLD_COLS, OVERWORLD_ROWS
from .levels import levels as built_in_levels, unsupported_defense_names, resnet_ids
from .precognition import is_precognitive
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
    """Immutable class representing a SpaceChem solution score."""
    INCOMPLETE_STR = '0-0-0'
    __slots__ = ()

    def __str__(self):
        """Convert to the format used in solution metadata."""
        return '-'.join(str(i) for i in self)

    @classmethod
    def is_score_str(cls, s):
        """Return True if the given string is formatted like a Spacechem score, e.g. 45-1-14.
        Incomplete score formats (0-0-0, and Incomplete-r-s from the old SCT) are included.
        """
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
        if s == cls.INCOMPLETE_STR or (parts[0] == 'Incomplete' and all(part.isdigit() for part in parts[1:])):
            return None

        return cls(*(int(x) for x in s.split('-')))


# Helper for tracking solution states, in particular the tree of states that arises when random inputs are involved
class StateTreeSegment:
    """A single consecutive sequence of states, uninfluenced by randomness.
    A solution's tree of states consists of junctions caused by random inputs, connected by these segments. A non-random
    level thus contains only a single segment.
    """
    __slots__ = 'cycles', 'output_cycles', 'next_node'

    def __init__(self, num_outputs):
        self.cycles = 0  # Number of cycles (states) in this segment
        # Cycles (relative to the start of this segment) during which a (successful) output occurred, for each output
        # component in the level
        self.output_cycles = [[] for _ in range(num_outputs)]
        self.next_node = None  # The next branch node caused by a random input


class Solution:
    """Class for constructing and running game entities from a given level object and solution code."""
    __slots__ = ('level', 'expected_score', 'author', 'name',
                 'components', 'boss', 'cycle',
                 # Hashing-related attributes
                 '_prior_states', '_state_tree_segments', '_state_tree_nodes',
                 '_cur_state_tree_segment_idx', '_cycles_to_new_state',
                 '_random_inputs', '_random_input_copies')

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

    @classmethod
    def parse_metadata(cls, s):
        """Given a solution export string or its SOLUTION line, return the level name, author, expected score,
        and solution name (or None if either of the latter are not set).
        Due to the use of comma-separated values in export lines, commas in the level or author name may cause
        errors in the parsing. To combat this, it is assumed that author names contain no commas, and that the first
        comma-separated string (after the second comma) which is formatted like a score (e.g. 45-1-14) is the score
        field. This is sufficient to uniquely identify the other fields.
        As a consequence, if either the author name contains a comma or the author/level names look like scores,
        it is possible for some fields to get mis-parsed. This is still a little better than SC which gets
        completely messed up by any commas in the level name.
        """
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

    @property
    def description(self):
        return self.describe(self.level.name, self.author, self.expected_score, self.name)

    def __init__(self, solution_str: Optional[str], level: Optional[Union[Level, str]] = None):
        """Load a solution string as exported by SC, instantiating both level-defined and solution-defined components.

        To create an empty solution for the given level, set solution_str to None.

        Args:
            solution_str: Solution string as exported by SpaceChem Community Edition (steam beta).
            level: The level to load the solution for. Either a Level object, or string of the puzzle export code.

                If level is not provided, the level name from the solution metadata is used to search for a matching
                official level(s).
                When multiple official levels match the solution metadata's level name, attempt to load into each in
                turn until one with matching features is found.
                Note that all official levels with duplicate titles have incompatible level features (bonder counts
                etc.), so it is guaranteed that the correct level will be loaded if possible.
        """
        self.cycle = 0

        # Load metadata
        if solution_str is not None:
            try:
                level_name, self.author, self.expected_score, self.name = self.parse_metadata(solution_str)
            except Exception as e:
                raise SolutionImportError(str(e)) from e
        elif level is None:
            raise ValueError("Solution string or level must be provided")
        else:
            self.author, self.expected_score, self.name = 'Unknown', None, None

        if level is None:
            # Determine the built-in game level to use based on the level name in its metadata
            if level_name in built_in_levels:
                # The official levels dict stores lists of level codes in the case of levels sharing a names
                if isinstance(built_in_levels[level_name], str):
                    level_codes = [built_in_levels[level_name]]
                    matching_resnet_ids = [resnet_ids[level_name] if level_name in resnet_ids else None]
                else:
                    level_codes = built_in_levels[level_name]
                    matching_resnet_ids = (resnet_ids[level_name] if level_name in resnet_ids
                                           else len(built_in_levels[level_name]) * [None])
            elif level_name in unsupported_defense_names:
                raise NotImplementedError(f"Defense level `{level_name}` is unsupported.")
            else:
                raise Exception(f"No known level `{level_name}`.")

            # Use the first level which can be successfully loaded, or raise the first exception if none can
            exceptions = []
            for level_code, resnet_id in zip(level_codes, matching_resnet_ids):
                level = Level(level_code)
                level.resnet_id = resnet_id
                try:
                    self._load(level=level, soln_export_str=solution_str)
                    return
                except SolutionImportError as e:
                    exceptions.append(e)

            raise exceptions[0]
        elif isinstance(level, Level):
            self._load(level=level, soln_export_str=solution_str)
        else:  # Accept raw level code
            self._load(level=Level(level), soln_export_str=solution_str)

    def _load(self, level: Level, soln_export_str=None):
        """Helper to do the bulk of the constructor's work, attempting to load the solution using the given level."""
        self.level = level

        # Set up the level terrain so we can look up input/output positions and any blocked terrain
        if level['type'].startswith('research'):
            terrain_id = 'research'
        elif level['type'].startswith('production') or level['type'] == 'defense':
            # Main game levels have unique terrain which we have to hardcode D:
            # We can at least avoid name collisions if we deliberately don't add the terrain field to their JSONs
            # TODO: This is a bit dangerous; the game auto-adds a terrain field if it ever gets its hands on the json
            terrain_id = level.name if ('terrain' not in level
                                        and level.name in terrains) else level['terrain']

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
        # TODO: Why even bother differentiating? Just parse each type appropriately
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
        elif self.level.type == 'sandbox':
            input_zone_types = ('random-input-zones', 'fixed-input-zones')
            output_zone_type = None
        else:  # Defense (I did these custom so they use whatever input type is handy)
            input_zone_types = ('random-input-zones', 'fixed-input-zones', 'random-input-components')
            output_zone_type = None

        for input_zone_type in input_zone_types:
            if input_zone_type not in self.level:
                continue

            if isinstance(self.level[input_zone_type], dict):
                for i, input_dict in self.level[input_zone_type].items():
                    i = int(i)

                    # For some reason production fixed inputs are defined less flexibly than in researches
                    # TODO: Input class should probably accept raw molecule string instead of having to stuff it into a
                    #       dict. One does wonder why Zach separated random from fixed inputs in productions but not
                    #       researches...
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

        self.boss = Boss(self.level['boss']) if 'boss' in self.level else None

        # Preset components
        if self.level['type'].startswith('research'):
            # Preset one reactor, treating the level dict as its component dict
            reactor_type = self.level['reactor-type'] if 'reactor-type' in self.level else DEFAULT_RESEARCH_REACTOR_TYPE
            new_component = Reactor(self.level.dict, _type=reactor_type, posn=Position(col=2, row=0))
            posn_to_component[new_component.posn] = new_component

            # Add disabled outputs for the unused zones of research levels (crash if given a molecule)
            if not ('has-large-output' in self.level and self.level['has-large-output']):
                for zone in range(2):
                    if zone not in set(int(used_zone) for used_zone in self.level[output_zone_type]):
                        _, component_posn = terrains[terrain_id][output_zone_type][zone]
                        posn_to_component[component_posn] = DisabledOutput(_type='disabled-output', posn=component_posn)
        else:
            # Recycler
            if 'has-recycler' in self.level and self.level['has-recycler']:
                component_type, component_posn = terrains[terrain_id]['recycler']
                posn_to_component[component_posn] = Recycler(_type=component_type, posn=component_posn)

            # Miscellaneous preset components (CE components + weapons)
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

                        # Allow control instructions in boss levels (disallowed-instructions can override this)
                        # Checking boss presence instead of defense level type correctly handles the Collapsar edge case
                        if 'boss' in self.level and 'has-controls' not in self.level:
                            component_dict['has-controls'] = True  # TODO: Stop mutating the level dict

                        new_component = Component(component_dict)

                        if isinstance(new_component, TeleporterInput):
                            # TODO: Bit hacky but since user levels can't use this, I just made sure the
                            #       Teleporter output is defined before its input in "Teleporters"
                            new_component.destination = posn_to_component[Position(col=component_dict['destination-x'],
                                                                                   row=component_dict['destination-y'])]
                        elif isinstance(new_component, Weapon):
                            new_component.boss = self.boss

                        posn_to_component[new_component.posn] = new_component

        try:
            # Add solution-defined components and update preset level components (e.g. inputs' pipes, preset reactor contents)
            if soln_export_str is not None:
                # Strip the SOLUTION: line
                _, *split_remainder = soln_export_str.replace('\r\n', '\n').strip('\n').split('\n', maxsplit=1)
                components_str = '' if not split_remainder else split_remainder[0].strip('\n')

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
                            component_dict.update(self.level['custom-reactors'][i])

                        # As a convenience for defining early game levels without needing custom reactors, propagate
                        # any top-level "disallowed-instructions" value from the level into each reactor
                        if 'disallowed-instructions' in self.level and 'disallowed-instructions' not in component_dict:
                            component_dict['disallowed-instructions'] = self.level['disallowed-instructions']

                        # Allow control instructions in boss levels
                        # Checking boss presence instead of defense level type correctly handles the Collapsar edge case
                        if 'boss' in self.level and 'has-controls' not in self.level:
                            component_dict['has-controls'] = True

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
                        update_pipes = True  # Non-preset components always have modifiable pipes
                    else:
                        # Handle preset reactors with immutable pipes
                        # 'mutable-pipes' is not used by SC, but is added to handle the fact that Ω-Pseudoethyne and
                        # Collapsar disallow mutating a preset reactor's 1-long pipe whereas custom levels allow it.
                        # The custom level codes for Ω-Pseudoethyne and Collapsar are the only ones to use this
                        component = posn_to_component[component_posn]
                        update_pipes = (not self.level['type'].startswith('research')
                                        and ('mutable-pipes' not in self.level
                                             or self.level['mutable-pipes']
                                             or not isinstance(component, Reactor)))

                    # Update the existing component (e.g. its pipes or reactor internals)
                    try:
                        component.update_from_export_str(component_str, update_pipes=update_pipes)
                    except Exception as e:
                        if not self.level['type'].startswith('research'):
                            raise type(e)(f"{component.type} at {component.posn}: {e}") from e

                        raise e

                    # TODO: Ensure the updated component pipes are within the overworld bounds

            # Now that we're done updating components, check that all components/pipes are validly placed
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

                    # Check the column to the right of the pipe for a component to connect to
                    # If we find one, calculate whether the pipe is actually touching one of its input pipe slots
                    # We assume that all components have their input pipes centered, leaning upward in case of imbalance
                    # This assumption holds true for all components except the laser in Don't Fear the
                    # Reaper, which is not currently planned to be implemented due to it having random results
                    for i in range(4):
                        other_posn = pipe_end + (1, -i)  # Check the column to the right
                        if other_posn in posn_to_component:
                            other = posn_to_component[other_posn]
                            # Calculate what index pipe this would be based on the component's shape / pipe count
                            pipe_rows_start = ((other.dimensions[1] - len(other.in_pipes)) // 2) + other_posn.row
                            pipe_idx = pipe_end.row - pipe_rows_start

                            if 0 <= pipe_idx < len(other.in_pipes):
                                other.in_pipes[pipe_idx] = pipe

                            break  # Safe since we check bottom up

            self.validate_components()

            # Hashing-related vars
            # TODO: These are probably better off living on some dedicated StateTree class
            self._prior_states = {}  # Set containing hashes of every previous state
            self._state_tree_segments = [StateTreeSegment(num_outputs=len(list(self.outputs)))]
            self._cur_state_tree_segment_idx = 0
            self._state_tree_nodes = []
            self._cycles_to_new_state = 0
        except Exception as e:
            raise SolutionImportError(str(e)) from e

    def __hash__(self):
        # Hash all components. Note that we didn't set outputs to hash their counts, and still include them in the hash
        # so we don't fail to hash the pipes of pass-through outputs
        return hash(tuple(c.hashable_repr(self.cycle) for c in self.components))

    def validate_components(self):
        """Validate that this solution is legal (whether or not it can run to completion)."""
        # Make sure the reactor limit has been respected
        if self.level['type'].startswith('production'):
            assert sum(1 if isinstance(component, Reactor) else 0
                       for component in self.components) <= self.level['max-reactors'], "Reactor limit exceeded"

    def export_str(self) -> str:
        """Re-export this solution. Sorted to ensure uniqueness, so may differ from the initially given export."""
        # Solution metadata
        fields = [self.level.name, self.author,
                  str(self.expected_score) if self.expected_score is not None else Score.INCOMPLETE_STR]
        if self.name is not None:
            fields.append(self.name)

        # Quote field names that contain a comma or start with a quote (and escape their internal quotes)
        export_str = "SOLUTION:" + ','.join(f"""'{s.replace("'", "''")}'""" if s and (',' in s or s[0] == "'") else s
                                            for s in fields)

        # Components
        # Exclude out-pipeless components (other than Reactors, as in the SuperLaserReactor special case), and
        # inputs/teleporter outputs whose pipes are length 1 (unmodified).
        # This is safe as these components are all always preset (whereas e.g. the player might place a storage tank
        # with a length 1 pipe, which we can't exclude).
        # TODO: This doesn't cover inputs with preset pipes > 1 long - which also shouldn't be included
        #       Really it's probably just every unmodified preset component.
        #       It's also improperly excluding output printers, which are player-placeable in sandbox levels.
        #       Probably need a 'preset' property on components(/pipes?) to cover everything without special-casing.
        for component in self.components:
            if ((component.out_pipes or isinstance(component, Reactor))  # Handles SuperLaserReactor special case
                and not (isinstance(component, (Input, TeleporterOutput))
                         and len(component.out_pipe) == 1)):
                export_str += '\n' + component.export_str()

        return export_str

    def __str__(self):
        """Return a string representing the overworld of this solution including all components/pipes, and the cycle."""
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
            if isinstance(component, NuclearMissile):
                for o, output in enumerate(component._outputs):
                    for i, s in enumerate(reversed(str(output.current_count))):
                        grid[component.posn.row + o][component.posn.col + 1 - i] = s
            elif isinstance(component, Output):
                # Display the output count in the component. All outputs are 2x3 so we should have room
                for i, s in enumerate(reversed(str(component.current_count))):
                    grid[component.posn.row + 1][component.posn.col + 1 - i] = s

            # Display each pipe with -, \, /, +, |, or a . for tiles containing a molecule
            for pipe in component.out_pipes:
                last_posn = Position(-1, -1)
                for relative_posn, molecule in zip(pipe.posns, pipe.to_list(self.cycle)):
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
        """Print the currently running solution state then clear it from the terminal.
        Args:
            duration: Seconds before clearing the printout from the screen. Default 0.5.
            reactor_idx: If specified, only print the contents of the reactor with specified index.
            flash_features: Whether to flash activated features, inputs, and outputs.
        """
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
                            raise type(e)(f"Reactor {i}: {e}") from e
                raise e

    def hash_and_check_state(self, debug=False):
        """Hash the current solution state and check if it matches a past state, fast-forwarding cycles when possible.
        Return True if the solution has been fast-forwarded to successful completion.
        """
        # If we previously did a lookahead to the next unexplored branch, we can remember how many cycles
        # we are from an unknown state and skip calculating the hash while we're advancing the solution to that state.
        # This also guarantees that every time the below code notices we just did a random input, we are adding a new
        # segment and not following an already-explored segment
        if self._cycles_to_new_state > 0:
            self._cycles_to_new_state -= 1
            return

        # If the current cycle included a molecule being consumed from a random input's pipe, add a new branch node if
        # the segment we're in doesn't already link to one (i.e. we didn't just fast-forward), and add a segment for the
        # new branch
        # Conveniently, our current pipe implementation already has to track the last pop cycle
        if any(rand_input.out_pipe._last_pop_cycle == self.cycle for rand_input in self._random_inputs):
            last_segment = self._state_tree_segments[self._cur_state_tree_segment_idx]

            if last_segment.next_node is None:  # next_node should only be present if we previously fast-forwarded
                self._state_tree_nodes.append({})
                last_segment.next_node = len(self._state_tree_nodes) - 1

            cur_node = self._state_tree_nodes[last_segment.next_node]

            # Add a new segment, link to it from the node, and set it as the current segment
            self._state_tree_segments.append(StateTreeSegment(num_outputs=len(list(self.outputs))))
            self._cur_state_tree_segment_idx = len(self._state_tree_segments) - 1

            # Create a key representing which random branch we entered, accounting for multiple random inputs
            # potentially having been produced in the same cycle
            # Could construct this for this block's `if`, but we want the check's main case to be as fast as possible
            input_branch_key = tuple(rand_input_copy.get_next_molecule_idx()  # Note that this advances the input copy
                                     if rand_input.out_pipe._last_pop_cycle == self.cycle else None
                                     for rand_input, rand_input_copy in zip(self._random_inputs, self._random_input_copies))
            assert input_branch_key not in cur_node  # Should be guaranteed by our cycles_to_new_state fast-forwarding
            cur_node[input_branch_key] = self._cur_state_tree_segment_idx

        # Update the current segment's cycle and output counts
        cur_segment = self._state_tree_segments[self._cur_state_tree_segment_idx]
        for output_idx, output in enumerate(self.outputs):
            if output.in_pipe is not None and output.in_pipe._last_pop_cycle == self.cycle:
                cur_segment.output_cycles[output_idx].append(cur_segment.cycles)  # Cycle relative to segment start
        cur_segment.cycles += 1

        # Check if we match a previous state
        cur_state = hash(self)
        if cur_state not in self._prior_states:
            # Store the current state and which segment it's in
            self._prior_states[cur_state] = self._cur_state_tree_segment_idx
            return

        # Check which segment we looped back to
        other_segment_idx = self._prior_states[cur_state]

        # Count how many cycles into the matched segment we skipped. We need to know this whether or not it's the same
        # as our current segment
        skipped_cycles = 1  # We always skip at least the cycle leading to the hash we merged with
        for state, segment_idx in self._prior_states.items():
            if segment_idx != other_segment_idx:  # Ignore irrelevant segments
                continue
            elif state == cur_state:
                break

            skipped_cycles += 1

        # If the matched hash is in the current segment of the hash tree, we found a deterministic loop and can
        # safely fast-forward or declare an infinite loop
        if other_segment_idx == self._cur_state_tree_segment_idx:
            # Special case: In all-0-count output levels (e.g. `Head or Tails?`), the below code would accidentally
            # think all outputs were complete and allow infinite loops to 'win'.
            # However note that in that case we can't have ever output before, or we'd have won, so any loop must be an
            # infinite loop
            if all(output.target_count == 0 for output in self.outputs):
                raise InfiniteLoopError("Solution contains an infinite loop.")

            # Figure out how many cycles it will take each output to complete. The max is our winning cycle
            remaining_cycles = 0
            for output_idx, output in enumerate(self.outputs):
                # Ignore the output if it's already complete. At least one must not be complete or we'd have
                # exited while executing this cycle
                remaining_outputs = output.target_count - output.current_count
                if remaining_outputs <= 0:
                    continue

                # Identify the outputs that are contained in the part of the segment we're looping over
                loop_outputs = [c for c in cur_segment.output_cycles[output_idx] if c >= skipped_cycles]
                if not loop_outputs:
                    # If this loop doesn't output any molecules into an incomplete output, we will never win
                    raise InfiniteLoopError("Solution contains an infinite loop.")

                # Note that the last loop can't be a full loop, so we add a -1 to the remaining outputs
                full_loops, outputs_remainder = divmod(remaining_outputs - 1, len(loop_outputs))
                outputs_remainder += 1  # Per the -1 above, we've ensured this is at least 1
                cycles_remainder = 0
                if outputs_remainder != 0:
                    # +1 since it's 0-indexed
                    cycles_remainder = loop_outputs[outputs_remainder - 1] - skipped_cycles + 1
                remaining_cycles = max(remaining_cycles,
                                       full_loops * (cur_segment.cycles - skipped_cycles) + cycles_remainder)
            self.cycle += remaining_cycles

            return True

        # At this point we know the loop wasn't internal to our current segment
        other_segment = self._state_tree_segments[other_segment_idx]

        # 'Extend' the current segment based on the other segment's remaining cycles
        # We will also initialize some incoming_<measure> vars that will be used in the lookahead after this
        incoming_cycles = other_segment.cycles - skipped_cycles
        incoming_outputs = []  # The amount each output will be incremented by in what remains of this segment
        for i in range(len(list(self.outputs))):  # TODO: Stop re-incurring the cost of counting output objects...
            other_outputs = [cur_segment.cycles + c - skipped_cycles
                             for c in other_segment.output_cycles[i]
                             if c >= skipped_cycles]
            incoming_outputs.append(len(other_outputs))
            cur_segment.output_cycles[i].extend(other_outputs)
        cur_segment.cycles += incoming_cycles
        cur_segment.next_node = other_segment.next_node
        incoming_cycles += 1  # I think the above had an off-by-1 error?

        # At this point we must have finished exploring a branch and have at least one loopback; look for another
        # unexplored branch. Since we don't store full copies of the solution, we can't actually advance the current
        # solution state to just before a new unexplored branch, but we can fast-forward the cycle/output counts past
        # any loops we foresee re-encountering and skip hash checks on our way to the new branch.
        # Starting from the matched state, search forward in the state tree, drawing random inputs as needed,
        # and tallying up the future cycle and output counts for each jump.
        # If we find...
        # * that our accumulation of output counts will make us win: stop and step through each state
        #   in the winning segment to figure out exactly what cycle we'll win on.
        # * An unexplored branch: stop searching and exit, keeping note of how many cycles to ignore hashes for
        # * A loop to a node we've visited: Add the loop's cycles/outputs to the solution's real counts,
        #   then reset the tally of future cycles/outputs to those from before the loop, and continue searching.
        segment_idx = self._cur_state_tree_segment_idx
        segment = cur_segment
        # Track the nodes we encounter while hopping through segments, along with how many cycles/outputs away they are
        visit_path = {}  # node_idx: [next_branch_input_key, cycles_until_this_node, outputs_until_this_node]
        while True:
            # Check if the level will be completed by the outputs from all segments forecasted so far
            if all(output.current_count + incoming_outputs[output_idx] >= output.target_count
                   for output_idx, output in enumerate(self.outputs)):
                if debug:
                    print(f'Fast-forwarding to end from cycle {self.cycle}')

                # Figure out exactly which cycle we won on from the segment we just fast-forwarded through
                # (we must have won in the current segment or we'd have caught it last loop).
                # We could check before fast-forwarding but this is more convenient since our first
                # lookahead above may have been only partway through the segment we matched
                for output_idx, output in enumerate(self.outputs):
                    output.current_count += incoming_outputs[output_idx]

                winning_cycle = 0
                for output_idx, output in enumerate(self.outputs):
                    output_diff = output.current_count - output.target_count
                    assert output_diff >= 0  # Sanity check that we actually won
                    if output_diff < len(segment.output_cycles[output_idx]):
                        winning_cycle = max(winning_cycle, segment.output_cycles[output_idx][-output_diff - 1])
                self.cycle += incoming_cycles - (segment.cycles - winning_cycle)

                return True

            # Check the next random branch node
            node_idx = segment.next_node
            node = self._state_tree_nodes[node_idx]
            # If this node is already in our visit path, we found a loop; pre-emptively increase our cycle/outputs,
            # re-jigger the corresponding input component/pipe to omit that part of the random sequence, and remove the
            # loop from our visit history.
            # E.g. our current forecast might be:
            # [---- A ------ B -------- A] where A and B are different states where a random input occurs, in which case
            # we should update to:
            # [---- A] while adding all the cycles and outputs from the removed loop, and removing the future random
            # sequence entries that will be drawn at B and the first A.
            if node_idx in visit_path:
                if debug:
                    print(f"Cycle {self.cycle}: Fast-forwarding a loop from cycles {self.cycle + visit_path[node_idx][1]}"
                          f" to {self.cycle + incoming_cycles}")

                # Add the loop's cycles/outputs pre-emptively; this is safe since we've already checked that we
                # won't win before or during the discovered loop
                self.cycle += incoming_cycles - visit_path[node_idx][1]  # Add loop cycles to our total
                for output_idx, output in enumerate(self.outputs):
                    output.current_count += incoming_outputs[output_idx] - visit_path[node_idx][2][output_idx]

                # Do surgery on the random input(s) and their pipes to remove all molecules that were part of this loop,
                # shuffling later molecules into their existing positions
                def remove_inputs(random_input: RandomInput, indices: set):
                    """Remove a set of input sequence indices from the given random input component and its pipe."""
                    max_idx = max(indices)
                    num_pipe_mols = len(random_input.out_pipe._molecules)

                    # To avoid unnecessary molecule construction, remove inputs not yet in the pipe first
                    restore_indices = []
                    for i in range(num_pipe_mols, 1 + max_idx):
                        mol_idx = random_input.get_next_molecule_idx()
                        if i not in indices:
                            restore_indices.append(mol_idx)
                        else:
                            indices.remove(i)
                    # Restore the indices we weren't supposed to remove (to the front of the input's queue)
                    random_input.forecast_queue.extend(reversed(restore_indices))

                    # The remaining indices should correspond to molecules already in the pipe; sort and remove them
                    # in order, while constructing and adding an equal number of molecules back into the pipe, shuffling
                    # all molecules along so they fill the same positions
                    idx_offset = 0  # As we remove indices, all remaining indices have to be shifted
                    for remove_idx in sorted(indices):
                        remove_idx -= idx_offset
                        # Remove from the molecules queue and add a new molecule from the input.
                        # Due to the nature of our timed queue pipe implementation, this implicitly shuffles the
                        # molecules along to fill each other's positions
                        # Make sure to use right-indexing
                        del random_input.out_pipe._molecules[num_pipe_mols - 1 - remove_idx]  # slow but oh well
                        new_molecule = random_input.molecules[random_input.get_next_molecule_idx()].copy()
                        random_input.out_pipe._molecules.appendleft(new_molecule)

                        idx_offset += 1

                # Identify all the molecules that were part of the loop and remove them
                # We need some helper vars here since not every node will necessarily consume all random inputs,
                # so the indices of each random input's sequence that we have to start removing from may vary
                cur_input_seq_indices = [0 for _ in self._random_inputs]
                input_seq_indices_to_remove = [set() for _ in self._random_inputs]
                in_loop = False  # So we know when to start removing
                for cur_node_idx, (input_key, _, _) in visit_path.items():
                    if cur_node_idx == node_idx:
                        in_loop = True

                    for input_idx, mol_idx in enumerate(input_key):
                        if mol_idx is not None:
                            if in_loop:
                                input_seq_indices_to_remove[input_idx].add(cur_input_seq_indices[input_idx])

                            cur_input_seq_indices[input_idx] += 1

                for rand_input, seq_indices in zip(self._random_inputs, input_seq_indices_to_remove):
                    remove_inputs(rand_input, seq_indices)

                # Reset accumulated cycles/outputs
                incoming_cycles = visit_path[node_idx][1]
                incoming_outputs = list(visit_path[node_idx][2])
                # Reset visit path to before the loop
                new_visit_path = {}
                for k, v in visit_path.items():
                    new_visit_path[k] = v
                    if k == node_idx:
                        break
                visit_path = new_visit_path

            # Examine one branch of this node (there must be at least one) to know which random input(s) we need,
            # then draw new inputs from these
            sample_input_key = next(iter(node))  # [(input1_mol_idx | None), (input2_mol_idx | None), ...]
            new_input_key = tuple(self._random_input_copies[i].get_next_molecule_idx()
                                  if sample_mol_idx is not None else None
                                  for i, sample_mol_idx in enumerate(sample_input_key))
            # In addition to the random input indices predicted, store the cycles and output counts accumulated
            # up to this node, so we can quickly measure loops and reset our counts after finding one
            visit_path[node_idx] = [new_input_key, incoming_cycles, tuple(incoming_outputs)]

            # Check if this branch is already explored or not
            if new_input_key in node:
                # Fast-forward through the explored segment we found, updating our accumulation of cycles/outputs
                segment_idx = node[new_input_key]
                segment = self._state_tree_segments[segment_idx]
                incoming_cycles += segment.cycles
                for output_idx, output_cycles in enumerate(segment.output_cycles):
                    incoming_outputs[output_idx] += len(output_cycles)
            else:
                # While we'll be skipping all hash checks until the unexplored branch and thus don't need to restore
                # most of the reference input' molecules that were extracted from the PRNG, we do need to restore
                # the one(s) that triggers the unexplored branch, as that will be used to add the new branch to the node
                # on the first cycle after renewing hash checking
                for input_idx, input_mol_idx in enumerate(new_input_key):
                    if input_mol_idx is not None:
                        self._random_input_copies[input_idx].forecast_queue.appendleft(input_mol_idx)

                # Indicate how many cycles we should skip state checking for, to avoid accidentally re-triggering
                # a search (as all the states we'll pass through on the way to the new branch are known)
                # The last forecast cycle is the one that will put us in a new branch and needs to be checked, so
                # we need a -1 here.
                # TODO: I think this is correct but the random input cycle - technically the start of a
                #       segment - may have been 'two-wrongs-make-a-right'ed
                self._cycles_to_new_state = incoming_cycles - 1
                # Also update the 'current segment' to what it will be when we start state-checking again
                self._cur_state_tree_segment_idx = segment_idx

                break

    def run(self, max_cycles: Optional[float] = None,  # Sucky way to make sure math.inf doesn't get yelled at
            hash_states: int = 1000,
            debug: Optional[DebugOptions] = False) -> Score:
        """Run this solution, returning a Score or raising an exception if it crashes or exceeds max_cycles.

        Args:
            max_cycles: Maximum cycle count to run to. Default 1.1x the expected cycle count in the solution metadata,
                        or 1,000,000 cycles if not provided. Also accepts math.inf.
            hash_states: Maximum number of unique cycle states to hash for the purposes of loop detection. Default 1000.
                         Pass 0 to disable hashing. Can theoretically result in invalid results due to hash collisions,
                         but in practice this is not a problem.
                     Default True.
            debug: Print an updating view of the solution while running. See DebugOptions.
        """
        # Set the maximum runtime if unspecified to ensure a broken solution can't infinite loop forever
        if max_cycles is None:
            if self.expected_score is not None:
                max_cycles = int(1.1 * self.expected_score.cycles)
            else:
                max_cycles = self.DEFAULT_MAX_CYCLES

        # Hashing can't handle defense and sandbox levels
        if self.level.type in {'defense', 'sandbox'}:
            hash_states = 0

        # Default debug view to the reactor's interior in research levels
        if debug and self.level['type'].startswith('research'):
            debug.reactor = 0

        reactors = list(self.reactors)
        outputs = list(self.outputs)

        # Hashing-related helpers that need to be dynamic (e.g. to pick up changes to the inputs' seeds that may have
        # been done after the constructor)
        if hash_states > 0:
            self._random_inputs = [c for c in self.components if isinstance(c, RandomInput)]  # Convenience
            # Reference copies of the random inputs
            # Ease tracking which molecules were just consumed from the random inputs' pipes
            self._random_input_copies = [copy.deepcopy(i) for i in self._random_inputs]

        # If we are continuing running from a pause, start with a movement phase
        # TODO: If it was a red pause, need to do the blue instants phase first... and if both
        #       there's no way to tell which it was without storing an extra variable in Solution
        if any(waldo.commands[waldo.position].type == InstructionType.PAUSE
               for reactor in reactors
               for waldo in reactor.waldos
               if waldo.position in waldo.commands):
            self.cycle_movement()

        # Run the level
        try:
            # In debug mode the cursor can annoyingly flicker into the middle of the printed output; hide it
            if debug:
                cursor.hide()

            while self.cycle < max_cycles + 1:
                self.cycle += 1

                if debug and self.cycle >= debug.cycle:
                    self.debug_print(duration=0.5 / debug.speed, reactor_idx=debug.reactor,
                                     show_instructions=debug.show_instructions)

                if self.boss is not None:
                    self.boss.do_instant_actions(self.cycle)

                # Execute instant actions (component inputs/outputs, waldo instructions)
                for component in self.components:
                    if component.do_instant_actions(self.cycle):
                        # Outputs return True the first time they reach their target count; whenever one does, check
                        # all the others for completion (we can't just count completions since 0/0 outputs should
                        # win even if never triggered). Not checking until an output completes also matches the expected
                        # behaviour of puzzles where all outputs are 0/0
                        # Weapons also return True if they just killed the boss
                        if all(output.current_count >= output.target_count for output in outputs):
                            if self.boss is not None:
                                self.cycle += self.boss.DEATH_ANIMATION_CYCLES

                            # -1 looks weird but seems provably right based on output vs pause comparison
                            return Score(self.cycle - 1, len(reactors), self.symbols)

                if debug and self.cycle >= debug.cycle:
                    self.debug_print(duration=0.5 / debug.speed, reactor_idx=debug.reactor,
                                     show_instructions=debug.show_instructions)

                self.cycle_movement()

                # Attempt to fast-forward if we're within our hash memory limit
                if len(self._prior_states) < hash_states and self.hash_and_check_state(debug=bool(debug)):
                    return Score(self.cycle - 1, len(reactors), self.symbols)

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

    # Helper so evaluate() can behave like validate() while still accessing the true score if it was obtained
    def _validate_setup(self, max_cycles):
        # Sanity check arguments
        if self.expected_score is None:
            raise ValueError(f"{self.description}: validate() requires a valid expected score (currently 0-0-0);"
                             " update the solution metadata line or use run() instead.")

        if max_cycles is not None:
            if self.expected_score.cycles > max_cycles:
                raise ValueError(f"{self.description}: Cannot validate; expected cycles"
                                 f" ({self.expected_score.cycles}) > max cycles ({max_cycles})")

            # If validating, limit the max cycles based on the expected score, to save time
            max_cycles = min(max_cycles, int(self.expected_score.cycles * 1.1))

        # Skip running if the reactor or symbol counts don't match
        reactors, symbols = len(list(self.reactors)), self.symbols
        if reactors != self.expected_score.reactors:
            raise ScoreError(f"{self.description}: Expected {self.expected_score.reactors} reactors but got {reactors}.")
        elif symbols != self.expected_score.symbols:
            raise ScoreError(f"{self.description}: Expected {self.expected_score.symbols} symbols but got {symbols}.")

        return max_cycles

    def validate(self, max_cycles: Optional[float] = None,
                 hash_states: int = 1000,
                 debug: Optional[DebugOptions] = False) -> Solution:
        """Same behaviour as Solution.run, but additionally raises an exception if the score does not match
        the expected score in solution metadata (expected_score). The run is skipped entirely if reactor/symbol counts
        don't match.

        See Solution.run for details and available arguments.
        """
        max_cycles = self._validate_setup(max_cycles)

        # Run the solution to completion
        cycles, _, _ = self.run(max_cycles=max_cycles, hash_states=hash_states, debug=debug)

        # Validate the cycle count
        if cycles != self.expected_score.cycles:
            raise ScoreError(f"{self.description}: Expected {self.expected_score.cycles} cycles but got {cycles}.")

        return self

    def is_precognitive(self, *args, **kwargs):
        return is_precognitive(self, *args, **kwargs)

    def evaluate(self, max_cycles: Optional[float] = None,
                 hash_states: int = 1000,
                 strict: bool = False,
                 check_precog: bool = False,
                 max_precog_check_cycles: Optional[float] = None,
                 verbosity: int = 0,
                 debug: Optional[DebugOptions] = False,
                 _run: bool = True) -> dict:
        """Run this solution, validating the expected score if it isn't None. Return a dict of summary data.

        Dict fields: level_name, resnet_id (a tuple of the ResearchNet volume + issue + puzzle, included only for
                     ReseachNet levels), cycles, reactors, symbols, author, and solution_name.

        If the solution cannot run to completion or the score check fails, an 'error' field containing any raised
        exception will be included, and the 'cycles' field may be omitted. Note that in the case of the solution passing
        but the cycle count not matching that expected, the 'cycles' field will be included, with the true cycle count.

        Args:
            max_cycles: Maximum cycle count to run to. Default 1.1x the expected cycle count in the solution metadata,
                or 1,000,000 cycles if no expected_score. Also accepts math.inf.
            hash_states: Maximum number of unique cycle states to hash for the purposes of loop detection. Default 1000.
                         Pass 0 to disable hashing.
            strict: Require that expected_score isn't None, always validating it.
            check_precog: If True, do additional runs on the solution to check if it fits the current community
                definition of a precognitive solution. Default False.
                Adds boolean 'precog' and string 'precog_explanation' fields to the returned dict, unless the precog
                check times out, in which case the TimeoutError is returned in the 'error' field.
                See `Solution.is_precognitive` for more info and a direct API.
            max_precog_check_cycles: The maximum total cycle count that may be used by all precognition-check runs; if
                this value is exceeded before sufficient confidence in an answer is obtained, a TimeoutError is raised,
                or in the case of return_json, the 'precog' field is set to None.
                Default 2,000,000 cycles (this is sufficient for basically any sub-10k solution).
            verbosity: 0 (default): No prints.
                       1: Print a message on the validated score, and a precog check report if the solution was precog.
                       2+: Print a message on the validated score, and a precog check report regardless of result.
            debug: Print an updating view of the solution while running; see DebugOptions. Default False.
        """
        result = {'level_name': self.level.name,
                  'resnet_id': self.level.resnet_id,
                  'author': self.author,
                  'cycles': None,  # May be deleted, but this keeps it in order
                  'reactors': len(list(self.reactors)),
                  'symbols': self.symbols,
                  'solution_name': self.name}

        # Catch runtime and score errors so we can still report which level the solution was for
        try:
            # When checking the score, we tighten the max cycles based on the expected score to save time. Keep the
            # original max cycles handy so we can use it for the precog check without over-restricting alternate runs
            original_max_cycles = max_cycles

            if strict or self.expected_score is not None:
                # _validate_setup returns an adjusted max_cycles and validates reactor/symbol counts
                # Note that we preserve the original value for the precog check's usage
                max_cycles = self._validate_setup(max_cycles)

            if _run:
                score = self.run(max_cycles=max_cycles, hash_states=hash_states, debug=debug)
                result['cycles'] = score.cycles

                # Validate the cycle count if expected score was present
                if self.expected_score is not None and score.cycles != self.expected_score.cycles:
                    raise ScoreError(f"{self.description}: Expected {self.expected_score.cycles} cycles but got"
                                     f" {score.cycles}.")

                if verbosity >= 1:
                    print(f"Validated {self.describe(self.level.name, self.author, score, self.name)}")
            else:
                # --no-run mode sets cycles to the expected cycles if available
                if self.expected_score is not None:
                    result['cycles'] = self.expected_score.cycles

                if verbosity >= 1:
                    print(self.description)

            if check_precog:
                result['precog'], result['precog_explanation'] = \
                    self.is_precognitive(max_cycles=original_max_cycles,
                                         max_total_cycles=max_precog_check_cycles,
                                         hash_states=hash_states,
                                         just_run_cycle_count=result['cycles'],
                                         include_explanation=True)

                # At the middle verbosity level, print the explanation only if the solution was precognitive
                if verbosity == 2 or (result['precog'] and verbosity == 1):
                    print(result['precog_explanation'])
        except Exception as e:
            result['error'] = e

            if verbosity >= 1:
                print(f"{type(e).__name__}: {e}", file=sys.stderr)

        # Drop fields rather than setting them to None
        for k, v in list(result.items()):
            if v is None:
                del result[k]

        return result

    def reset(self):
        """Reset this solution as if it had not yet been run."""
        for component in self.components:
            component.reset()

        if self.boss is not None:
            self.boss.reset()

        self.cycle = 0

        # Hashing-related vars
        self._prior_states = {}  # Set containing hashes of every previous state
        self._state_tree_segments = [StateTreeSegment(num_outputs=len(list(self.outputs)))]
        self._cur_state_tree_segment_idx = 0
        self._state_tree_nodes = []
        self._cycles_to_new_state = 0

        return self


# Ugly but this basically keeps precognition analysis in its own module
Solution.is_precognitive.__doc__ = is_precognitive.__doc__
