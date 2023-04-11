#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import collections
import itertools
import math
from typing import List, Optional

from .elements import elements_dict
from .exceptions import *
from .grid import *
from .molecule import Molecule, Atom, ATOM_RADIUS
from .schem_random import SChemRandom
from .waldo import Waldo, Instruction, InstructionType


# Dimensions of components that differ from the standard dimensions for their type
COMPONENT_SHAPES = {
    # SC stores co-ordinates as col, row
    'drag-research-input': (1, 1),
    'drag-research-output': (1, 1),
    'disabled-output': (1, 1),
    'drag-silo-input': (5, 5),
    'drag-atmospheric-input': (2, 2),
    'drag-oceanic-input': (2, 2),
    'drag-powerplant-input': (14, 15),
    'drag-mining-input': (3, 2),
    'drag-ancient-input': (2, 2),
    'drag-spaceship-input': (2, 2)}  # TODO: Actually (2,3) but its pipe isn't in the middle which fucks our assumptions

# Custom and ResNet research levels use the custom research reactor type and specify all features directly in the
# level JSON. Specify the features of all other reactor types.
DEFAULT_RESEARCH_REACTOR_TYPE = 'custom-research-reactor'
REACTOR_TYPES = {
    # Standard production/research reactor types
    'drag-starter-reactor': {'bonder-count': 4},
    'drag-disassembly-reactor': {'bonder-minus-count': 4, 'has-bottom-input': False},
    'drag-assembly-reactor': {'bonder-plus-count': 4, 'has-bottom-output': False},
    'drag-advanced-reactor': {'bonder-count': 4, 'has-sensor': True},
    'drag-fusion-reactor': {'bonder-count': 4, 'has-fuser': True},
    'drag-superbonder-reactor': {'bonder-count': 8},
    'drag-nuclear-reactor': {'bonder-count': 4, 'has-fuser': True, 'has-splitter': True},
    'drag-quantum-reactor': {'bonder-count': 4, 'has-sensor': True, 'has-teleporter': True,
                             'quantum-walls-y': {'5': [0, 1, 2, 3, 4, 5, 6, 7]}},
    'drag-sandbox-reactor': {'bonder-count': 8, 'has-sensor': True, 'has-fuser': True, 'has-splitter': True},
    # Reactor types that only appear in official non-ResNet research levels.
    'empty-research-reactor': {},  # 1-1 - 1-3
    'tutorial-research-reactor-2': {'bonder-count': 4},  # 1-4
    'drag-reduced-reactor': {'bonder-count': 2},  # 6-1, 7-2
    'drag-largeoutput-reactor': {'bonder-count': 4, 'has-sensor': True, 'has-large-output': True},  # 6-3, 7-4, 8-2
    'drag-superlaser-reactor': {'bonder-count': 8, 'has-sensor': True},  # 6-D
    'drag-advancedfusion-reactor': {'bonder-count': 2, 'has-sensor': True, 'has-fuser': True},  # 7-3
    'drag-quantum-reactor-x': {'has-teleporter': True, 'quantum-walls-y': {'5': [0, 1, 2, 3, 4, 5, 6, 7]}},  # QT-1
    'drag-quantum-reactor-s': {'bonder-count': 2, 'has-sensor': True, 'has-teleporter': True,  # QT-2 - QT-4
                               'quantum-walls-y': {'5': [0, 1, 2, 3, 4, 5, 6, 7]}}}


class Pipe:
    """A SpaceChem component's pipe. All posns are relative to the parent component's posn."""
    __slots__ = 'posns', '_molecules', '_add_cycles', '_last_pop_cycle'

    def __init__(self, posns: List[Position]):
        """Construct a pipe. posns should be defined relative to the pipe's parent component posn."""
        self.posns = posns
        # To avoid incurring any performance costs for moving molecules through the pipe, implement a pipe as a timed
        # queue, storing the cycle each molecule was added on and only allowing them to exit the pipe if their age in
        # cycles is >= len(self) - 1 (e.g. for a 1-long pipe, they are available in the same cycle).
        self._molecules = collections.deque()
        self._add_cycles = collections.deque()  # Kept in lockstep with _molecules
        # To ensure reactors can't input from a pipe twice in a cycle, track the last cycle of a successful pop()
        self._last_pop_cycle = -1

    def __len__(self):
        return len(self.posns)

    def hashable_repr(self, cycle):
        """Return a hashable representation of this pipe's contents."""
        # TODO: I *think* returning a frozenset of the molecules and their indices will be better in terms of reducing
        #       hash collisions since every empty value in the pipe adds a chance to find a collision (and we don't need
        #       the hash to work well across pipe lengths so removing the empty values makes sense), but it's incredibly
        #       hard to get concrete info on this and/or if the extra inclusion of indices would reduce entropy more
        #       than it was increased by frozenset's de-ordering.
        return tuple((None if molecule is None else molecule.hashable_repr())
                     for molecule in self.to_list(cycle))

    def get(self, idx: int, cycle: int) -> Optional[Molecule]:
        """Return the molecule at the given index, else None."""
        if not self._molecules:
            return None
        # Provide O(1) access for either end which is all we use, else fall back to O(N)
        elif idx == 0:
            # Cases where there is a molecule at the front:
            if (len(self._molecules) == len(self)  # Pipe is full
                    or cycle == self._add_cycles[0]  # A molecule was just added this cycle (hasn't moved yet)
                    # Pipe was just full, and remaining (non-empty) molecules haven't moved yet
                    or (cycle == self._last_pop_cycle and 1 <= len(self._molecules) == len(self) - 1)):
                return self._molecules[0]
            else:
                return None
        elif idx == -1:
            # Make sure it's had time to reach the end. Note that if a pop() just happened this cycle, the molecules
            # won't have moved yet
            if cycle - self._add_cycles[-1] >= len(self) - 1 and cycle != self._last_pop_cycle:
                return self._molecules[-1]
            else:
                return None
        else:
            return self.to_list(cycle)[idx]

    def push(self, molecule: Molecule, cycle: int) -> bool:
        """Attempt to pass the given molecule to the pipe. Return False if there is no room at the front of the pipe.
        Note that since cycle incrementation controls pipe movement, if a molecule is being added that shouldn't be
        moved in the same cycle (namely, all components except reactor), cycle + 1 should be given as the cycle the
        molecule was added instead.
        """
        if self._molecules and (len(self._molecules) == len(self)  # Pipe is full
                                or cycle == self._add_cycles[0]):  # A molecule was already added this cycle
            return False
        else:
            self._molecules.appendleft(molecule)
            self._add_cycles.appendleft(cycle)
            return True

    def pop(self, cycle: int) -> Optional[Molecule]:
        """Remove and return the molecule at the end of the pipe, or None if there is none."""
        if (not self._molecules
                or cycle - self._add_cycles[-1] < len(self) - 1  # Make sure it's had time to reach the end
                or cycle == self._last_pop_cycle):  # Make sure a component can't extract two molecules in a cycle
            return None
        else:
            self._last_pop_cycle = cycle
            self._add_cycles.pop()
            return self._molecules.pop()

    def to_list(self, cycle: int) -> list:
        """Return a list representing the current positions of molecule in the pipe, with None for empty spaces."""
        result = [None for _ in self.posns]

        # Insert molecules starting from the back, to account for backed up molecules
        cur_pipe_idx = len(self) - 1
        for molecule, cycle_added in zip(reversed(self._molecules), reversed(self._add_cycles)):
            # Add this molecule in the farthest position it could have travelled to, accounting for clogs
            if cycle - cycle_added >= cur_pipe_idx:
                result[cur_pipe_idx] = molecule
                # Update the current pipe idx
                cur_pipe_idx -= 1
            else:  # Once we are past any clogged molecules, we will never need cur_pipe_idx again
                result[cycle - cycle_added] = molecule

        return result

    @classmethod
    def from_preset_string(cls, start_posn: Position, dirns_str: str) -> Pipe:
        """Construct a pipe from the given CE pipe string, e.g. 'RRDRUULR', moving in the indicated directions
        (U = Up, R = Right, D = Down, L = Left) from the start_posn (should be relative to the parent component's posn).
        """
        posns = [start_posn]
        char_to_dirn = {'U': UP, 'R': RIGHT, 'D': DOWN, 'L': LEFT}
        for dirn_char in dirns_str:
            posns.append(posns[-1] + char_to_dirn[dirn_char])

        assert len(posns) == len(set(posns)), "Pipe overlaps with itself"

        return Pipe(posns)

    @classmethod
    def from_export_str(cls, export_str: str):
        """Note that a pipe's solution lines might not be contiguous. It is expected that the caller filters
        out the lines for a single pipe and passes them as a single string to this method.
        """
        lines = [s for s in export_str.split('\n') if s]  # Split into non-empty lines
        # Ensure all non-empty lines are valid and for the same-indexed pipe
        assert all(s.startswith('PIPE:0,') for s in lines) or all(s.startswith('PIPE:1,') for s in lines), \
            "Invalid lines in pipe export string"

        # Extract and store the pipe's positions, checking for discontinuities in the given pipe positions
        posns = []
        for line in lines:
            fields = line.split(',')
            assert len(fields) == 3, f"Invalid num fields in PIPE line:\n{line}"

            posn = Position(col=int(fields[1]), row=int(fields[2]))
            if posns:
                assert abs(posn - posns[-1]) in ((0, 1), (1, 0)), "Pipe is not contiguous"
            posns.append(posn)

        assert posns, "Expected at least one PIPE line"
        assert len(posns) == len(set(posns)), "Pipe overlaps with itself"

        return Pipe(posns)

    def export_str(self, pipe_idx: int = 0) -> str:
        """Represent this pipe in solution export string format."""
        return '\n'.join(f'PIPE:{pipe_idx},{posn.col},{posn.row}' for posn in self.posns)

    def reset(self):
        """Empty this pipe."""
        self._molecules = collections.deque()
        self._add_cycles = collections.deque()
        self._last_pop_cycle = -1


class Component:
    """Informal Interface class defining methods overworld objects will implement one or more of."""
    __slots__ = 'type', 'posn', 'dimensions', 'in_pipes', 'out_pipes'

    def __new__(cls, component_dict=None, _type=None, **kwargs):
        """Return a new object of the appropriate subclass based on the component type."""
        # If this is being called from a child class, behave like a normal __new__ implementation (to avoid recursion)
        if cls != Component:
            return object.__new__(cls)

        if _type is None:
            _type = component_dict['type']

        # Instantiate the appropriate subclass. If the subclass also overrides __new__ for deeper subclassing, we'll
        # pass through their __new__ method too, otherwise we can save a recursion by using super() (object)
        # instead
        parts = _type.split('-')
        if len(parts) >= 3:
            if parts[2] == 'reactor':
                return Reactor.__new__(Reactor, _type=_type)
            elif parts[2] == 'input':
                return Input.__new__(Input, component_dict)
            elif _type == 'drag-printer-output':
                return super().__new__(OutputPrinter)
            elif _type == 'drag-printer-passthrough':
                return super().__new__(PassThroughPrinter)
            elif parts[2] in ('output', 'production'):
                return super().__new__(Output)
            elif _type == 'drag-storage-tank':
                return super().__new__(StorageTank)
            elif _type == 'drag-storage-tank-infinite':
                return super().__new__(InfiniteStorageTank)
            elif _type == 'drag-qpipe-in':
                return super().__new__(TeleporterInput)
            elif _type == 'drag-qpipe-out':
                return super().__new__(TeleporterOutput)
            elif parts[1] == 'weapon':  # SuperLaserReactor is handled by Reactor
                return Weapon.__new__(Weapon, component_dict, _type=_type)
        elif _type == 'drag-recycler':
            return super().__new__(Recycler)
        elif _type == 'freeform-counter':
            return super().__new__(PassThroughCounter)

        raise ValueError(f"Unrecognized component type {_type}")

    def __init__(self, component_dict=None, _type=None, posn=None, num_in_pipes=0, num_out_pipes=0):
        self.type = _type if _type is not None else component_dict['type']
        self.posn = Position(*posn) if posn is not None else Position(col=component_dict['x'], row=component_dict['y'])
        self.dimensions = COMPONENT_SHAPES[self.type] if self.type in COMPONENT_SHAPES else self.DEFAULT_SHAPE

        self.in_pipes = [None for _ in range(num_in_pipes)]
        # Initialize output pipes in middle, rounded down, accounting for any level-preset pipes
        self.out_pipes = []
        pipe_start_posn = Position(col=self.dimensions[0], row=(self.dimensions[1] - 1) // 2)
        if component_dict is not None and 'output-pipes' in component_dict:
            assert len(component_dict['output-pipes']) == num_out_pipes, f"Unexpected number of output pipes for {self.type}"
            for pipe_dirns_str in component_dict['output-pipes']:
                self.out_pipes.append(Pipe.from_preset_string(pipe_start_posn, pipe_dirns_str))
                pipe_start_posn += DOWN
        else:
            for _ in range(num_out_pipes):
                self.out_pipes.append(Pipe(posns=[pipe_start_posn]))
                pipe_start_posn += DOWN

    def hashable_repr(self, cycle):
        """Represent this component and its pipes in a hashable format."""
        # Hashing pipes every cycle loses all the gains from the O(1) pipe rework... but we won't hash pipes in the
        # non-naive version of hashing so leaving it be
        # By default a component's only hash-relevant contents are its pipes
        return tuple(pipe.hashable_repr(cycle) for pipe in self.out_pipes)

    @classmethod
    def parse_metadata(cls, s):
        """Given a component export string or its COMPONENT line, return its component type and posn."""
        component_line = s.strip('\n').split('\n', maxsplit=1)[0]  # Get first non-empty line

        # Parse COMPONENT line
        assert component_line.startswith('COMPONENT:'), "Missing COMPONENT line in export string"
        fields = component_line.split(',')
        assert len(fields) == 4, f"Unrecognized component line format:\n{component_line}"

        component_type = fields[0][len('COMPONENT:'):].strip("'")
        component_posn = Position(int(fields[1]), int(fields[2]))
        # TODO: Still don't know what the 4th field does...

        return component_type, component_posn

    def update_from_export_str(self, export_str, update_pipes=True):
        """Given a matching export string, update this component. Optionally ignore pipe updates (namely necessary
        for Ω-Pseudoethyne which disallows mutating a 1-long pipe where custom levels do not.
        """
        component_line, *pipe_lines = (s for s in export_str.split('\n') if s)  # Remove empty lines and get first line

        _, component_posn = self.parse_metadata(component_line)
        assert component_posn == self.posn, f"No component at posn {component_posn}"
        # TODO: Is ignoring component type checks unsafe?
        #assert component_type == self.type, \
        #    f"Component of type {self.type} cannot be overwritten with component of type {component_type}"

        # Check that any pipe lines are superficially valid (all PIPE:0 or PIPE:1), which SC does even if
        # the component does not accept pipe updates (e.g. research reactors)
        # Ensure all non-empty lines are valid
        for pipe_line in pipe_lines:
            if not (pipe_line.startswith('PIPE:0') or pipe_line.startswith('PIPE:1')):
                raise ValueError(f"Unexpected line in component pipes: `{pipe_line}`")

        if update_pipes and self.out_pipes:
            # Expect the remaining lines to define the component's output pipes
            # If the pipes on an existing component are updatable, all of them must be specified during an update
            # (as testable by playing around with preset reactors in CE production levels)
            # Whereas when updating presets with non-updatable pipes (e.g. research reactors), all pipes must be included
            assert pipe_lines, f"Some pipes are missing for component {self.type}"
            pipe_export_strs = ['\n'.join(s for s in pipe_lines if s.startswith(f'PIPE:{i},'))
                                for i in range(2)]
            new_out_pipes = [Pipe.from_export_str(s) for s in pipe_export_strs if s]
            assert len(new_out_pipes) == len(self.out_pipes), f"Unexpected number of pipes for component {self.type}"

            for i, pipe in enumerate(new_out_pipes):
                # Preset pipes of length > 1 are immutable
                if len(self.out_pipes[i]) == 1:
                    # Ensure this pipe starts from the correct position
                    assert pipe.posns[0] == Position(col=self.dimensions[0], row=((self.dimensions[1] - 1) // 2) + i), \
                        f"Invalid start position for pipe {i} of component {self.type}"

                    self.out_pipes[i] = pipe

    def __str__(self):
        return f'{self.type},{self.posn}'

    def do_instant_actions(self, _):
        """Do any instant actions (e.g. execute waldo instructions, spawn/consume molecules)."""
        return

    def move_contents(self, _):
        """Move the contents of this object (e.g. waldos/molecules)."""
        pass

    def reset(self):
        """Reset this component and its pipe's contents as if it has never been run."""
        for pipe in self.out_pipes:
            pipe.reset()

        return self

    def export_str(self):
        """Represent this component in solution export string format."""
        # By SC's convention, set the first lines of the export to be the first segment of each pipe
        # Reverse zip to separate the first lines of each export
        first_segment_lines, remainders = [], []
        for i, pipe in enumerate(self.out_pipes):
            first_segment_line, *remainder = pipe.export_str(pipe_idx=i).split('\n', maxsplit=1)
            first_segment_lines.append(first_segment_line)
            remainders.extend(remainder)

        return '\n'.join([f"COMPONENT:'{self.type}',{self.posn.col},{self.posn.row},''",
                          *first_segment_lines, *remainders])


class Input(Component):
    DEFAULT_SHAPE = (2, 3)
    __slots__ = 'molecules', 'input_rate', 'num_inputs'

    # Convenience property for when we know we're dealing with an Input
    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def __new__(cls, input_dict=None, *args, **kwargs):
        """Convert to a random or programmed input if relevant."""
        # Avoid having a required positional arg so stuff like deepcopy works
        if input_dict is None:
            return object.__new__(cls)

        if 'repeating-molecules' in input_dict:
            return object.__new__(ProgrammedInput)

        molecules_key = 'inputs' if 'inputs' in input_dict else 'molecules'
        if len(input_dict[molecules_key]) <= 1:
            return object.__new__(cls)
        else:
            return object.__new__(RandomInput)

    def __init__(self, input_dict, _type=None, posn=None, is_research=False):
        super().__init__(input_dict, _type=_type, posn=posn, num_out_pipes=1)

        # Handle either vanilla or Community Edition nomenclature
        molecules_key = 'inputs' if 'inputs' in input_dict else 'molecules'

        assert len(input_dict[molecules_key]) != 0, "No molecules in input dict"
        self.molecules = [Molecule.from_json_string(input_mol_dict['molecule'])
                          for input_mol_dict in input_dict[molecules_key]]

        if is_research:
            self.input_rate = 1
        elif 'production-delay' in input_dict:
            self.input_rate = input_dict['production-delay']
        else:
            self.input_rate = 10

        self.num_inputs = 0

    def hashable_repr(self, cycle):
        # As a small optimization, ignore the identities of the molecules in the pipe since they can't have had their
        # state affected by a reactor yet; hash only the positional indices of the molecules in the pipe and the number
        # of cycles remaining until the next output.
        return (cycle - 1) % self.input_rate, frozenset(i for i, mol in enumerate(self.out_pipe.to_list(cycle))
                                                        if mol is not None)

    def move_contents(self, cycle):
        """Create a new molecule if on the correct cycle and the pipe has room."""
        # -1 necessary since starting cycle is 1 not 0, while mod == 1 would break on rate = 1
        # Note that we tell the output pipe it's the next cycle, to 'move' its contents before outputting.
        # This prevents double-moving the molecule and allows for continuous flow in the rate = 1 case
        if (cycle - 1) % self.input_rate == 0 and self.out_pipe.get(0, cycle + 1) is None:
            self.out_pipe.push(self.molecules[0].copy(), cycle + 1)
            self.num_inputs += 1

    def reset(self):
        super().reset()
        self.num_inputs = 0

        return self


class RandomInput(Input):
    __slots__ = 'seed', 'random_generator', 'input_counts', 'random_bucket', 'forecast_queue'

    def __init__(self, input_dict, _type=None, posn=None, is_research=False):
        super().__init__(input_dict, _type=_type, posn=posn, is_research=is_research)

        assert len(self.molecules) > 1, "Fixed input passed to RandomInput ctor"

        # Create a random generator with the given seed. Most levels default to seed 0
        self.seed = input_dict['random-seed'] if 'random-seed' in input_dict else 0
        self.random_generator = SChemRandom(seed=self.seed)
        self.random_bucket = []  # Bucket of indices for the molecules in the current balancing bucket

        molecules_key = 'inputs' if 'inputs' in input_dict else 'molecules'
        self.input_counts = [input_mol_dict['count'] for input_mol_dict in input_dict[molecules_key]]

        # Required so the state fast-forwarder can search ahead in the random sequence while putting the drawn values
        # back afterward
        self.forecast_queue = collections.deque()

    # Use the same hashable_repr definition as Input, which (as an optimization) ignores the identities of the pipe's
    # molecules. This allows us to be slightly less naive in our hashing, avoiding the state being considered different
    # just because of a bunch of unprocessed random molecules piling up in the pipe. We'll do our state branching based
    # on the molecules exiting the pipe rather than those currently being produced

    def get_next_molecule_idx(self):
        """Get the next input molecule's index. Exposed to allow for tracking branches in random level states."""
        # If the state fast-forwarder looked ahead in our sequence, use any values it restored first
        if self.forecast_queue:
            return self.forecast_queue.pop()

        # Create the next balance bucket if we've run out.
        # The bucket stores an index identifying one of the 2-3 molecules
        if not self.random_bucket:
            # TODO: Check this method of drawing from the bucket matches results from research levels with two random
            #       inputs (if I recall, both draw from e.g. the 'bottom 6th' of the inputs at the same time regardless
            #       of if that is a 1/6 chance or a larger chance molecule in the respective zone - which is why I think
            #       this implementation is correct)
            for mol_idx, mol_count in enumerate(self.input_counts):
                self.random_bucket.extend(mol_count * [mol_idx])

        # Randomly remove one entry from the bucket and return it
        bucket_idx = self.random_generator.next(len(self.random_bucket))

        return self.random_bucket.pop(bucket_idx)

    def move_contents(self, cycle):
        # -1 necessary since starting cycle is 1 not 0, while mod == 1 would break on rate = 1
        # Note that we tell the output pipe it's the next cycle, to 'move' its contents before outputting
        if (cycle - 1) % self.input_rate == 0 and self.out_pipe.get(0, cycle + 1) is None:
            self.out_pipe.push(self.molecules[self.get_next_molecule_idx()].copy(), cycle + 1)
            self.num_inputs += 1

    def reset(self):
        super().reset()
        self.random_generator = SChemRandom(seed=self.seed)
        self.random_bucket = []

        return self


class ProgrammedInput(Input):
    __slots__ = 'starting_molecules', 'starting_idx', 'repeating_molecules', 'repeating_idx'

    def __init__(self, input_dict, _type=None, posn=None, is_research=False):
        super(Input, self).__init__(input_dict, _type=_type, posn=posn, num_out_pipes=1)  # Skip Input init

        assert len(input_dict['repeating-molecules']) != 0, "No repeating molecules in input dict"
        self.starting_molecules = [Molecule.from_json_string(s) for s in input_dict['starting-molecules']]
        self.starting_idx = 0
        self.repeating_molecules = [Molecule.from_json_string(s) for s in input_dict['repeating-molecules']]
        self.repeating_idx = 0

        if is_research:
            self.input_rate = 1
        elif 'production-delay' in input_dict:
            self.input_rate = input_dict['production-delay']
        else:
            self.input_rate = 10

        self.num_inputs = 0

    def hashable_repr(self, cycle):
        # Given the info of what repetition index the programmed input is on, the identities of the molecules in the
        # pipe can be deduced, so we're safe to again use Input's identity-unaware handling of the pipe's molecules
        return self.starting_idx, self.repeating_idx, super().hashable_repr(cycle)

    def move_contents(self, cycle):
        # -1 necessary since starting cycle is 1 not 0, while mod == 1 would break on rate = 1
        # Note that we tell the output pipe it's the next cycle, to 'move' its contents before outputting
        if (cycle - 1) % self.input_rate == 0 and self.out_pipe.get(0, cycle + 1) is None:
            if self.starting_idx == len(self.starting_molecules):
                self.out_pipe.push(self.repeating_molecules[self.repeating_idx].copy(), cycle + 1)
                self.repeating_idx = (self.repeating_idx + 1) % len(self.repeating_molecules)
            else:
                self.out_pipe.push(self.starting_molecules[self.starting_idx].copy(), cycle + 1)
                self.starting_idx += 1

            self.num_inputs += 1

    def reset(self):
        super().reset()
        self.starting_idx = self.repeating_idx = 0

        return self


class Output(Component):
    __slots__ = 'output_molecule', 'target_count', 'current_count'
    DEFAULT_SHAPE = (2, 3)

    # Convenience property for when we know we're dealing with an Output
    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def __init__(self, output_dict, _type=None, posn=None, **kwargs):
        super().__init__(output_dict, _type=_type, posn=posn, num_in_pipes=1, **kwargs)

        # CE output components are abstracted one level higher than vanilla output zones; unwrap if needed
        if 'output-target' in output_dict:
            output_dict = output_dict['output-target']

        self.output_molecule = Molecule.from_json_string(output_dict['molecule'])
        self.target_count = output_dict['count']
        self.current_count = 0

    def do_instant_actions(self, cycle):
        """Check for and process any incoming molecule, and return True if this output is completed, else False."""
        if self.in_pipe is None:
            return False

        molecule = self.in_pipe.pop(cycle)
        if molecule is not None:
            if not molecule.isomorphic(self.output_molecule):
                raise InvalidOutputError(f"Invalid output molecule; expected:\n{self.output_molecule}\n\nbut got:\n{molecule}")

            if self.current_count < self.target_count:
                self.current_count += 1

            if self.current_count == self.target_count:
                return True

        return False

    def reset(self):
        super().reset()
        self.current_count = 0

        return self


class PassThroughCounter(Output):
    __slots__ = 'stored_molecule',

    def __init__(self, output_dict):
        # 'count' and 'molecule' are nested in 'target' for pass-through counters; unwrap before calling Output's init
        super().__init__({**output_dict, **output_dict['target']}, num_out_pipes=1)
        self.stored_molecule = None

    def hashable_repr(self, cycle):
        return (self.stored_molecule.hashable_repr() if self.stored_molecule is not None else None,
                super().hashable_repr(cycle))

    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def do_instant_actions(self, cycle):
        """Check for and process any incoming molecule, and return True if this output just completed (in which case
        the caller should check if the other outputs are also done). This avoids checking all output counts every cycle.
        """
        if self.in_pipe is None:
            return False

        # If the stored slot is empty, store the next molecule and 'output' it while we do so
        if self.in_pipe.get(-1, cycle) is not None and self.stored_molecule is None:
            self.stored_molecule = self.in_pipe.get(-1, cycle)
            return super().do_instant_actions(cycle)  # This will remove the molecule from the pipe

        return False

    def move_contents(self, cycle):
        # If there is a molecule stored (possibly stored just now), put it in the output pipe if possible
        # Note that we tell the output pipe it's the next cycle, to 'move' its contents before outputting
        if self.stored_molecule is not None and self.out_pipe.get(0, cycle + 1) is None:
            self.out_pipe.push(self.stored_molecule, cycle + 1)
            self.stored_molecule = None

    def reset(self):
        super().reset()
        self.stored_molecule = None

        return self


# It's less confusing for output counting and user-facing purposes if this is not an Output subclass
class DisabledOutput(Component):
    """Used by research levels, which actually crash if a wrong output is used unlike assembly reactors."""
    DEFAULT_SHAPE = (1, 1)
    __slots__ = ()

    @property
    def in_pipe(self):
        return self.in_pipes[0]

    def __init__(self, *, _type, posn):
        super().__init__(_type=_type, posn=posn, num_in_pipes=1)

    def do_instant_actions(self, cycle):
        # Technically should check for `in_pipe is None` first but I'd also be curious to see this crash since disabled
        # outputs are only used in research levels, where it should be impossible to not connect to the disabled output
        if self.in_pipe.get(-1, cycle) is not None:
            raise InvalidOutputError("A molecule was passed to a disabled output.")


class OutputPrinter(Component):
    """Displays the last 3 molecules passed to it. For now this is effectively going to be a recycler..."""
    DEFAULT_SHAPE = (2, 3)
    __slots__ = ()

    def __init__(self, component_dict=None, _type=None, posn=None, **kwargs):
        super().__init__(component_dict, _type=_type, posn=posn, num_in_pipes=1, **kwargs)

    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def move_contents(self, cycle):
        """Consume and print incoming molecules."""
        # TODO: The lack of an internal buffer on pass-through printers necessitates breaking our standard
        #       of components consuming molecules during the instant_actions phase. Perhaps all non-reactor components
        #       need only use move_contents, and do_instant_actions can be a Reactor-only method.
        if self.in_pipe is not None:
            # TODO: Print received molecules when in --debug somehow
            self.in_pipe.pop(cycle)


class PassThroughPrinter(OutputPrinter):
    """Displays the last 3 molecules passed to it and passes them on. Unlike a PassThroughCounter, has no internal
    slot for holding a molecule (so does not pass-through/print a molecule if its output pipe is full).
    """
    __slots__ = ()

    def __init__(self, component_dict=None, _type=None, posn=None):
        super().__init__(component_dict, _type=_type, posn=posn, num_out_pipes=1)

    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def move_contents(self, cycle):
        """Pass a molecule from the input to the output pipe if possible, and print it."""
        if self.in_pipe is None:
            return

        molecule = self.in_pipe.get(-1, cycle)  # Don't pop to avoid interfering with the super() method
        # Note that we tell the output pipe it's the next cycle, to 'move' its contents before outputting
        if molecule is not None and self.out_pipe.get(0, cycle + 1) is None:
            super().move_contents(cycle)  # This will consume and print the input molecule
            self.out_pipe.push(molecule, cycle + 1)


class Recycler(Component):
    DEFAULT_SHAPE = (5, 5)
    __slots__ = ()

    def __init__(self, component_dict=None, _type=None, posn=None):
        super().__init__(component_dict=component_dict, _type=_type, posn=posn, num_in_pipes=3)

    def do_instant_actions(self, cycle):
        for pipe in self.in_pipes:
            if pipe is not None:
                pipe.pop(cycle)


# TODO: Ideally this would subclass both deque and Component but doing so gives me
#       "multiple bases have instance lay-out conflict". Need to investigate.
class StorageTank(Component):
    DEFAULT_SHAPE = (3, 3)
    MAX_CAPACITY = 25
    __slots__ = 'contents',

    def __init__(self, component_dict=None, _type=None, posn=None):
        super().__init__(component_dict=component_dict, _type=_type, posn=posn, num_in_pipes=1, num_out_pipes=1)
        self.contents = collections.deque()

    # Convenience properties
    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def hashable_repr(self, cycle):
        # Include the stored molecules in the hash
        return (tuple(mol.hashable_repr() for mol in self.contents),
                super().hashable_repr(cycle))

    def do_instant_actions(self, cycle):
        if self.in_pipe is None:
            return

        if self.in_pipe.get(-1, cycle) is not None and len(self.contents) < self.MAX_CAPACITY:
            self.contents.appendleft(self.in_pipe.pop(cycle))

    def move_contents(self, cycle):
        """Add a molecule to the output pipe if the storage tank is not empty."""
        # Note that we tell the output pipe it's the next cycle, to 'move' its contents before outputting
        if self.out_pipe.get(0, cycle + 1) is None and self.contents:
            self.out_pipe.push(self.contents.pop(), cycle + 1)

    @classmethod
    def from_export_str(cls, export_str):
        # First line must be the COMPONENT line
        component_line, pipe_str = export_str.strip().split('\n', maxsplit=1)
        assert component_line.startswith('COMPONENT:'), "StorageTank.from_export_str expects COMPONENT line included"
        fields = component_line.split(',')
        assert len(fields) == 4, f"Unrecognized component line format:\n{component_line}"

        component_type = fields[0][len('COMPONENT:'):].strip("'")
        component_posn = Position(int(fields[1]), int(fields[2]))

        return cls(component_type, component_posn, out_pipe=Pipe.from_export_str(pipe_str))

    def reset(self):
        super().reset()
        self.contents = collections.deque()

        return self


class InfiniteStorageTank(StorageTank):
    MAX_CAPACITY = math.inf


class TeleporterInput(Component):
    DEFAULT_SHAPE = (3, 1)
    __slots__ = 'destination',

    def __init__(self, component_dict):
        super().__init__(component_dict, num_in_pipes=1)
        self.destination = None

    # Convenience properties
    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def do_instant_actions(self, cycle):
        """Note that the teleporter pair behaves differently from a pass-through counter insofar as the pass-through
        counter stores any molecule it receives internally when its output pipe is clogged, whereas the teleporter
        refuses to accept the next molecule until the output pipe is clear (i.e. behaves like a single discontinuous
        pipe that also happens to only allow single atoms through).
        """
        if self.in_pipe is None:
            return

        if self.destination.out_pipe.get(0, cycle) is None:
            molecule = self.in_pipe.pop(cycle)
            if molecule is not None:
                assert len(molecule) == 1, f"An invalid molecule was passed to Teleporter (Input): {molecule}"

                self.destination.out_pipe.push(molecule, cycle + 1)


class TeleporterOutput(Component):
    DEFAULT_SHAPE = (3, 1)
    __slots__ = ()

    def __init__(self, component_dict):
        super().__init__(component_dict, num_out_pipes=1)

    # Convenience properties
    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p


class Reactor(Component):
    DEFAULT_SHAPE = (4, 4)  # Size in overworld
    # For convenience during float-precision rotation co-ordinates, we consider the center of the
    # top-left cell to be at (0,0), and hence the top-left reactor corner is (-0.5, -0.5).
    # Further, treat the walls as being one atom radius closer, so that we can efficiently check if an atom will collide
    # with them given only the atom's center co-ordinates
    NUM_COLS = 10
    NUM_ROWS = 8
    NUM_WALDOS = 2
    NUM_MOVE_CHECKS = 10  # Number of times to check for collisions during molecule movement
    walls = {UP: -0.5 + ATOM_RADIUS, DOWN: NUM_ROWS - 0.5 - ATOM_RADIUS,
             LEFT: -0.5 + ATOM_RADIUS, RIGHT: NUM_COLS - 0.5 - ATOM_RADIUS}
    # Names of features as stored in attributes
    FEATURE_NAMES = ('bonders', 'sensors', 'fusers', 'splitters', 'swappers')
    __slots__ = ('in_pipes', 'out_pipes',
                 'waldos', 'molecules',
                 'large_output', *FEATURE_NAMES,
                 'bond_plus_pairs', 'bond_minus_pairs',
                 'quantum_walls_x', 'quantum_walls_y', 'disallowed_instrs',
                 'annotations', 'debug')

    def __new__(cls, component_dict=None, _type=None, **kwargs):
        """Convert to SuperLaserReactor as needed."""
        _type = component_dict['type'] if _type is None else _type
        if _type == 'drag-superlaser-reactor':
            return object.__new__(SuperLaserReactor)
        else:
            return object.__new__(Reactor)

    def __init__(self, component_dict=None, _type=None, posn=None,
                 _num_in_pipes=2, _num_out_pipes=2):  # Hidden args so these can be overridden by SuperLaserReactor
        """Initialize a reactor from only its component dict, doing e.g. default placements of features. Used for
        levels with preset reactors.
        """
        if component_dict is None:
            component_dict = {}

        # If the reactor type is known, look up its properties and merge them (with lower priority) into the given dict
        # TODO: May want to be stricter about yelling at unknown reactor types (while still playing nice with CE custom
        #       reactors)
        _type = _type if _type is not None else component_dict['type']
        if _type in REACTOR_TYPES:
            component_dict = {**REACTOR_TYPES[_type], **component_dict}  # TODO: Use py3.9's dict union operator

        # If the has-bottom attributes are unspecified, they default to True, unlike most attribute flags
        _num_in_pipes -= 1 if 'has-bottom-input' in component_dict and not component_dict['has-bottom-input'] else 0
        _num_out_pipes -= 1 if 'has-bottom-output' in component_dict and not component_dict['has-bottom-output'] else 0

        super().__init__(component_dict, _type=_type, posn=posn,
                         num_in_pipes=_num_in_pipes, num_out_pipes=_num_out_pipes)

        # Set disallowed instructions
        self.disallowed_instrs = set() if 'disallowed-instructions' not in component_dict else set(component_dict['disallowed-instructions'])

        # Toggles are assumed disallowed unless explicitly set otherwise
        # Note that if controls are specifically set in disallowed-instructions, we ignore has-controls even if present
        if not ('has-controls' in component_dict and component_dict['has-controls']):
            self.disallowed_instrs.add('instr-control')

        # Place all features. For simplicity we will put each feature type in its own column(s) in a fresh reactor
        cur_col = 0

        # Place bonders. Different bonder types go in the same struct so they can share a priority index
        self.bonders = []
        for feature_name, abbrev in (('bonder', '+-'), ('bonder-plus', '+'), ('bonder-minus', '-')):
            if f'{feature_name}-count' in component_dict:
                self.bonders.extend([(Position(cur_col, i), abbrev)
                                     for i in range(component_dict[f'{feature_name}-count'])])
            cur_col += 1
        # Disallow bond commands if there are no bonders
        if not self.bonders:
            self.disallowed_instrs.add('instr-bond')

        # Place remaining features while further disallowing instructions related to non-present features
        # TODO: The evidence that a Feature class is needed continues to grow...
        for attr_name, feature_name, instr_type, feature_width, default_count in \
                (('sensors', 'sensor', 'instr-sensor', 1, 1),
                 ('fusers', 'fuser', 'instr-fuse', 2, 1),
                 ('splitters', 'splitter', 'instr-split', 2, 1),
                 ('swappers', 'teleporter', 'instr-swap', 1, 2)):
            if f'{feature_name}-count' in component_dict:
                setattr(self, attr_name, [Position(cur_col, i) for i in range(component_dict[f'{feature_name}-count'])])
            elif f'has-{feature_name}' in component_dict and component_dict[f'has-{feature_name}']:
                setattr(self, attr_name, [Position(cur_col, i) for i in range(default_count)])
            else:
                setattr(self, attr_name, [])
                self.disallowed_instrs.add(instr_type)

            cur_col += feature_width

        # Pre-compute active bond pairs
        self.bond_plus_pairs, self.bond_minus_pairs = self.bond_pairs()

        self.large_output = 'has-large-output' in component_dict and component_dict['has-large-output']

        # Place Waldo starts at default locations
        self.waldos = [Waldo(idx=i, arrows={}, commands={Position(4, 1 + 5*i): Instruction(InstructionType.START,
                                                                                           direction=LEFT)})
                       for i in range(self.NUM_WALDOS)]

        # Parse any quantum walls from the reactor definition
        self.quantum_walls_x = []
        self.quantum_walls_y = []
        for quantum_walls_key, out_list in [['quantum-walls-x', self.quantum_walls_x],
                                            ['quantum-walls-y', self.quantum_walls_y]]:
            if quantum_walls_key in component_dict and component_dict[quantum_walls_key] is not None:
                # a/b abstract row/col vs col/row while handling the two quantum wall orientations
                # E.g. "quantum-walls-x": {"row1": [col1, col2, col3]}, "quantum-walls-y": {"col1": [row1, row2, row3]}
                for a, bs in component_dict[quantum_walls_key].items():
                    assert len(bs) > 0, "Unexpected empty list in quantum wall definitions"
                    # Since we consider (0, 0) to be the center of the reactor's top-left cell, all quantum walls are on
                    # the half-coordinate grid edges
                    a = int(a) - 0.5  # Unstringify the json key and convert to reactor co-ordinates

                    # Store consecutive quantum walls as one entity. This will reduce collision check operations
                    bs.sort()
                    wall_min = wall_max = bs[0]
                    for b in bs:
                        # If there was a break in the wall, store the last wall and reset. Else extend the wall
                        if b > wall_max + 1:
                            out_list.append((a, (wall_min - 0.5, wall_max + 0.5)))
                            wall_min = wall_max = b
                        else:
                            wall_max = b
                    # Store the remaining wall
                    out_list.append((a, (wall_min - 0.5, wall_max + 0.5)))

        self.annotations = []

        # Store molecules as dict keys to be ordered (preserving Spacechem's hidden 'least recently modified' rule)
        # and to have O(1) add/delete. Dict values are ignored.
        self.molecules = {}

    def bond_pairs(self):
        """For each of + and - bond commands, return a tuple of (bonder_A_posn, bonder_B_posn, dirn) triplets,
        sorted in priority order.
        """
        pair_lists = []
        for bond_type in ('+', '-'):
            # Store the relevant types of bonders in a dict paired up with their indices for fast lookup/sorting (below)
            bonders = {posn: i for i, (posn, bond_types) in enumerate(self.bonders) if bond_type in bond_types}
            pair_lists.append(tuple((posn, neighbor_posn, direction)
                                     for posn in bonders
                                     for neighbor_posn, direction in
                                     sorted([(posn + direction, direction)
                                             for direction in (RIGHT, DOWN)
                                             if posn + direction in bonders],
                                            key=lambda x: bonders[x[0]])))

        return pair_lists

    def update_from_export_str(self, export_str, update_pipes=True):
        features = {'bonders': [], 'sensors': [], 'fusers': [], 'splitters': [], 'swappers': []}

        # One map for each waldo, of positions to pairs of arrows (directions) and/or non-arrow instructions
        waldo_cmd_maps = [{} for _ in range(self.NUM_WALDOS)]  # Can't use * or else dict gets multi-referenced
        waldo_arrow_maps = [{} for _ in range(self.NUM_WALDOS)]

        feature_posns = set()  # for verifying features were not placed illegally

        # Break the component string up into its individual sections, while removing empty lines
        component_line, *lines = (s for s in export_str.split('\n') if s)

        # Member lines
        pipes_idx = next((i for i, s in enumerate(lines) if s.startswith('PIPE:')), len(lines))
        member_lines, lines = lines[:pipes_idx], lines[pipes_idx:]
        if not member_lines:
            raise ValueError("Missing MEMBER lines in reactor component")

        # Pipe and annotation lines
        annotations_idx = next((i for i, s in enumerate(lines) if s.startswith('ANNOTATION:')), len(lines))
        pipe_lines, annotation_lines = lines[:annotations_idx], lines[annotations_idx:]

        # Validates COMPONENT line and updates pipes
        super().update_from_export_str(component_line + '\n' + '\n'.join(pipe_lines), update_pipes=update_pipes)

        # Add members (features and instructions)
        for line in member_lines:
            assert line.startswith('MEMBER:'), f"Unexpected line in reactor members: `{line}`"
            fields = line.split(',')

            if len(fields) != 8:
                raise Exception(f"Unrecognized member line format:\n{line}")

            member_name = fields[0][len('MEMBER:'):].strip("'")

            # Game stores directions in degrees, with right = 0, up = -90 (reversed so sin math works on
            # the reversed vertical axis)
            direction = None if int(fields[1]) == -1 else Direction(1 + int(fields[1]) // 90)

            # Red has a field which is 64 for arrows, 128 for instructions
            # The same field in Blue is 16 for arrows, 32 for instructions
            waldo_idx = 0 if int(fields[3]) >= 64 else 1

            position = Position(int(fields[4]), int(fields[5]))
            assert 0 <= position.col < self.NUM_COLS and 0 <= position.row < self.NUM_ROWS, \
                f"Member {member_name} is out-of-bounds"

            if member_name.startswith('feature-'):
                if position in feature_posns:
                    raise Exception(f"Overlapping features at {position}")
                feature_posns.add(position)

                # Sanity check the other half of double-size features
                if member_name in ('feature-fuser', 'feature-splitter'):
                    position2 = position + RIGHT
                    assert position2.col < self.NUM_COLS, f"Member {member_name} is out-of-bounds"
                    if position2 in feature_posns:
                        raise Exception(f"Overlapping features at {position2}")
                    feature_posns.add(position2)

                if member_name == 'feature-bonder':
                    features['bonders'].append((position, '+-'))
                elif member_name == 'feature-sensor':
                    features['sensors'].append(position)
                elif member_name == 'feature-fuser':
                    features['fusers'].append(position)
                elif member_name == 'feature-splitter':
                    features['splitters'].append(position)
                elif member_name == 'feature-tunnel':
                    features['swappers'].append(position)
                elif member_name == 'feature-bonder-plus':
                    features['bonders'].append((position, '+'))
                elif member_name == 'feature-bonder-minus':
                    features['bonders'].append((position, '-'))
                else:
                    raise Exception(f"Unrecognized member type {member_name}")

                continue

            # Make sure this instruction is legal
            if member_name in self.disallowed_instrs:
                raise ValueError(f"Disallowed instruction type: {repr(member_name)}")

            if member_name == 'instr-arrow':
                assert position not in waldo_arrow_maps[waldo_idx], f"Overlapping arrows at {position}"
                waldo_arrow_maps[waldo_idx][position] = direction
                continue

            assert position not in waldo_cmd_maps[waldo_idx], f"Overlapping commands at {position}"

            if member_name == 'instr-start':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.START, direction=direction)
                continue

            # Note: Some similar instructions have the same name but are sub-typed by the
            #       second integer field
            # TODO: Validate sub-type range, e.g. no using third input in SuperLaserReactor
            instr_sub_type = int(fields[2])
            if member_name == 'instr-input':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.INPUT, target_idx=instr_sub_type)
            elif member_name == 'instr-output':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.OUTPUT, target_idx=instr_sub_type)
            elif member_name == 'instr-grab':
                if instr_sub_type == 0:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.GRAB_DROP)
                elif instr_sub_type == 1:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.GRAB)
                else:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.DROP)
            elif member_name == 'instr-rotate':
                if instr_sub_type == 0:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.ROTATE,
                                                                      direction=Direction.CLOCKWISE)
                else:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.ROTATE,
                                                                      direction=Direction.COUNTER_CLOCKWISE)
            elif member_name == 'instr-sync':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.SYNC)
            elif member_name == 'instr-bond':
                if instr_sub_type == 0:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.BOND_PLUS)
                else:
                    waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.BOND_MINUS)
            elif member_name == 'instr-sensor':
                # The last CSV field is used by the sensor for the target atomic number
                atomic_num = int(fields[7])
                if atomic_num not in elements_dict:
                    raise Exception(f"Invalid atomic number {atomic_num} on sensor command.")
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.SENSE,
                                                                  direction=direction,
                                                                  target_idx=atomic_num)
            elif member_name == 'instr-fuse':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.FUSE)
            elif member_name == 'instr-split':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.SPLIT)
            elif member_name == 'instr-swap':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.SWAP)
            elif member_name == 'instr-toggle':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.FLIP_FLOP, direction=direction)
            elif member_name == 'instr-debug':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.PAUSE)
            elif member_name == 'instr-control':
                waldo_cmd_maps[waldo_idx][position] = Instruction(InstructionType.CONTROL, direction=direction)
            else:
                raise Exception(f"Unrecognized member type {member_name}")

        self.waldos = [Waldo(idx=i, arrows=waldo_arrow_maps[i], commands=waldo_cmd_maps[i])
                       for i in range(self.NUM_WALDOS)]

        # Since bonders of different types get stored together to share a priority idx, check their individual counts
        # match the existing counts
        for bonder_type, feature_name in (('+-', 'bonders'),
                                          ('+', 'bonder-pluses'),
                                          ('-', 'bonder-minuses')):
            actual_count = sum(1 for _, bt in features['bonders'] if bt == bonder_type)
            expected_count = sum(1 for _, bt in self.bonders if bt == bonder_type)
            assert actual_count == expected_count, \
                f"Expected {expected_count} {feature_name} for {self.type} reactor but got {actual_count}"

        # Sanity-check and set features
        for feature_name, posns in features.items():
            assert len(posns) == len(getattr(self, feature_name)), \
                f"Expected {len(getattr(self, feature_name))} {feature_name} for {self.type} reactor but got {len(posns)}"
            setattr(self, feature_name, posns)

        self.bond_plus_pairs, self.bond_minus_pairs = self.bond_pairs()  # Re-precompute bond pairings

        # Store the annotations sorted by output idx (the first field)
        self.annotations = sorted(annotation_lines)

    def export_str(self):
        """Represent this reactor in solution export string format."""
        # Generate the generic component export, and separate it into the metadata and pipe lines
        component_line, *pipes = super().export_str().split('\n', maxsplit=1)

        # By SC's convention, MEMBER lines are ordered as waldo starts, then features, then remaining waldo instructions
        # Grab the waldo start lines from the front of each waldo's export
        waldo_starts, waldo_instrs = [], []
        for waldo in self.waldos:
            start, *instrs = waldo.export_str().split('\n', maxsplit=1)
            waldo_starts.append(start)
            waldo_instrs.extend(instrs)

        # TODO: Make reactors more agnostic of feature types
        features = []
        for (posn, bond_types) in self.bonders:
            if bond_types == '+-':
                feature_name = 'bonder'
            elif bond_types == '+':
                feature_name = 'bonder-plus'
            elif bond_types == '-':
                feature_name = 'bonder-minus'
            else:
                raise Exception("Invalid bonder type in internal data")

            features.append(f"MEMBER:'feature-{feature_name}',-1,0,1,{posn.col},{posn.row},0,0")
        for posn in self.sensors:
            features.append(f"MEMBER:'feature-sensor',-1,0,1,{posn.col},{posn.row},0,0")
        for posn in self.fusers:
            features.append(f"MEMBER:'feature-fuser',-1,0,1,{posn.col},{posn.row},0,0")
        for posn in self.splitters:
            features.append(f"MEMBER:'feature-splitter',-1,0,1,{posn.col},{posn.row},0,0")
        for posn in self.swappers:
            features.append(f"MEMBER:'feature-tunnel',-1,0,1,{posn.col},{posn.row},0,0")

        return '\n'.join([component_line, *waldo_starts, *features, *waldo_instrs, *pipes, *self.annotations])

    def hashable_repr(self, cycle):
        # TODO: Investigate using an incremental hash function for performance
        return (tuple(molecule.hashable_repr() for molecule in self.molecules),
                tuple(self.waldos),
                super().hashable_repr(cycle))

    def __hash__(self):
        """Hash of the current reactor state."""
        return hash((tuple(molecule.hashable_repr() for molecule in self.molecules),
                     tuple(self.waldos)))

    def __str__(self, flash_features=True, show_instructions=False):
        """Return a rich-format pretty-print string representing this reactor."""
        # Each cell gets two characters, + 1 space between cells (we'll use that space to show waldos reticles)
        cells = [[[' ', ' '] for _ in range(self.NUM_COLS)] for _ in range(self.NUM_ROWS)]
        borders = [[' ' for _ in range(self.NUM_COLS + 1)] for _ in range(self.NUM_ROWS)]  # Waldos and zone edges

        # Add faint lines at edges of input/output zones (horizontal border excluded)
        for c in (4, 6):
            for r in range(self.NUM_ROWS):
                if borders[r][c] == ' ':  # Don't overwrite waldos
                    borders[r][c] = '[light_slate_grey]│[/]'

        # Add faint traces of the waldo cmd paths
        waldo_traces = [waldo.trace_path(num_cols=self.NUM_COLS, num_rows=self.NUM_ROWS) for waldo in self.waldos]
        for i, (waldo, color) in enumerate(zip(self.waldos, ('dim red', 'dim blue'))):
            for posn, dirns in waldo_traces[i].items():
                c, r = posn
                path_char = Waldo.dirns_to_char[frozenset(dirns)]

                # Fill in this waldo's half of the cell
                cells[r][c][i] = f'[{color}]{path_char}[/]'

                # If the other waldo has nothing to draw in this cell and our directions include right (red) or left
                # (blue) fill in the other waldo's spot with an extending line
                if i == 0 and RIGHT in dirns and posn not in waldo_traces[1]:
                    cells[r][c][1 - i] = f'[{color}]─[/]'
                elif i == 1 and LEFT in dirns and posn not in waldo_traces[0]:
                    cells[r][c][1 - i] = f'[{color}]─[/]'

                # Extend a line through the border to our left for as far as won't cross or touch the other waldo
                if (borders[r][c] == ' '
                        and LEFT in dirns
                        and (posn not in waldo_traces[1 - i] or i == 0)
                        and (posn + LEFT not in waldo_traces[1 - i]
                             or RIGHT not in waldo_traces[1 - i][posn + LEFT]  # Ok to use border if we won't touch them
                             or i == 1)):
                    borders[r][c] = f'[{color}]─[/]'
                # Vice versa for right
                if (borders[r][c + 1] == ' '
                        and RIGHT in dirns
                        and (posn not in waldo_traces[1 - i] or i == 1)  # Blue can cheat
                        and (posn + RIGHT not in waldo_traces[1 - i]
                             or LEFT not in waldo_traces[1 - i][posn + RIGHT]  # Ok to use border if we won't touch them
                             or i == 0)):
                    borders[r][c + 1] = f'[{color}]─[/]'

        # Add waldo instructions (priority over waldo paths)
        if show_instructions:
            for i, (waldo, color) in enumerate(zip(self.waldos, ('red', 'blue'))):
                for (c, r), cmd in waldo.commands.items():
                    cells[r][c][i] = f'[{color}]{cmd}[/]'

        # Add waldo reticles
        for i, (waldo, color) in enumerate(zip(self.waldos, ('bold red', 'bold blue'))):
            c, r = waldo.position
            waldo_chars = ('|', '|') if waldo.molecule is None else ('(', ')')
            for j in range(2):
                # Color purple where two waldos overlap
                mixed_color = 'bold purple' if '[bold red]' in borders[r][c + j] else color
                borders[r][c + j] = f'[{mixed_color}]{waldo_chars[j]}[/]'

        # Map out the molecules in the reactor (priority over waldo paths/cmds)
        for molecule in self.molecules:
            for (c, r), atom in molecule.atom_map.items():
                # Round co-ordinates in case we are printing mid-rotate
                cell = cells[round(r)][round(c)]
                cell[0] = atom.element.symbol[0]
                if len(atom.element.symbol) >= 2:
                    cell[1] = atom.element.symbol[1]

        # Use grey background for feature cells (bonders, fusers, etc.)
        # Though they're stored together for priority reasons in Reactor.bonders, color + and - bonders separately
        # from regular bonders so as to distinguish them during + vs - commands
        feature_colors = {k: 'light_slate_grey'
                          for k in list(Reactor.FEATURE_NAMES) + ['bonder_pluses', 'bonder_minuses']}
        input_colors = {}
        output_colors = {}

        # Flash the appropriate feature background cells on waldo input, output, bond +/-, etc.
        if flash_features:
            for i, (waldo, waldo_color) in enumerate(zip(self.waldos, ('red', 'blue'))):
                if waldo.position not in waldo.commands:
                    continue

                cmd = waldo.commands[waldo.position]
                if cmd.type == InstructionType.INPUT:
                    input_colors[cmd.target_idx] = waldo_color if cmd.target_idx not in input_colors else 'purple'
                elif cmd.type == InstructionType.OUTPUT:
                    output_colors[cmd.target_idx] = waldo_color if cmd.target_idx not in output_colors else 'purple'
                elif cmd.type == InstructionType.BOND_PLUS:
                    feature_colors['bonders'] = waldo_color if feature_colors['bonders'] != 'red' else 'purple'
                    feature_colors['bonder_pluses'] = waldo_color if feature_colors['bonder_pluses'] != 'red' else 'purple'
                elif cmd.type == InstructionType.BOND_MINUS:
                    feature_colors['bonders'] = waldo_color if feature_colors['bonders'] != 'red' else 'purple'
                    feature_colors['bonder_minuses'] = waldo_color if feature_colors['bonder_minuses'] != 'red' else 'purple'
                elif cmd.type == InstructionType.FUSE:
                    feature_colors['fusers'] = waldo_color if feature_colors['fusers'] != 'red' else 'purple'
                elif cmd.type == InstructionType.SPLIT:
                    feature_colors['splitters'] = waldo_color if feature_colors['splitters'] != 'red' else 'purple'
                elif cmd.type == InstructionType.SWAP:
                    feature_colors['swappers'] = waldo_color if feature_colors['swappers'] != 'red' else 'purple'
                elif cmd.type == InstructionType.SENSE:
                    feature_colors['sensors'] = waldo_color if feature_colors['sensors'] != 'red' else 'purple'

        # Color background of feature cells
        for feature_name, feature_color in feature_colors.items():
            # Extract individual bonder types from their shared struct
            if feature_name == 'bonders':
                cell_posns = set(p for p, bond_types in self.bonders if bond_types == '+-')
            elif feature_name == 'bonder_pluses':
                cell_posns = set(p for p, bond_types in self.bonders if bond_types == '+')
            elif feature_name == 'bonder_minuses':
                cell_posns = set(p for p, bond_types in self.bonders if bond_types == '-')
            else:
                cell_posns = set(getattr(self, feature_name))

            for c, r in cell_posns:
                cells[r][c][0] = f'[on {feature_color}]{cells[r][c][0]}[/]'
                cells[r][c][1] = f'[on {feature_color}]{cells[r][c][1]}[/]'

                # Add the second posn and the border for double-length features
                if feature_name in ('fusers', 'splitters'):
                    # Merging these would be nicer to rich.print but can get screwy if any cell is overridden after
                    borders[r][c + 1] = f'[on {feature_color}]{borders[r][c + 1]}[/]'
                    cells[r][c + 1][0] = f'[on {feature_color}]{cells[r][c + 1][0]}[/]'
                    cells[r][c + 1][1] = f'[on {feature_color}]{cells[r][c + 1][1]}[/]'
                # Fill in the borders of adjacent bonders
                elif 'bonder' in feature_name:
                    if (c + 1, r) in cell_posns:
                        borders[r][c + 1] = f'[on {feature_color}]{borders[r][c + 1]}[/]'

        # Color background of inputs/outputs when activated, excepting already-colored features
        for input_idx, input_color in input_colors.items():
            for c, r in itertools.product(range(4), (range(4) if input_idx == 0 else range(4, 8))):
                if not cells[r][c][0].startswith('[on '):
                    cells[r][c][0] = f'[on {input_color}]{cells[r][c][0]}[/]'
                    cells[r][c][1] = f'[on {input_color}]{cells[r][c][1]}[/]'

            for c, r in itertools.product(range(1, 4), (range(4) if input_idx == 0 else range(4, 8))):
                if not borders[r][c].startswith('[on '):
                    borders[r][c] = f'[on {input_color}]{borders[r][c]}[/]'

        for output_idx, output_color in output_colors.items():
            for c, r in itertools.product(range(6, 10), (range(4) if output_idx == 0 else range(4, 8))):
                if not cells[r][c][0].startswith('[on '):
                    cells[r][c][0] = f'[on {output_color}]{cells[r][c][0]}[/]'
                    cells[r][c][1] = f'[on {output_color}]{cells[r][c][1]}[/]'

            for c, r in itertools.product(range(7, 10), (range(4) if output_idx == 0 else range(4, 8))):
                if not borders[r][c].startswith('[on '):
                    borders[r][c] = f'[on {output_color}]{borders[r][c]}[/]'

        result = f" {self.NUM_COLS * '___'}_ \n"
        for r in range(self.NUM_ROWS):
            result += f"│{''.join(b + c[0] + c[1] for b, c in zip(borders[r], cells[r] + [['', '']]))}│\n"
        result += f" {self.NUM_COLS * '‾‾‾'}‾ "

        return result

    def do_instant_actions(self, cycle):
        for waldo in self.waldos:
            if waldo.position in waldo.arrows:
                waldo.direction = waldo.arrows[waldo.position]

            if waldo.position in waldo.commands:
                self.exec_waldo_cmd(waldo, cycle)

    def move_contents(self, cycle):
        """Move all waldos in this reactor and any molecules they are holding."""
        # If the waldo is facing a wall, mark it as stalled (may also be stalled due to sync, input, etc.)
        for waldo in self.waldos:
            if ((waldo.direction == UP and waldo.position.row == 0)
                    or (waldo.direction == DOWN and waldo.position.row == 7)
                    or (waldo.direction == LEFT and waldo.position.col == 0)
                    or (waldo.direction == RIGHT and waldo.position.col == 9)):
                waldo.is_stalled = True

        # If any waldo is about to rotate a molecule, don't skimp on collision checks
        # Note that a waldo might be marked as rotating (and stalled accordingly) while holding nothing, in the case
        # that red hits a rotate and then has its atom fused or swapped away by blue in the same cycle.
        # We'll ignore those here, hence the waldo.molecule is not None check.
        rotating_waldos = [waldo for waldo in self.waldos if waldo.is_rotating and waldo.molecule is not None]
        sliding_waldos = [waldo for waldo in self.waldos if not waldo.is_stalled and waldo.molecule is not None]
        if rotating_waldos:
            # If both waldos are holding the same molecule and either of them is rotating, a crash occurs
            # (even if they're in the same position and rotating the same direction)
            if self.waldos[0].molecule is self.waldos[1].molecule:
                raise ReactionError("Molecule pulled apart")

            # Check if we can approximate the movement checks based on cells they vs other molecules span
            # This is slightly expensive, but not as expensive as 10x float math collision checks against all molecules
            rotation_spans = [self.rotate_span(waldo.molecule,
                                               pivot=waldo.position,
                                               direction=waldo.cur_cmd().direction)
                              for waldo in rotating_waldos]
            slide_spans = [waldo.molecule.atom_map.keys() | {posn + waldo.direction for posn in waldo.molecule.atom_map}
                           for waldo in sliding_waldos]
            moving_molecules = {waldo.molecule for waldo in self.waldos
                                if waldo.molecule is not None and waldo.is_rotating or not waldo.is_stalled}
            # We already know static molecules don't overlap so pre-merging their spans is safe
            static_molecules_span = {posn for molecule in self.molecules
                                     for posn in molecule.atom_map.keys()
                                     if molecule not in moving_molecules}

            # To check if any moving/rotating/static molecules' spans intersect, union them all and check if
            # the length equals the sum of their individual lengths.
            spans_sum = (sum(len(span) for span in rotation_spans)
                         + sum(len(span) for span in slide_spans)
                         + len(static_molecules_span))
            spans_union = static_molecules_span.union(*rotation_spans, *slide_spans)

            # Note that crossing a wall does not guarantee a collision, both due to quantum walls and because our
            # rotation spans aren't necessarily tight bounds.
            def span_crosses_wall(span):
                if not all(0 <= p.row < self.NUM_ROWS and 0 <= p.col < self.NUM_COLS for p in span):
                    return True

                if self.quantum_walls_y or self.quantum_walls_x:
                    # Check for atoms on both sides of the wall within the span.
                    for c, (r1, r2) in self.quantum_walls_y:
                        if (any(p.col <= c and r1 < p.row < r2 for p in span)
                                and any(p.col >= c and r1 < p.row < r2 for p in span)):
                            return True
                    for r, (c1, c2) in self.quantum_walls_x:
                        if (any(p.row <= r and c1 < p.col < c2 for p in span)
                                and any(p.row >= r and c1 < p.col < c2 for p in span)):
                            return True

                return False

            if spans_sum == len(spans_union) and not any(span_crosses_wall(span)
                                                         for span in (*rotation_spans, *slide_spans)):
                # Given that no molecules are moving near each other or across walls, we can do a single move step and
                # skip further collision checks
                for waldo in sliding_waldos:
                    waldo.molecule.move(waldo.direction)

                for waldo in rotating_waldos:
                    waldo.molecule.rotate(pivot_pos=waldo.position, direction=waldo.cur_cmd().direction)
            else:
                # Otherwise, move each waldo's molecule partway at a time and check for collisions each time
                step_radians = math.pi / (2 * self.NUM_MOVE_CHECKS)
                step_distance = 1 / self.NUM_MOVE_CHECKS
                for _ in range(self.NUM_MOVE_CHECKS):
                    # Move all molecules currently being held by a waldo forward a step
                    for waldo in self.waldos:
                        if waldo.molecule is not None and not waldo.is_stalled:
                            waldo.molecule.move(waldo.direction, distance=step_distance)
                        elif waldo.is_rotating:
                            waldo.molecule.rotate_fine(pivot_pos=waldo.position,
                                                       direction=waldo.cur_cmd().direction,
                                                       radians=step_radians)

                    # After moving all molecules, check each rotated molecule for collisions with walls or other molecules
                    # Though all molecules had to move, only the rotating one(s) needs to do checks at each step, since we
                    # know the other waldo will only have static molecules left to check against, and translation movements
                    # can't clip through a static atom without ending on top of it
                    # Note: This only holds true for <= 2 waldos and since we checked that at least one waldo is rotating
                    for waldo in self.waldos:
                        if waldo.is_rotating:
                            self.check_collisions(waldo.molecule)

                # After completing all steps of the movement, convert moved molecules back to integer co-ordinates and do
                # any final checks/updates
                for waldo in self.waldos:
                    if waldo.molecule is not None and not waldo.is_stalled:
                        waldo.molecule.round_posns()
                        # Do the final check we skipped for non-rotating molecules
                        self.check_collisions_lazy(waldo.molecule, direction=waldo.direction)
                    elif waldo.is_rotating:
                        waldo.molecule.round_posns()
                        # Rotate atom bonds
                        waldo.molecule.rotate_bonds(waldo.cur_cmd().direction)
        elif sliding_waldos:
            # If we are not doing any rotates, we can skip the full collision checks
            # Non-rotating molecules can cause collisions/errors if:
            # * The waldos are pulling a molecule apart
            # * OR The final destination of a moved molecule overlaps any other molecule after the move
            # * OR The final destination of a moved molecule overlaps the initial position of another moving molecule,
            #      and the offending waldos were not moving in the same direction

            if self.waldos[0].molecule is self.waldos[1].molecule:
                # Given that we know one is moving, if the waldos share a molecule they must move in the same direction
                if (any(waldo.is_stalled for waldo in self.waldos)
                        or self.waldos[0].direction != self.waldos[1].direction):
                    raise ReactionError("A molecule has been grabbed by both waldos and pulled apart.")

                # Only mark one waldo as moving a molecule so we don't move their molecule twice
                sliding_waldos = [self.waldos[0]]
            else:
                # (skipped if both waldos holding same molecule)
                # Check if a molecule being moved will bump into the back of another moving molecule
                if len(sliding_waldos) == 2 and self.waldos[0].direction != self.waldos[1].direction:
                    for waldo in self.waldos:
                        # Intersect the target positions of this waldo's molecule with the current positions of the
                        # other waldo's molecules
                        other_waldo = self.waldos[1 - waldo.idx]
                        target_posns = set(posn + waldo.direction for posn in waldo.molecule.atom_map)
                        if not target_posns.isdisjoint(other_waldo.molecule.atom_map):
                            raise ReactionError("Collision between molecules")

            # Move all molecules
            for waldo in sliding_waldos:
                waldo.molecule.move(waldo.direction)

            # Perform collision checks for the moved molecules (AFTER they have all moved)
            for waldo in sliding_waldos:
                self.check_collisions_lazy(waldo.molecule, direction=waldo.direction)

        # Move waldos and mark them as no longer stalled. Note that is_rotated must be left alone to tell it not to
        # rotate twice
        for waldo in self.waldos:
            if not waldo.is_stalled:
                waldo.position += waldo.direction
            waldo.is_stalled = False

    def check_molecule_collisions_lazy(self, molecule):
        """Raise an exception if the given molecule collides with any other molecules.
        Assumes integer co-ordinates in all molecules.
        """
        for other_molecule in self.molecules:
            molecule.check_collisions_lazy(other_molecule)  # Implicitly ignores self

    def check_wall_collisions(self, molecule):
        """Raise an exception if the given molecule collides with any walls."""
        if not all(self.walls[UP] < p.row < self.walls[DOWN]
                   and self.walls[LEFT] < p.col < self.walls[RIGHT]
                   for p in molecule.atom_map):
            raise ReactionError("A molecule has collided with a wall")

    def check_wall_collisions_lazy(self, molecule, direction):
        """Raise an exception if the given molecule collides with the specified wall."""
        if direction == RIGHT:
            if all(p.col < self.NUM_COLS for p in molecule.atom_map):
                return
        elif direction == UP:
            if all(p.row >= 0 for p in molecule.atom_map):
                return
        elif direction == DOWN:
            if all(p.row < self.NUM_ROWS for p in molecule.atom_map):
                return
        elif direction == LEFT:
            if all(p.col >= 0 for p in molecule.atom_map):
                return

        raise ReactionError("A molecule has collided with a wall")

    def check_quantum_wall_collisions(self, molecule):
        """Given a molecule (possibly on float coordinates), check if it collided with a quantum wall."""
        for r, (c1, c2) in self.quantum_walls_x:
            for p in molecule.atom_map:
                # If the atom's center (p) is in line with the wall, check its not too close to the wall segment
                if c1 < p.col < c2:
                    if abs(p.row - r) < ATOM_RADIUS:
                        raise ReactionError("A molecule has collided with a quantum wall")
                # If p is not in line with the wall, we just need to make sure it's not near the wall's endpoints
                elif max((p.col - c1)**2, (p.col - c2)**2) + (p.row - r)**2 < ATOM_RADIUS**2:
                    raise ReactionError("A molecule has collided with a quantum wall")

        for c, (r1, r2) in self.quantum_walls_y:
            for p in molecule.atom_map:
                # If the atom's center (p) is in line with the wall, check its not too close to the wall segment
                if r1 < p.row < r2:
                    if abs(p.col - c) < ATOM_RADIUS:
                        raise ReactionError("A molecule has collided with a quantum wall")
                # If p is not in line with the wall, we just need to make sure it's not near the wall's endpoints
                elif max((p.row - r1)**2, (p.row - r2)**2) + (p.col - c)**2 < ATOM_RADIUS**2:
                    raise ReactionError("A molecule has collided with a quantum wall")

    def check_quantum_wall_collisions_lazy(self, molecule, direction):
        """Given a molecule that just moved and is on integer coordinates, check if it collided with a quantum wall."""
        # Recall that we treat integer coordinates as the centers of cells, so quantum walls are on half-coordinates.
        # E.g. the top wall of the reactor would be row: -0.5, cols: (-0.5, 9.5).
        if direction == UP:
            # Check if the atom just crossed the wall given its movement
            if any(p.row < r < p.row + 1 and c1 < p.col < c2
                   for r, (c1, c2) in self.quantum_walls_x
                   for p in molecule.atom_map):
                raise ReactionError("A molecule has collided with a quantum wall")
        elif direction == DOWN:
            if any(p.row - 1 < r < p.row and c1 < p.col < c2
                   for r, (c1, c2) in self.quantum_walls_x
                   for p in molecule.atom_map):
                raise ReactionError("A molecule has collided with a quantum wall")
        elif direction == RIGHT:
            if any(p.col - 1 < c < p.col and r1 < p.row < r2
                   for c, (r1, r2) in self.quantum_walls_y
                   for p in molecule.atom_map):
                raise ReactionError("A molecule has collided with a quantum wall")
        elif direction == LEFT:
            if any(p.col < c < p.col + 1 and r1 < p.row < r2
                   for c, (r1, r2) in self.quantum_walls_y
                   for p in molecule.atom_map):
                raise ReactionError("A molecule has collided with a quantum wall")

    @staticmethod
    def rotate_span(molecule, pivot, direction):
        """Given a waldo about to rotate a molecule, return a superset of every cell that is at least partially covered
        by the rotation. Runs in O(len(molecule)) time.
        """
        # Calculate a bounding box (in terms of cell coordinates, ignoring atom widths) for the molecule
        t1 = min(posn.row for posn in molecule.atom_map)  # Top
        b1 = max(posn.row for posn in molecule.atom_map)  # Bottom
        l1 = min(posn.col for posn in molecule.atom_map)  # Left
        r1 = max(posn.col for posn in molecule.atom_map)  # Right

        # Calculate the bounding box after rotation
        if direction == Direction.CLOCKWISE:
            t2 = pivot.row - (pivot.col - l1)  # Dist from pivot to left edge, but now above the pivot
            b2 = pivot.row + (r1 - pivot.col)  # and so on...
            l2 = pivot.col - (b1 - pivot.row)
            r2 = pivot.col + (pivot.row - t1)
        else:
            t2 = pivot.row - (r1 - pivot.col)  # Dist from pivot to right edge, but now above the pivot
            b2 = pivot.row + (pivot.col - l1)  # and so on...
            l2 = pivot.col - (pivot.row - t1)
            r2 = pivot.col + (b1 - pivot.row)

        # Calculate the bounding box that supersets the before and after boxes
        t = min(t1, t2)
        r = max(r1, r2)
        b = max(b1, b2)
        l = min(l1, l2)

        # Fatten each box edge by a worst case factor (rounded up) if its relevant adjacent quadrant is non-empty
        # The intuition is that a rotation from say the top-left quadrant to the top-right quadrant can only exit the
        # top of the combined bounding box, and atoms in line with the pivot won't exit the box.
        # This will overcount cells near the corners of the bounding box but is fast to calculate.
        # Claim 1: Finding the true maximum distance of any atom per quadrant gives a tighter bound but isn't worth the
        #          extra computation.
        # Claim 2: Because of the nature of how we calculated the overall bounding box before and after the rotation,
        #          every non-empty quadrant of the bounding box is roughly square, so the use of sqrt(2) is correct.
        extended_nw, extended_ne, extended_se, extended_sw = False, False, False, False  # Compass coordinates
        for posn in molecule.atom_map:
            # Strictly speaking any atom within the diagonal of the quadrant's sides can't extend the box
            if posn.col == pivot.col or posn.row == pivot.row:  # We know these can't extend the box
                continue

            # Keep in mind the molecule is still unrotated
            if not extended_nw and posn.col < pivot.col and posn.row < pivot.row:
                if direction == Direction.CLOCKWISE:
                    t -= math.ceil((math.sqrt(2) - 1) * (pivot.row - t))  # Extends by ~41% which is usually 1-2 cells
                else:
                    l -= math.ceil((math.sqrt(2) - 1) * (pivot.col - l))
                extended_nw = True
            elif not extended_ne and posn.col > pivot.col and posn.row < pivot.row:
                if direction == Direction.CLOCKWISE:
                    r += math.ceil((math.sqrt(2) - 1) * (r - pivot.col))
                else:
                    t -= math.ceil((math.sqrt(2) - 1) * (pivot.row - t))
                extended_ne = True
            elif not extended_se and posn.col > pivot.col and posn.row > pivot.row:
                if direction == Direction.CLOCKWISE:
                    b += math.ceil((math.sqrt(2) - 1) * (b - pivot.row))
                else:
                    r += math.ceil((math.sqrt(2) - 1) * (r - pivot.col))
                extended_se = True
            elif not extended_sw and posn.col < pivot.col and posn.row > pivot.row:
                if direction == Direction.CLOCKWISE:
                    l -= math.ceil((math.sqrt(2) - 1) * (pivot.col - l))
                else:
                    b += math.ceil((math.sqrt(2) - 1) * (b - pivot.row))
                extended_sw = True

        return set(Position(col=col, row=row) for col in range(l, r + 1) for row in range(t, b + 1))

    def check_collisions(self, molecule):
        """Raise an exception if the given molecule is colliding with any other molecules, walls, or quantum walls."""
        for other_molecule in self.molecules:
            molecule.check_collisions(other_molecule)  # Implicitly ignores self

        self.check_wall_collisions(molecule)
        if self.quantum_walls_y or self.quantum_walls_x:
            self.check_quantum_wall_collisions(molecule)

    def check_collisions_lazy(self, molecule, direction):
        """Raise an exception if the given molecule, that just moved in the given direction, collided with any other
        molecules, walls, or quantum walls. Assumes integer co-ordinates in all molecules.
        """
        # Don't abuse direction knowledge for molecule checks since one of them may also have just moved
        self.check_molecule_collisions_lazy(molecule)
        self.check_wall_collisions_lazy(molecule, direction=direction)
        if self.quantum_walls_y or self.quantum_walls_x:
            self.check_quantum_wall_collisions_lazy(molecule, direction=direction)

    def exec_waldo_cmd(self, waldo, cycle):
        cmd = waldo.commands[waldo.position]

        if cmd.type == InstructionType.INPUT:
            self.input(waldo, cmd.target_idx, cycle)
        elif cmd.type == InstructionType.OUTPUT:
            self.output(waldo, cmd.target_idx, cycle)
        elif cmd.type == InstructionType.GRAB:
            self.grab(waldo)
        elif cmd.type == InstructionType.DROP:
            self.drop(waldo)
        elif cmd.type == InstructionType.GRAB_DROP:
            if waldo.molecule is None:
                self.grab(waldo)
            else:
                self.drop(waldo)
        elif cmd.type == InstructionType.ROTATE:
            # If we are holding a molecule and weren't just rotating, start rotating
            # In all other cases, stop rotating
            waldo.is_rotating = waldo.is_stalled = waldo.molecule is not None and not waldo.is_rotating
        elif cmd.type == InstructionType.BOND_PLUS:
            self.bond_plus()
        elif cmd.type == InstructionType.BOND_MINUS:
            self.bond_minus()
        elif cmd.type == InstructionType.SYNC:
            # Mark this waldo as stalled if both waldos aren't on a Sync
            other_waldo = self.waldos[1 - waldo.idx]
            waldo.is_stalled = other_waldo.cur_cmd() is None or other_waldo.cur_cmd().type != InstructionType.SYNC
        elif cmd.type == InstructionType.FUSE:
            self.fuse()
        elif cmd.type == InstructionType.SPLIT:
            self.split()
        elif cmd.type == InstructionType.SENSE:
            for posn in self.sensors:
                molecule = self.get_molecule(posn)
                if molecule is not None and molecule.atom_map[posn].element.atomic_num == cmd.target_idx:
                    waldo.direction = cmd.direction
                    break
        elif cmd.type == InstructionType.FLIP_FLOP:
            # Update the waldo's direction if the flip-flop is on
            if waldo.flipflop_states[waldo.position]:
                waldo.direction = cmd.direction

            waldo.flipflop_states[waldo.position] = not waldo.flipflop_states[waldo.position]  # ...flip it
        elif cmd.type == InstructionType.SWAP:
            self.swap()
        elif cmd.type == InstructionType.PAUSE:
            raise PauseException("Pause command encountered")
        elif cmd.type == InstructionType.CONTROL:
            raise ControlError("CTRL commands are unsupported.")

    def input(self, waldo, input_idx, cycle):
        # If there is no such pipe, stall the waldo
        if input_idx > len(self.in_pipes) - 1 or self.in_pipes[input_idx] is None:
            waldo.is_stalled = True
            return

        # Grab the molecule from the appropriate pipe or stall if no such molecule
        # TODO: Kick out 3.7 some day and walrus-ify this into one if
        new_molecule = self.in_pipes[input_idx].pop(cycle)
        if new_molecule is None:  # Not merged with above case to avoid calling both Pipe.get and Pipe.pop
            waldo.is_stalled = True
            return

        sample_posn = next(iter(new_molecule.atom_map))
        # If the molecule came from a previous reactor, shift its columns from output to input co-ordinates
        # We don't do this immediately on output to save a little work when the molecule is going to an output component
        # anyway (since output checks are agnostic of absolute co-ordinates)
        if sample_posn.col >= 6:
            new_molecule.move(LEFT, 6)
        # Update the molecule's co-ordinates to those of the correct zone if it came from an opposite output zone
        if input_idx == 0 and sample_posn.row >= 4:
            new_molecule.move(UP, 4)
        elif input_idx == 1 and sample_posn.row < 4:
            new_molecule.move(DOWN, 4)

        self.molecules[new_molecule] = None  # Dummy value

        self.check_molecule_collisions_lazy(new_molecule)

    def output(self, waldo, output_idx, cycle):
        # If the there is no such output pipe (e.g. assembly reactor, large output research), do nothing
        if (output_idx > len(self.out_pipes) - 1
                or self.out_pipes[output_idx] is None):
            return

        # TODO: It'd be nice to only have to calculate this for molecules that have been
        #       debonded or dropped, etc. However, the cost of pre-computing it every time
        #       we do such an action is probably not worth the cost of just doing it once
        #       over all molecules whenever output is called.
        # TODO 2: On the other hand, solutions with a waldo wall-stalling on output just
        #         got fucked
        # This manual iter is a little awkward but helps ensure we don't iterate more than once into this dict while
        # we're deleting from it
        molecules_in_zone = iter(molecule for molecule in self.molecules
                                 # Ignore grabbed molecules
                                 if not any(waldo.molecule is molecule for waldo in self.waldos)
                                 and molecule.output_zone_idx(large_output=self.large_output) == output_idx)
        molecule = next(molecules_in_zone, None)

        # Try to output the first molecule in the zone if an output hasn't already been done this cycle
        if molecule is not None:
            # Most components (input, storage tanks, quantum pipes, etc.) behave as though the molecule gets put into
            # their pipe in the movement phase, with an animation of the molecule moving out of the component, and the
            # molecule not being available to the downstream component until the next cycle's instant actions phase.
            # Reactors behave differently, with the spawned molecule appearing directly in the next segment without
            # movement animation, and being available immediately during the same instant actions phase for the next
            # component to consume. To support this with the updated dependence of pipes on the cycle count, treat the
            # molecule as though it was pushed during the previous movement phase (-1 to cycle count since instant
            # actions come before movement in a cycle).
            # TODO: Should figure out if there's a way to make this less awkward.
            if self.out_pipes[output_idx].get(0, cycle) is None:
                # Put the molecule in the pipe and remove it from the reactor
                self.out_pipes[output_idx].push(molecule, cycle)

                # Look for any other outputable molecule
                molecule = next(molecules_in_zone, None)

                # Remove the just-output molecule from the reactor (AFTER finishing with the dict iterator)
                del self.molecules[self.out_pipes[output_idx].get(0, cycle)]

        # If there is any output(s) remaining in this zone (regardless of whether we outputted), stall this waldo
        waldo.is_stalled = molecule is not None

    def get_molecule(self, position):
        """Select the molecule at the given grid position, or None if no such molecule.
        Used by Grab, Bond+/-, Fuse, etc.
        """
        return next((molecule for molecule in self.molecules if position in molecule), None)

    def grab(self, waldo):
        if waldo.molecule is None:
            waldo.molecule = self.get_molecule(waldo.position)

    def drop(self, waldo):
        waldo.molecule = None  # Remove the reference to the molecule

    def bond_plus(self):
        for position, neighbor_posn, direction in self.bond_plus_pairs:
            # Identify the molecule on each bonder (may be same, doesn't matter for now)
            molecule_A = self.get_molecule(position)
            if molecule_A is None:
                continue

            molecule_B = self.get_molecule(neighbor_posn)
            if molecule_B is None:
                continue

            atom_A = molecule_A[position]

            # If the bond being increased is already at the max bond size of 3, don't do
            # anything. However, due to weirdness of Spacechem's bonding algorithm, we still
            # mark the molecule as modified below
            if direction not in atom_A.bonds or atom_A.bonds[direction] != 3:
                atom_B = molecule_B[neighbor_posn]

                # Do nothing if either atom is at (or above) its bond limit (spacechem does not mark any molecules as
                # modified in this case unless the bond was size 3)
                if (sum(atom_A.bonds.values()) >= atom_A.element.max_bonds
                        or sum(atom_B.bonds.values()) >= atom_B.element.max_bonds):
                    continue

                direction_B = direction.opposite()

                if direction not in atom_A.bonds:
                    atom_A.bonds[direction] = 0
                atom_A.bonds[direction] += 1
                if direction_B not in atom_B.bonds:
                    atom_B.bonds[direction_B] = 0
                atom_B.bonds[direction_B] += 1

            if molecule_A is molecule_B:
                # Mark molecule as modified by popping it to the back of the reactor's queue
                del self.molecules[molecule_A]
                self.molecules[molecule_A] = None  # dummy value
            else:
                # Add the smaller molecule to the larger one (faster), then delete the smaller
                # and mark the larger as modified
                molecules = [molecule_A, molecule_B]
                molecules.sort(key=len)
                molecules[1] += molecules[0]

                # Also make sure that any waldos holding the to-be-deleted molecule are updated
                # to point at the combined molecule
                for waldo in self.waldos:
                    if waldo.molecule is molecules[0]:
                        waldo.molecule = molecules[1]

                del self.molecules[molecules[0]]
                del self.molecules[molecules[1]]
                self.molecules[molecules[1]] = None  # dummy value

    def bond_minus(self):
        for position, _, direction in self.bond_minus_pairs:
            molecule = self.get_molecule(position)

            # Skip if there isn't a molecule with a bond over this pair
            if molecule is None or direction not in molecule[position].bonds:
                continue

            # Now that we know for sure the molecule will be mutated, debond the molecule
            # and check if this broke the molecule in two
            split_off_molecule = molecule.debond(position, direction)

            # Mark the molecule as modified
            del self.molecules[molecule]
            self.molecules[molecule] = None  # Dummy value

            # If a new molecule broke off, add it to the reactor molecules
            if split_off_molecule is not None:
                self.molecules[split_off_molecule] = None  # Dummy value

                # If the molecule got broken apart, ensure any waldos holding it are now holding
                # the correct piece of it
                for waldo in self.waldos:
                    if waldo.molecule is molecule and waldo.position in split_off_molecule:
                        waldo.molecule = split_off_molecule

    def defrag_molecule(self, molecule, posn):
        """Given a molecule that has had some of its bonds broken from the given position, update reactor.molecules
        based on any molecules that broke off. Note that this always at least moves the molecule to the back of the
        priority queue, even if it did not break apart (this should be safe since defrag should only be called when the
        molecule is modified).
        """
        # Update the reactor molecules based on how the molecule broke apart
        del self.molecules[molecule]
        for new_molecule in molecule.defrag(posn):
            self.molecules[new_molecule] = None  # Dummy value

            # Update the references of any waldos that were holding the molecule
            for waldo in self.waldos:
                if waldo.molecule is molecule and waldo.position in new_molecule:
                    waldo.molecule = new_molecule

    def delete_atom_bonds(self, posn):
        """Helper used by fuse and swap to remove all bonds from an atom and break up its molecule if needed.
        If no atom at the given position, does nothing.
        Note that this properly updates all neighbors' molecule priorities, but for performance reasons does not
        necessarily update the atom itself if it had no bonds to start with.
        """
        molecule = self.get_molecule(posn)
        if molecule is None:
            return

        atom = molecule.atom_map[posn]
        if atom.bonds:  # Safe optimization since callers always update the target molecule's priority themselves
            num_bonds = len(atom.bonds)

            # Remove bonds
            for dirn in list(atom.bonds):  # Copy so we don't mutate what we're iterating on
                neighbor_atom = molecule.atom_map[posn + dirn]
                del atom.bonds[dirn]
                del neighbor_atom.bonds[dirn.opposite()]

            # Skip graph-searching the molecule when we know the atom wasn't a connector
            if num_bonds == 1:
                # Extract the atom into its own molecule
                del molecule.atom_map[posn]
                new_molecule = Molecule(atom_map={posn: atom})
                self.molecules[new_molecule] = None
                # Update the references of any waldos that were holding the atom
                for waldo in self.waldos:
                    if waldo.molecule is molecule and waldo.position == posn:
                        waldo.molecule = new_molecule

                # Update the priority of the old molecule
                del self.molecules[molecule]
                self.molecules[molecule] = None
            else:
                self.defrag_molecule(molecule, posn)

    def reduce_excess_bonds(self, posn):
        """Helper used by fuse and split to reduce bonds on a mutated atom down to its new max count, and break up its
        molecule if needed.
        """
        molecule = self.get_molecule(posn)
        atom = molecule.atom_map[posn]

        excess_bonds = sum(atom.bonds.values()) - atom.element.max_bonds
        max_bond_size = max(atom.bonds.values(), default=0)
        bonds_broke = False
        neighbor_atoms = {}  # So we don't have to repeatedly incur the get_molecule cost
        while excess_bonds > 0:
            # The order here is deliberately hardcoded to match empirical observations of SpaceChem's behavior
            for dirn in (RIGHT, LEFT, UP, DOWN):
                # Reduce triple bonds first, then double bonds, etc.
                if dirn in atom.bonds and atom.bonds[dirn] == max_bond_size:
                    if dirn not in neighbor_atoms:
                        neighbor_posn = posn + dirn
                        neighbor_atoms[dirn] = self.get_molecule(neighbor_posn)[neighbor_posn]

                    atom.bonds[dirn] -= 1
                    neighbor_atoms[dirn].bonds[dirn.opposite()] -= 1
                    if atom.bonds[dirn] == 0:
                        del atom.bonds[dirn]
                        del neighbor_atoms[dirn].bonds[dirn.opposite()]
                        bonds_broke = True

                    excess_bonds -= 1
                    if excess_bonds == 0:
                        break

            max_bond_size -= 1

        if bonds_broke:
            # Update the reactor molecules based on how the molecule broke apart (if at all)
            self.defrag_molecule(molecule, posn)
        else:
            # If no bonds broke we can save a little work and just directly mark the molecule as updated
            del self.molecules[molecule]
            self.molecules[molecule] = None  # Dummy value

    def fuse(self):
        for left_posn in self.fusers:
            left_molecule = self.get_molecule(left_posn)
            if left_molecule is None:
                continue

            right_posn = left_posn + RIGHT
            right_molecule = self.get_molecule(right_posn)
            if right_molecule is None:
                continue

            left_atom = left_molecule[left_posn]
            right_atom = right_molecule[right_posn]

            # If the target atoms can't be legally fused, do nothing
            fused_atomic_num = left_atom.element.atomic_num + right_atom.element.atomic_num
            if fused_atomic_num > 109:
                continue

            # Remove all bonds from the left atom
            self.delete_atom_bonds(left_posn)

            # Delete the left molecule (now just a single atom). Note that the molecule handle will have changed after
            # delete_atom_bonds. The right atom may be part of a new molecule but its handle shouldn't have changed
            left_molecule = self.get_molecule(left_posn)
            for waldo in self.waldos:
                if waldo.molecule is left_molecule:
                    waldo.molecule = None
            del self.molecules[left_molecule]

            # Update the right atom's element, reducing its bonds as needed
            right_atom.element = elements_dict[fused_atomic_num]
            self.reduce_excess_bonds(right_posn)

    def split(self):
        for splitter_posn in self.splitters:
            split_molecule = self.get_molecule(splitter_posn)
            if split_molecule is None:
                continue

            split_atom = split_molecule[splitter_posn]
            if split_atom.element.atomic_num <= 1:
                continue

            # Split the left atom
            new_atomic_num = split_atom.element.atomic_num // 2
            split_atom.element = elements_dict[split_atom.element.atomic_num - new_atomic_num]
            self.reduce_excess_bonds(splitter_posn)  # Reduce the left atom's bonds if its new bond count is too low

            # Lastly create the new molecule (and check for collisions in its cell)
            new_molecule = Molecule(atom_map={splitter_posn + RIGHT: Atom(element=elements_dict[new_atomic_num])})
            self.check_molecule_collisions_lazy(new_molecule)
            self.molecules[new_molecule] = None  # Dummy value

    def swap(self):
        """Swap atoms between swappers. Note that the order of operations here was carefully chosen to modify the
        internal priority order of reactor molecules the same way that SpaceChem does.
        """
        # Debond all atoms on swappers from their neighbors
        for posn in self.swappers:
            self.delete_atom_bonds(posn)  # Does nothing if no atom on the swapper

        # Swap the atoms, ensuring that waldos don't drop if their held atom is replaced
        # Make sure we get all the molecules to be swapped before we mess up get_molecule by moving them
        for i, (posn, molecule) in enumerate([(p, self.get_molecule(p)) for p in self.swappers]):
            next_posn = self.swappers[(i + 1) % len(self.swappers)]

            if molecule is not None:
                # Update the molecule's atom position and move it to the back of the reactor priority queue
                molecule.atom_map[next_posn] = molecule.atom_map[posn]
                del molecule.atom_map[posn]
                del self.molecules[molecule]
                self.molecules[molecule] = None  # Dummy value

            # If there are any waldos holding something on the next swapper, update their contents
            for waldo in self.waldos:
                if waldo.position == next_posn and waldo.molecule is not None:
                    waldo.molecule = molecule  # May be None, which handles the no molecule case correctly

    def reset(self):
        super().reset()

        self.molecules = {}

        for waldo in self.waldos:
            waldo.reset()

        return self


# Not subclassed off component since bosses don't have in/out pipes nor do they necessarily need a posn.
class Boss:
    """Boss of a defense level."""
    __slots__ = 'hp',
    # The cycle upon which the boss destroys the player base (independent of dmg received)
    # In other words if the last hit is done on LOSS_CYCLE, the player will lose first
    LOSS_CYCLE = math.inf
    MAX_HP = 1
    DEATH_ANIMATION_CYCLES = 0  # Extra cycles tacked on to the final score

    def __new__(cls, _type):
        """Convert to the specific boss subclass."""
        if _type == 'Gorgathar':
            return super().__new__(Gorgathar)
        elif _type == 'Yarugolek':
            return super().__new__(Yarugolek)
        elif _type == 'Xothothor':
            return super().__new__(Xothothor)
        else:
            raise ValueError(f"Invalid boss type `{_type}`")

    def __init__(self, _type):
        self.hp = self.MAX_HP

    def do_instant_actions(self, cycle):
        """Raise a DeathError if LOSS_CYCLE has been reached."""
        if cycle == self.LOSS_CYCLE:
            raise DeathError("The planet has been destroyed.")

    def take_damage(self, dmg, cycle):
        """Take the specified damage, returning True if hp has hit 0."""
        self.hp = max(self.hp - dmg, 0)

        return self.hp <= 0

    def reset(self):
        """Reset this boss to its starting state."""
        self.hp = self.MAX_HP
        return self


class Gorgathar(Boss):
    """Sikutar. No special properties."""
    __slots__ = ()
    LOSS_CYCLE = 6979
    DEATH_ANIMATION_CYCLES = 1452


class Yarugolek(Boss):
    """Hephaestus IV. Moves back and forth; can only be hit at certain times."""
    __slots__ = ()
    LOSS_CYCLE = 5446
    MAX_HP = 5
    DEATH_ANIMATION_CYCLES = 701  # TODO: This is sus, surely it should be 700

    def take_damage(self, dmg, cycle):
        # Boss is only vulnerable for two windows while crossing back and forth
        if 184 <= (cycle % 1821) < 420 or 847 <= (cycle % 1821) <= 1377:  # Measured empirically
            return super().take_damage(dmg, cycle)


class Xothothor(Boss):
    """Flidais. Like Quororque, opens and closes eye periodically, but also can only be damaged by the correct
    colour of laser, rotating colours once damaged.
    Unlike Quororque, only fires a laser if it was not hit first.
    """
    __slots__ = 'eye_color', '_last_cycle_damaged', 'dmg_dealt'
    LOSS_CYCLE = math.inf  # Not constant so we'll do real checks
    MAX_HP = 5
    DEATH_ANIMATION_CYCLES = 702  # I feel like this should be 701 like Yarugolek, but...
    STUN_DURATION = 201  # How many cycles after being damaged the boss restarts its animation

    def __init__(self, _type):
        super().__init__(_type)
        self.eye_color = 0  # Red -> Green -> Blue but we can just index
        self._last_cycle_damaged = -self.STUN_DURATION  # This ensures the boss starts unstunned
        self.dmg_dealt = 0  # How many times the boss attacked the control center

    def do_instant_actions(self, cycle):
        """Fire a laser if on correct cycle, and raise DeathError if this was the 5th hit."""
        # Make sure animation cycle isn't negative while stunned so modular math works
        animation_cycle = max(cycle - self._last_cycle_damaged - self.STUN_DURATION, 0)
        if animation_cycle % 851 == 701:  # 150 cycles after the eye opens
            self.dmg_dealt += 1
            if self.dmg_dealt >= 5:
                raise DeathError("The planet has been destroyed.")

    def take_damage(self, dmg, cycle, color=None):
        animation_cycle = max(cycle - self._last_cycle_damaged - self.STUN_DURATION, 0)
        if 551 <= animation_cycle % 851 <= 800 and color == self.eye_color:
            self.eye_color = (self.eye_color + 1) % 3
            self._last_cycle_damaged = cycle
            return super().take_damage(dmg, cycle)

    def reset(self):
        super().reset()
        self.eye_color = 0
        self._last_cycle_damaged = -self.STUN_DURATION
        self.dmg_dealt = 0
        return self


# Not sub-classed from Output since not all weapons behave like an output
class Weapon(Component):
    """Output-like component used in defense levels."""
    # Can't use __slots__ since python hates multiple inheritance from two classes having (non-empty) __slots__,
    # and More than Machine necessitates a Reactor/Weapon class. We also need boss so can't have empty slots.
    DEFAULT_SHAPE = (3, 3)

    def __new__(cls, component_dict, _type=None, **kwargs):
        """Convert to the specific weapon subclass based on component name."""
        _type = component_dict['type'] if _type is None else _type
        if _type == 'drag-weapon-nuclearmissile':
            return object.__new__(NuclearMissile)
        elif _type == 'drag-superlaser-reactor':
            return object.__new__(SuperLaserReactor)
        elif _type == 'drag-weapon-chemicallaser':
            return object.__new__(ChemicalLaser)
        elif _type == 'drag-weapon-consumer':
            return object.__new__(InternalStorageTank)
        elif _type == 'drag-weapon-canister':
            return object.__new__(CrashCanister)
        else:
            raise ValueError(f"Invalid weapon type `{component_dict['type']}`")

    def __init__(self, component_dict, _type=None, posn=None, **kwargs):
        super().__init__(component_dict, _type=_type, posn=posn, **kwargs)
        self.boss = None


# TODO: Might be nice if the internal outputs were exposed to Solution.outputs somehow
class NuclearMissile(Weapon):
    """Sikutar. This is effectively just a multi-output component."""

    def __init__(self, component_dict, _type=None, posn=None):
        super().__init__(component_dict, _type=_type, posn=posn, num_in_pipes=3)

        self._outputs = [Output(mol_dict, _type='drag-research-output', posn=(0, 0))  # Dummy values
                         for mol_dict in component_dict['molecules']]

    def do_instant_actions(self, cycle):
        for i, output in enumerate(self._outputs):
            output.in_pipe = self.in_pipes[i]  # Fuck it, connect the internal pipes at runtime
            output.do_instant_actions(cycle)  # We'll skip the return value for simplicity

        if all(output.current_count >= output.target_count for output in self._outputs):
            return self.boss.take_damage(1, cycle)

    def reset(self):
        super().reset()

        for output in self._outputs:
            output.reset()

        return self


class SuperLaserReactor(Reactor, Weapon):
    """Hephaestus IV. Special defense level reactor which has a large output zone in the bottom four rows, and flushes
    it when a specified molecule appears in a special third input pipe. Beta input is in top right instead.
    """
    __slots__ = 'discharge_molecule', 'gain_medium_molecule', 'cooling_until',
    COOLDOWN_CYCLES = 250

    def __init__(self, component_dict, _type=None, posn=None):
        super().__init__(component_dict, _type=_type, posn=posn, _num_in_pipes=3, _num_out_pipes=0)
        self.disallowed_instrs.add('instr-output')

        self.discharge_molecule = Molecule.from_json_string(component_dict['discharge-molecule'])
        self.gain_medium_molecule = Molecule.from_json_string(component_dict['gain-molecule'])
        self.cooling_until = 0  # The earliest cycle the next discharge can be performed on

    def reset(self):
        super().reset()
        self.cooling_until = 0
        return self

    def do_instant_actions(self, cycle):
        # TODO: Pause command suggests this goes after discharge, but sensor suggest it goes before
        super().do_instant_actions(cycle)  # Reactor's actions

        # If at least 250 cycles have passed since the last discharge cycle, check if the third input pipe has
        # a molecule, and if it does, validate that it's the discharge molecule, then discharge the output zone
        if cycle >= self.cooling_until and self.in_pipes[2] is not None:
            molecule = self.in_pipes[2].pop(cycle)
            if molecule is not None:
                if not molecule.isomorphic(self.discharge_molecule):
                    raise InvalidOutputError(f"Invalid discharge molecule; expected:\n{self.discharge_molecule}\n\nbut got:\n{molecule}")

                self.cooling_until = cycle + self.COOLDOWN_CYCLES
                return self.discharge(cycle)

    def discharge(self, cycle):
        """Clear all molecules from the output zone and fire the laser once for every ruby crystal therein."""
        # Pre-evaluated to avoid iterating over self.molecules while deleting from it
        molecules_in_zone = [molecule for molecule in self.molecules
                             # Ignore grabbed molecules
                             if not any(waldo.molecule is molecule for waldo in self.waldos)
                             and all(posn.row >= 4 for posn in molecule.atom_map)]

        # Clear all molecules from the zone and fire the laser once for each gain medium molecule (could be multiple)
        for m in molecules_in_zone:
            del self.molecules[m]
            if m.isomorphic(self.gain_medium_molecule):
                # Fire the laser
                if self.boss.take_damage(1, cycle=cycle):  # Returns True if the boss died
                    return True  # Check if we killed the boss

    # Override Reactor's input method to account for the shifted beta input
    def input(self, waldo, input_idx, cycle):
        # If there is no such pipe or it has no molecule available, stall the waldo
        if (input_idx > 1  # Quick hack to prevent the third input being abused by a sub-type 2 input instruction
                or self.in_pipes[input_idx] is None
                or self.in_pipes[input_idx].get(-1, cycle) is None):
            waldo.is_stalled = True
            return

        # Grab the molecule from the appropriate pipe or stall if no such molecule (or no pipe)
        new_molecule = self.in_pipes[input_idx].pop(cycle)

        sample_posn = next(iter(new_molecule.atom_map))
        # Normalize the molecule's column positions if necessary, depending on the input type and source columns
        if input_idx == 0 and sample_posn.col >= 6:  # Ensure alpha input is on the left
            new_molecule.move(LEFT, 6)
        elif input_idx == 1 and sample_posn.col < 6:  # Ensure beta input is on the right
            new_molecule.move(RIGHT, 6)

        # Normalize the molecule's row positions if necessary, depending on the source rows
        if sample_posn.row >= 4:  # Both inputs in this reactor want to be in the top half
            new_molecule.move(UP, 4)

        self.molecules[new_molecule] = None  # Dummy value

        self.check_molecule_collisions_lazy(new_molecule)


class ChemicalLaser(Weapon):
    """Accepts multiple molecules and fires a laser of corresponding color."""
    COOLDOWN_CYCLES = 200

    def __init__(self, component_dict, _type=None, posn=None):
        super().__init__(component_dict, _type=_type, posn=posn, num_in_pipes=1)

        self.molecules = [Molecule.from_json_string(mol_str) for mol_str in component_dict['molecules']]
        self.cooling_until = 0  # The earliest cycle the laser can be fired again on

    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def do_instant_actions(self, cycle):
        if cycle >= self.cooling_until and self.in_pipe is not None:
            molecule = self.in_pipe.pop(cycle)
            if molecule is not None:
                for color, target_molecule in enumerate(self.molecules):
                    if molecule.isomorphic(target_molecule):
                        self.cooling_until = cycle + self.COOLDOWN_CYCLES
                        return self.boss.take_damage(1, cycle=cycle, color=color)

                raise InvalidOutputError("An invalid molecule was passed to Chemical Laser\n"
                                         + f"Produced:\n{molecule}\nbut expected:\n"
                                         + "\nor\n".join(str(m) for m in self.molecules))

    def reset(self):
        super().reset()
        self.cooling_until = 0
        return self


class InternalStorageTank(Weapon, Output):
    """Collapsar. While its component name indicates it's a weapon, it's effectively a 0-count output."""
    __slots__ = ()
    DEFAULT_SHAPE = (2, 3)


class CrashCanister(Weapon, Output):
    """Collapsar. While it's categorized as a weapon, it's effectively just an output with some end delay."""
    __slots__ = 'canister_drop_cycle',
    DEFAULT_SHAPE = (4, 4)

    def __init__(self, component_dict, *args, **kwargs):
        super().__init__(component_dict, *args, **kwargs)
        self.canister_drop_cycle = None

    # For most defense levels, the animation at the end halts all waldos, causing them to repeat the last cycle until
    # the animation completes
    # However in Collapsar, when the output is complete it stops accepting molecules, while the canister drops, but the
    # solution keeps running as normal and must not crash due to any clogs that result.
    # To simulate this, instead of adding a fixed value to the end cycle count like we do for other defense levels,
    # have the canister not mark itself as complete until 2000 cycles after the 40th output
    def do_instant_actions(self, cycle):
        # Behave like a normal output until complete
        if self.canister_drop_cycle is None:
            if super().do_instant_actions(cycle):
                # Once complete, save the cycle and artificially lower the complete count to ensure the solution can't
                # end even if another output completes (since we're treating the hydrogen tank as a 0-count output)
                self.current_count -= 1
                self.canister_drop_cycle = cycle
        elif cycle == self.canister_drop_cycle + 2000:
            # Once the target count is met, stop processing inputs and wait for 2000 cycles before setting the target
            # count back to its true value and indicating to the caller that the output just completed
            self.current_count += 1
            return True

    def reset(self):
        super().reset()

        self.canister_drop_cycle = None

        return self

# TODO: Implement ControlCenter/'drag-defense-plant' so its remaining HP can be shown in debug view
