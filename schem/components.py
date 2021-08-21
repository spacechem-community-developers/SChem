#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import copy
import itertools
import math

from .elements import elements_dict
from .exceptions import *
from .grid import *
from .molecule import Molecule, Atom, ATOM_RADIUS
from .schem_random import SChemRandom
from .waldo import Waldo, Instruction, InstructionType


# Dimensions of component types
# TODO: Refactor so default dimensions and allowable variations on type/shape are built-in to each subclass of Component
#       E.g. Component.DEFAULT_DIMENSIONS = (2, 3), Recycler.DEFAULT_DIMENSIONS = (5, 5)
#       Then we should be able to avoid type string manipulations ('output' in type) and only need to specify the types
#        that change their shape from the class' default
COMPONENT_SHAPES = {
    # SpaceChem stores co-ordinates as col, row
    'research-input': (1, 1),
    'research-output': (1, 1),
    'disabled-output': (1, 1),
    'reactor': (4, 4),
    'output': (2, 3),  # All production level outputs appear to be 2x3
    'drag-recycler': (5, 5),
    'drag-storage-tank': (3, 3),
    'drag-storage-tank-infinite': (3, 3),
    'freeform-counter': (2, 3),
    'drag-arbitrary-input': (2, 3),
    'drag-silo-input': (5, 5),
    'drag-atmospheric-input': (2, 2),
    'drag-oceanic-input': (2, 2),
    'drag-powerplant-input': (14, 15),
    'drag-mining-input': (3, 2),
    'drag-ancient-input': (2, 2),
    'drag-spaceship-input': (2, 2),  # TODO: Actually (2,3) but its pipe isn't in the middle which fucks our assumptions
    'drag-programmed-input': (2, 3),
    'drag-qpipe-in': (3, 1),
    'drag-qpipe-out': (3, 1),
    'drag-printer-output': (2, 3),
    'drag-printer-passthrough': (2, 3)}

# Production level codes don't specify available reactor properties like research levels; encode them here
REACTOR_TYPES = {
    'drag-starter-reactor': {'bonder-count': 4},
    'drag-disassembly-reactor': {'bonder-minus-count': 4, 'has-bottom-input': False},
    'drag-assembly-reactor': {'bonder-plus-count': 4, 'has-bottom-output': False},
    'drag-advanced-reactor': {'bonder-count': 4, 'has-sensor': True},
    'drag-fusion-reactor': {'bonder-count': 4, 'has-fuser': True},
    'drag-superbonder-reactor': {'bonder-count': 8},
    'drag-nuclear-reactor': {'bonder-count': 4, 'has-fuser': True, 'has-splitter': True},
    'drag-quantum-reactor': {'bonder-count': 4, 'has-sensor': True, 'has-teleporter': True,
                             'quantum-walls-y': {5: [0, 1, 2, 3, 4, 5, 6, 7]}},
    'drag-sandbox-reactor': {'bonder-count': 8, 'has-sensor': True, 'has-fuser': True, 'has-splitter': True}}


class Pipe(list):
    """A SpaceChem component's pipe. All posns are relative to the parent component's posn."""
    __slots__ = 'posns',

    def __init__(self, posns):
        '''Construct a pipe. posns should be defined relative to the pipe's parent component posn.'''
        super().__init__([None for _ in posns])
        self.posns = posns

    def move_contents(self):
        '''Shift all molecules in this pipe forward one if possible.'''
        # Iterate from the back to make clog-handling simple
        for i in range(len(self) - 2, -1, -1):  # Note that we don't need to shift the last element
            if self[i + 1] is None:
                self[i], self[i + 1] = None, self[i]

    @classmethod
    def from_preset_string(cls, start_posn, dirns_str):
        '''Construct a pipe from the given CE pipe string, e.g. 'RRDRUULR', moving in the indicated directions
        (U = Up, R = Right, D = Down, L = Left) from the start_posn (should be relative to the parent component's posn).
        '''
        posns = [start_posn]
        char_to_dirn = {'U': UP, 'R': RIGHT, 'D': DOWN, 'L': LEFT}
        for dirn_char in dirns_str:
            posns.append(posns[-1] + char_to_dirn[dirn_char])

        assert len(posns) == len(set(posns)), "Pipe overlaps with itself"

        return Pipe(posns)

    @classmethod
    def from_export_str(cls, export_str):
        '''Note that a pipe's solution lines might not be contiguous. It is expected that the caller filters
        out the lines for a single pipe and passes them as a single string to this method.
        '''
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

    def export_str(self, pipe_idx=0):
        '''Represent this pipe in solution export string format.'''
        return '\n'.join(f'PIPE:{pipe_idx},{posn.col},{posn.row}' for posn in self.posns)


class Component:
    '''Informal Interface class defining methods overworld objects will implement one or more of.'''
    __slots__ = 'type', 'posn', 'dimensions', 'in_pipes', 'out_pipes'

    def __new__(cls, component_dict=None, _type=None, **kwargs):
        '''Return a new object of the appropriate subclass based on the component type.'''
        # If this is being called from a child class, behave like a normal __new__ implementation (to avoid recursion)
        if cls != Component:
            return object.__new__(cls)

        if _type is None:
            _type = component_dict['type']

        parts = _type.split('-')
        if 'reactor' in parts:
            return super().__new__(Reactor)
        elif 'input' in parts:
            return super().__new__(Input)
        elif _type == 'drag-printer-output':
            return super().__new__(OutputPrinter)
        elif _type == 'drag-printer-passthrough':
            return super().__new__(PassThroughPrinter)
        elif 'woncay' in parts:
            return super().__new__(WoncayOutput)
        elif 'output' in parts or 'production-target' in _type:
            return super().__new__(Output)
        elif _type == 'drag-recycler':
            return super().__new__(Recycler)
        elif _type == 'drag-storage-tank':
            return super().__new__(StorageTank)
        elif _type == 'drag-storage-tank-infinite':
            return super().__new__(InfiniteStorageTank)
        elif _type == 'freeform-counter':
            return super().__new__(PassThroughCounter)
        elif _type == 'drag-qpipe-in':
            return super().__new__(TeleporterInput)
        elif _type == 'drag-qpipe-out':
            return super().__new__(TeleporterOutput)
        else:
            raise ValueError(f"Unrecognized component type {_type}")

    def __init__(self, component_dict=None, _type=None, posn=None, num_in_pipes=0, num_out_pipes=0):
        self.type = _type if _type is not None else component_dict['type']
        self.posn = Position(*posn) if posn is not None else Position(col=component_dict['x'], row=component_dict['y'])
        self.in_pipes = [None for _ in range(num_in_pipes)]

        # Initialize output pipes, accounting for any level-preset pipes
        self.out_pipes = []

        type_parts = self.type.split('-')
        if self.type in COMPONENT_SHAPES:
            self.dimensions = COMPONENT_SHAPES[self.type]
        elif 'output' in type_parts or 'woncay' in type_parts or 'production-target' in self.type:
            self.dimensions = COMPONENT_SHAPES['output']
        elif 'reactor' in type_parts:
            self.dimensions = COMPONENT_SHAPES['reactor']
        else:
            raise ValueError(f"Dimensions of component {self.type} are unknown")

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

    @classmethod
    def parse_metadata(cls, s):
        '''Given a component export string or its COMPONENT line, return its component type and posn.'''
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
        '''Given a matching export string, update this component. Optionally ignore pipe updates (namely necessary
        for Ω-Pseudoethyne which disallows mutating a 1-long pipe where custom levels do not.
        '''
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
        ''''Do any instant actions (e.g. execute waldo instructions, spawn/consume molecules).'''
        return

    def move_contents(self, _):
        '''Move the contents of this object (e.g. waldos/molecules), including its pipes.'''
        for pipe in self.out_pipes:
            pipe.move_contents()

    def export_str(self):
        '''Represent this component in solution export string format.'''
        # TODO: I'm still not sure what the 4th component field is used for. Custom reactor names maybe?

        return f"COMPONENT:'{self.type}',{self.posn[0]},{self.posn[1]},''\n" \
               + '\n'.join(pipe.export_str(pipe_idx=i) for i, pipe in enumerate(self.out_pipes))


class Input(Component):
    __slots__ = 'molecules', 'input_rate'

    # Convenience property for when we know we're dealing with an Input
    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def __new__(cls, input_dict, *args, **kwargs):
        '''Convert to a random or programmed input if relevant.'''
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

    def move_contents(self, cycle):
        """Move pipe contents, then add a new molecule to pipe if on correct cycle and there is room."""
        super().move_contents(cycle)

        # -1 necessary since starting cycle is 1 not 0, while mod == 1 would break on rate = 1
        if (cycle - 1) % self.input_rate == 0 and self.out_pipe[0] is None:
            self.out_pipe[0] = copy.deepcopy(self.molecules[0])


class RandomInput(Input):
    __slots__ = 'random_generator', 'input_counts', 'random_bucket'

    def __init__(self, input_dict, _type=None, posn=None, is_research=False):
        super().__init__(input_dict, _type=_type, posn=posn, is_research=is_research)

        assert len(self.molecules) > 1, "Fixed input passed to RandomInput ctor"

        # Create a random generator with the given seed. Most levels default to seed 0
        seed = input_dict['random-seed'] if 'random-seed' in input_dict else 0
        self.random_generator = SChemRandom(seed=seed)
        self.random_bucket = []  # Bucket of indices for the molecules in the current balancing bucket

        molecules_key = 'inputs' if 'inputs' in input_dict else 'molecules'
        self.input_counts = [input_mol_dict['count'] for input_mol_dict in input_dict[molecules_key]]

    def get_next_molecule_idx(self):
        '''Get the next input molecule's index. Exposed to allow for tracking branches in random level states.'''
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
        """Move pipe contents, then add a new molecule to pipe if on correct cycle and there is room."""
        super(Input, self).move_contents(cycle)

        # -1 necessary since starting cycle is 1 not 0, while mod == 1 would break on rate = 1
        if (cycle - 1) % self.input_rate == 0 and self.out_pipe[0] is None:
            self.out_pipe[0] = copy.deepcopy(self.molecules[self.get_next_molecule_idx()])


class ProgrammedInput(Input):
    __slots__ = 'starting_molecules', 'repeating_molecules', 'repeating_idx'

    def __init__(self, input_dict, _type=None, posn=None, is_research=False):
        super(Input, self).__init__(input_dict, _type=_type, posn=posn, num_out_pipes=1)

        assert len(input_dict['repeating-molecules']) != 0, "No repeating molecules in input dict"
        self.starting_molecules = [Molecule.from_json_string(s) for s in input_dict['starting-molecules']]
        self.repeating_molecules = [Molecule.from_json_string(s) for s in input_dict['repeating-molecules']]
        self.repeating_idx = 0

        if is_research:
            self.input_rate = 1
        elif 'production-delay' in input_dict:
            self.input_rate = input_dict['production-delay']
        else:
            self.input_rate = 10

    def move_contents(self, cycle):
        """Move pipe contents, then add a new molecule to pipe if on correct cycle and there is room."""
        super(Input, self).move_contents(cycle)

        # -1 necessary since starting cycle is 1 not 0, while mod == 1 would break on rate = 1
        if (cycle - 1) % self.input_rate == 0 and self.out_pipe[0] is None:
            if not self.starting_molecules:
                self.out_pipe[0] = copy.deepcopy(self.repeating_molecules[self.repeating_idx])
                self.repeating_idx = (self.repeating_idx + 1) % len(self.repeating_molecules)
            else:
                self.out_pipe[0] = copy.deepcopy(self.starting_molecules.pop(0))


class Output(Component):
    __slots__ = 'output_molecule', 'target_count', 'current_count'

    # Convenience property for when we know we're dealing with an Output
    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def __init__(self, output_dict, _type=None, posn=None):
        super().__init__(output_dict, _type=_type, posn=posn, num_in_pipes=1)

        # CE output components are abstracted one level higher than vanilla output zones; unwrap if needed
        if 'output-target' in output_dict:
            output_dict = output_dict['output-target']

        self.output_molecule = Molecule.from_json_string(output_dict['molecule'])
        self.target_count = output_dict['count']
        self.current_count = 0

    def do_instant_actions(self, _):
        '''Check for and process any incoming molecule, and return True if this output just completed (in which case
        the caller should check if the other outputs are also done). This avoids checking all output counts every cycle.
        '''
        if self.in_pipe is None:
            return False

        molecule = self.in_pipe[-1]
        if molecule is not None:
            if not molecule.isomorphic(self.output_molecule):
                raise InvalidOutputError(f"Invalid output molecule; expected:\n{self.output_molecule}\n\nbut got:\n{molecule}")

            self.in_pipe[-1] = None

            if self.current_count < self.target_count:
                self.current_count += 1

            if self.current_count == self.target_count:
                return True

        return False


class WoncayOutput(Output):
    __slots__ = 'matching_input',

    def __init__(self, output_dict):
        super().__init__(output_dict)

        self.output_molecule = None  # Will be updated dynamically

        # Create an equivalent copy of the random input for calculating the corresponding outputs
        self.matching_input = RandomInput({
            "molecules": [{"molecule": "Titanium;Ti;302200", "count": 180},
                          {"molecule": "Ruthenium;Ru;304400", "count": 180}],
            "random-seed": 0,
            "production-delay": 1,
            "output-pipes": ["R"],
            "type": "drag-arbitrary-input", "x": 3, "y": 6})

    def do_instant_actions(self, cycle):
        '''Check for and process any incoming molecule, and return True if this output just completed (in which case
        the caller should check if the other outputs are also done). This avoids checking all output counts every cycle.
        '''
        if self.in_pipe is None or self.in_pipe[-1] is None:
            return False

        # Fetch the next 9 atoms from the simulated input, and calculate the expected output molecule accordingly
        input_bits = [self.matching_input.get_next_molecule_idx() for _ in range(9)]
        # Map the bits to the position they ended up in based on the first reactor construction:
        # 9 8 7
        # 4 5 6
        # 3 2 1
        input_bits_block = [input_bits[8:5:-1], input_bits[3:6], input_bits[2::-1]]

        # Apply Woncay's rules to calculate the output bits
        output_bits_block = [[], [], []]
        for r in range(3):
            for c in range(3):
                neighbors_sum = sum([input_bits_block[r - 1][c] if r > 0 else 0,
                                     input_bits_block[r + 1][c] if r < 2 else 0,
                                     input_bits_block[r][c - 1] if c > 0 else 0,
                                     input_bits_block[r][c + 1] if c < 2 else 0])

                if neighbors_sum == 2:  # Reproduce
                    output_bits_block[r].append(1)
                elif neighbors_sum == 3:  # Propagate
                    output_bits_block[r].append(input_bits_block[r][c])
                else:  # Die
                    output_bits_block[r].append(0)

        def bit_block_to_molecule(bit_block):
            """Helper function to convert a bit block into a molecule."""
            # Create a titanium block and update it based on all the live bits
            molecule = Molecule.from_json_string("Titanium;Ti;002211;102211;202201;012211;112211;212201;022210;122210;222200")

            for r in range(3):
                for c in range(3):
                    if bit_block[r][c]:
                        molecule[(c, r)].element = elements_dict[44]

            return molecule

        # Update the expected molecule, then run the normal output check
        self.output_molecule = bit_block_to_molecule(output_bits_block)

        try:
            return super().do_instant_actions(cycle)
        except InvalidOutputError as e:
            # Friendlier error
            input_molecule = bit_block_to_molecule(input_bits_block)
            raise InvalidOutputError(f"For original input block:\n{input_molecule}\n\n{e}")


class PassThroughCounter(Output):
    __slots__ = 'stored_molecule',

    def __init__(self, output_dict):
        super(Output, self).__init__(output_dict, num_in_pipes=1, num_out_pipes=1)

        self.output_molecule = Molecule.from_json_string(output_dict['target']['molecule'])
        self.target_count = output_dict['target']['count']
        self.current_count = 0

        self.stored_molecule = None

    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def do_instant_actions(self, cycle):
        '''Check for and process any incoming molecule, and return True if this output just completed (in which case
        the caller should check if the other outputs are also done). This avoids checking all output counts every cycle.
        '''
        if self.in_pipe is None:
            return False

        # TODO: I thought from eyeballing it that a molecule should go in and out of a pass-through counter
        #       within the same cycle, but it seems I had to put this block first to avoid an off-by-1-cycle error.
        #       Double-check what's going on here, I think this may diverge from SC when the pass-through's pipe gets
        #       clogged
        #       Probably because it can't be teleported in the same cycle or it will be moved that cycle too, which
        #       effectively makes it move twice
        # If there is a molecule stored (possibly stored just now), put it in the output pipe if possible
        if self.stored_molecule is not None and self.out_pipe[0] is None:
            self.stored_molecule, self.out_pipe[0] = None, self.stored_molecule

        # If the stored slot is empty, store the next molecule and 'output' it while we do so
        if self.in_pipe[-1] is not None and self.stored_molecule is None:
            self.stored_molecule = self.in_pipe[-1]
            return super().do_instant_actions(cycle)  # This will set self.in_pipe[-1] to None

        return False


# It's less confusing for output counting and user-facing purposes if this is not an Output subclass
class DisabledOutput(Component):
    '''Used by research levels, which actually crash if a wrong output is used unlike assembly reactors.'''
    __slots__ = ()

    @property
    def in_pipe(self):
        return self.in_pipes[0]

    def __init__(self, *, _type, posn):
        super().__init__(_type=_type, posn=posn, num_in_pipes=1)

    def do_instant_actions(self, _):
        # Technically should check for `in_pipe is None` first but I'd also be curious to see this crash since disabled
        # outputs are only used in research levels, where it should be impossible to not connect to the disabled output
        if self.in_pipe[-1] is not None:
            raise InvalidOutputError("A molecule was passed to a disabled output.")


class OutputPrinter(Component):
    """Displays the last 3 molecules passed to it. For now this is effectively going to be a recycler..."""
    __slots__ = ()

    def __init__(self, component_dict=None, _type=None, posn=None):
        super().__init__(component_dict, _type=_type, posn=posn, num_in_pipes=1)

    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def do_instant_actions(self, _):
        """Consume and print incoming molecules."""
        if self.in_pipe is not None:
            # TODO: Print received molecules when in --debug somehow
            self.in_pipe[-1] = None


class PassThroughPrinter(OutputPrinter):
    """Displays the last 3 molecules passed to it and passes them on."""
    __slots__ = 'stored_molecule',

    def __init__(self, component_dict=None, _type=None, posn=None):
        super(OutputPrinter, self).__init__(component_dict, _type=_type, posn=posn, num_in_pipes=1, num_out_pipes=1)

        self.stored_molecule = None

    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def do_instant_actions(self, cycle):
        '''Check for and process any incoming molecule, and return True if this output just completed (in which case
        the caller should check if the other outputs are also done). This avoids checking all output counts every cycle.
        '''
        if self.in_pipe is None:
            return

        # If there is a molecule stored (possibly stored just now), put it in the output pipe if possible
        if self.stored_molecule is not None and self.out_pipe[0] is None:
            self.stored_molecule, self.out_pipe[0] = None, self.stored_molecule

        # If the stored slot is empty, store the next molecule and 'print' it while we do so
        if self.in_pipe[-1] is not None and self.stored_molecule is None:
            self.stored_molecule = self.in_pipe[-1]
            super().do_instant_actions(cycle)  # This will consume and print the input molecule


class Recycler(Component):
    __slots__ = ()

    def __init__(self, component_dict=None, _type=None, posn=None):
        super().__init__(component_dict=component_dict, _type=_type, posn=posn, num_in_pipes=3)

    def do_instant_actions(self, _):
        for pipe in self.in_pipes:
            if pipe is not None:
                pipe[-1] = None


# TODO: Ideally this would subclass both deque and Component but doing so gives me
#       "multiple bases have instance lay-out conflict". Need to investigate.
class StorageTank(Component):
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

    def do_instant_actions(self, _):
        if self.in_pipe is None:
            return

        if self.in_pipe[-1] is not None and len(self.contents) < self.MAX_CAPACITY:
            self.contents.append(self.in_pipe[-1])
            self.in_pipe[-1] = None

    def move_contents(self, cycle):
        """Fill the output pipe if the storage tank is not empty. This does not occur in do_instant_actions to reflect
        the fact that a reactor-tank-reactor setup cannot pass a molecule between reactors in the same cycle, unlike
        with two adjacent reactors.
        """
        super().move_contents(cycle)

        if self.out_pipe[0] is None and self.contents:
            self.out_pipe[0] = self.contents.popleft()

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


class InfiniteStorageTank(StorageTank):
    MAX_CAPACITY = math.inf


class TeleporterInput(Component):
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

    def do_instant_actions(self, _):
        '''Note that the teleporter pair behaves differently from a pass-through counter insofar as the pass-through
        counter stores any molecule it receives internally when its output pipe is clogged, whereas the teleporter
        refuses to accept the next molecule until the output pipe is clear (i.e. behaves like a single discontinuous
        pipe that also happens to only allow single atoms through).
        '''
        if self.in_pipe is None:
            return

        molecule = self.in_pipe[-1]
        if molecule is not None and self.destination.out_pipe[0] is None:
            assert len(molecule) == 1, f"An invalid molecule was passed to Teleporter (Input): {molecule}"

            self.in_pipe[-1], self.destination.molecule = None, molecule


class TeleporterOutput(Component):
    # TODO: Needing an internal molecule slot is awkward but I couldn't find a cleaner way to avoid the molecule
    #       being both teleported and moved in the same cycle if the teleporters have the wrong relative component
    #       priorities
    __slots__ = 'destination', 'molecule'

    def __init__(self, component_dict):
        super().__init__(component_dict, num_out_pipes=1)
        self.molecule = None

    # Convenience properties
    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def move_contents(self, cycle):
        # Move pipe contents first, then add the teleported molecule to the front to avoid double-moving it
        super().move_contents(cycle)
        self.molecule, self.out_pipe[0] = None, self.molecule


class Reactor(Component):
    # For convenience during float-precision rotation co-ordinates, we consider the center of the
    # top-left cell to be at (0,0), and hence the top-left reactor corner is (-0.5, -0.5).
    # Further, treat the walls as being one atom radius closer, so that we can efficiently check if an atom will collide
    # with them given only the atom's center co-ordinates
    NUM_COLS = 10
    NUM_ROWS = 8
    NUM_WALDOS = 2
    NUM_MOVE_CHECKS = 10  # Number of times to check for collisions during molecule movement
    walls = {UP: -0.5 + ATOM_RADIUS, DOWN: 7.5 - ATOM_RADIUS,
             LEFT: -0.5 + ATOM_RADIUS, RIGHT: 9.5 - ATOM_RADIUS}
    # Names of features as stored in attributes
    FEATURE_NAMES = ('bonders', 'sensors', 'fusers', 'splitters', 'swappers')

    __slots__ = ('in_pipes', 'out_pipes',
                 'waldos', 'molecules',
                 'large_output', 'bonders', 'sensors', 'fusers', 'splitters', 'swappers',
                 'bond_plus_pairs', 'bond_minus_pairs',
                 'quantum_walls_x', 'quantum_walls_y', 'disallowed_instrs',
                 'debug')

    def __init__(self, component_dict=None, _type=None, posn=None):
        '''Initialize a reactor from only its component dict, doing e.g. default placements of features. Used for
        levels with preset reactors.
        '''
        if component_dict is None:
            component_dict = {}

        # If the reactor type is known, look up its properties and merge them (with lower priority) into the given dict
        _type = _type if _type is not None else component_dict['type']
        if _type in REACTOR_TYPES:
            component_dict = {**REACTOR_TYPES[_type], **component_dict}  # TODO: Use py3.9's dict union operator

        # If the has-bottom attributes are unspecified, they default to True, unlike most attribute flags
        num_in_pipes = 1 if 'has-bottom-input' in component_dict and not component_dict['has-bottom-input'] else 2
        num_out_pipes = 1 if 'has-bottom-output' in component_dict and not component_dict['has-bottom-output'] else 2

        super().__init__(component_dict,
                         _type=_type, posn=posn,
                         num_in_pipes=num_in_pipes, num_out_pipes=num_out_pipes)

        # Place all features
        cur_col = 0  # For simplicity we will put each feature type in its own column(s)

        # Place bonders. Different bonder types go in the same struct so they can share a priority index
        self.bonders = []
        for feature_name, abbrev in (('bonder', '+-'), ('bonder-plus', '+'), ('bonder-minus', '-')):
            if f'{feature_name}-count' in component_dict:
                self.bonders.extend([(Position(cur_col, i), abbrev)
                                     for i in range(component_dict[f'{feature_name}-count'])])
            cur_col += 1

        # Place remaining features
        for attr_name, feature_name, feature_width, default_count in (('sensors', 'sensor', 1, 1),
                                                                      ('fusers', 'fuser', 2, 1),
                                                                      ('splitters', 'splitter', 2, 1),
                                                                      ('swappers', 'teleporter', 1, 2)):
            if f'{feature_name}-count' in component_dict:
                setattr(self, attr_name, [Position(cur_col, i) for i in range(component_dict[f'{feature_name}-count'])])
            elif f'has-{feature_name}' in component_dict and component_dict[f'has-{feature_name}']:
                setattr(self, attr_name, [Position(cur_col, i) for i in range(default_count)])
            else:
                setattr(self, attr_name, [])

            cur_col += feature_width

        # Pre-compute active bond pairs
        self.bond_plus_pairs, self.bond_minus_pairs = self.bond_pairs()

        self.large_output = 'has-large-output' in component_dict and component_dict['has-large-output']

        # Place Waldo starts at default locations
        self.waldos = [Waldo(idx=i, instr_map={Position(4, 1 + 5*i): (None, Instruction(InstructionType.START,
                                                                                        direction=LEFT))})
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

        self.disallowed_instrs = set() if 'disallowed-instructions' not in component_dict else set(component_dict['disallowed-instructions'])

        # Store molecules as dict keys to be ordered (preserving Spacechem's hidden 'least recently modified' rule)
        # and to have O(1) add/delete. Dict values are ignored.
        self.molecules = {}

    def bond_pairs(self):
        '''For each of + and - bond commands, return a tuple of (bonder_A_posn, bonder_B_posn, dirn) triplets,
        sorted in priority order.
        '''
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
        # TODO: usage might be cleaner if separate arrow_maps and instr_maps... but probably more space
        waldo_instr_maps = [{} for _ in range(self.NUM_WALDOS)]  # Can't use * or else dict gets multi-referenced

        feature_posns = set()  # for verifying features were not placed illegally

        # Break the component string up into its individual sections, while removing empty lines
        component_line, *lines = (s for s in export_str.split('\n') if s)

        pipes_idx = next((i for i, s in enumerate(lines) if s.startswith('PIPE:')), len(lines))
        member_lines, lines = lines[:pipes_idx], lines[pipes_idx:]
        if not member_lines:
            raise ValueError("Missing MEMBER lines in reactor component")

        annotations_idx = next((i for i, s in enumerate(lines) if s.startswith('ANNOTATION:')), len(lines))
        pipes_str = '\n'.join(lines[:annotations_idx])

        # Validates COMPONENT line and updates pipes
        super().update_from_export_str(component_line + '\n' + pipes_str, update_pipes=update_pipes)

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

            # Since this member is an instr and not a feature, prep a slot in the instr map
            if position not in waldo_instr_maps[waldo_idx]:
                waldo_instr_maps[waldo_idx][position] = [None, None]

            if member_name == 'instr-arrow':
                assert waldo_instr_maps[waldo_idx][position][0] is None, f"Overlapping arrows at {position}"
                waldo_instr_maps[waldo_idx][position][0] = direction
                continue

            assert waldo_instr_maps[waldo_idx][position][1] is None, f"Overlapping commands at {position}"

            if member_name == 'instr-start':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.START, direction=direction)
                continue

            # Note: Some similar instructions have the same name but are sub-typed by the
            #       second integer field
            instr_sub_type = int(fields[2])
            if member_name == 'instr-input':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.INPUT, target_idx=instr_sub_type)
            elif member_name == 'instr-output':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.OUTPUT,
                                                                       target_idx=instr_sub_type)
            elif member_name == 'instr-grab':
                if instr_sub_type == 0:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.GRAB_DROP)
                elif instr_sub_type == 1:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.GRAB)
                else:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.DROP)
            elif member_name == 'instr-rotate':
                if instr_sub_type == 0:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.ROTATE,
                                                                           direction=Direction.CLOCKWISE)
                else:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.ROTATE,
                                                                           direction=Direction.COUNTER_CLOCKWISE)
            elif member_name == 'instr-sync':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SYNC)
            elif member_name == 'instr-bond':
                if instr_sub_type == 0:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.BOND_PLUS)
                else:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.BOND_MINUS)
            elif member_name == 'instr-sensor':
                # The last CSV field is used by the sensor for the target atomic number
                atomic_num = int(fields[7])
                if atomic_num not in elements_dict:
                    raise Exception(f"Invalid atomic number {atomic_num} on sensor command.")
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SENSE,
                                                                       direction=direction,
                                                                       target_idx=atomic_num)
            elif member_name == 'instr-fuse':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.FUSE)
            elif member_name == 'instr-split':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SPLIT)
            elif member_name == 'instr-swap':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SWAP)
            elif member_name == 'instr-toggle':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.FLIP_FLOP, direction=direction)
            elif member_name == 'instr-debug':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.PAUSE)
            else:
                raise Exception(f"Unrecognized member type {member_name}")

        self.waldos = [Waldo(idx=i, instr_map=waldo_instr_maps[i]) for i in range(self.NUM_WALDOS)]

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

        self.bond_plus_pairs, self.bond_minus_pairs = self.bond_pairs()  # Re-precompute bond pairings based on the changes

    def export_str(self):
        '''Represent this reactor in solution export string format.'''
        export_str = f"COMPONENT:'{self.type}',{self.posn.col},{self.posn.row},''"

        # TODO: Make reactors more agnostic of feature types
        for (posn, bond_types) in self.bonders:
            if bond_types == '+-':
                feature_name = 'bonder'
            elif bond_types == '+':
                feature_name = 'bonder-plus'
            elif bond_types == '-':
                feature_name = 'bonder-minus'
            else:
                raise Exception("Invalid bonder type in internal data")

            export_str += f"\nMEMBER:'feature-{feature_name}',-1,0,1,{posn.col},{posn.row},0,0"
        for posn in self.sensors:
            export_str += f"\nMEMBER:'feature-sensor',-1,0,1,{posn.col},{posn.row},0,0"
        for posn in self.fusers:
            export_str += f"\nMEMBER:'feature-fuser',-1,0,1,{posn.col},{posn.row},0,0"
        for posn in self.splitters:
            export_str += f"\nMEMBER:'feature-splitter',-1,0,1,{posn.col},{posn.row},0,0"
        for posn in self.swappers:
            export_str += f"\nMEMBER:'feature-tunnel',-1,0,1,{posn.col},{posn.row},0,0"

        export_str += '\n' + '\n'.join(waldo.export_str() for waldo in self.waldos)
        export_str += '\n' + '\n'.join(pipe.export_str(pipe_idx=i) for i, pipe in enumerate(self.out_pipes))

        return export_str

    def __hash__(self):
        '''Hash of the current reactor state.'''
        return hash((tuple(molecule.hashable_repr() for molecule in self.molecules),
                     tuple(self.waldos)))

    def __str__(self, flash_features=True, show_instructions=False):
        '''Return a rich-format pretty-print string representing this reactor.'''
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
                for (c, r), (_, cmd) in waldo.instr_map.items():
                    if cmd is not None:
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
                if waldo.position not in waldo.instr_map:
                    continue

                cmd = waldo.instr_map[waldo.position][1]
                if cmd is None:
                    continue

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

    def do_instant_actions(self, _):
        for waldo in self.waldos:
            self.exec_instrs(waldo)

    def move_contents(self, cycle):
        '''Move all waldos in this reactor and any molecules they are holding.'''
        super().move_contents(cycle)

        # If the waldo is facing a wall, mark it as stalled (may also be stalled due to sync, input, etc.)
        for waldo in self.waldos:
            if ((waldo.direction == UP and waldo.position.row == 0)
                    or (waldo.direction == DOWN and waldo.position.row == 7)
                    or (waldo.direction == LEFT and waldo.position.col == 0)
                    or (waldo.direction == RIGHT and waldo.position.col == 9)):
                waldo.is_stalled = True

        # If any waldo is about to rotate a molecule, don't skimp on collision checks
        # Note that a waldo might be marked as rotating (and stalled accordingly) while holding nothing, in the case
        # that red hits a rotate and then has its atom fused or swapped away by blue in the same cycle
        # Hence the waldo.molecule is not None check is necessary
        if any(waldo.is_rotating and waldo.molecule is not None for waldo in self.waldos):
            # If both waldos are holding the same molecule and either of them is rotating, a crash occurs
            # (even if they're in the same position and rotating the same direction)
            if self.waldos[0].molecule is self.waldos[1].molecule:
                raise ReactionError("Molecule pulled apart")

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
                    self.check_collisions_lazy(waldo.molecule)
                elif waldo.is_rotating:
                    waldo.molecule.round_posns()
                    # Rotate atom bonds
                    waldo.molecule.rotate_bonds(waldo.cur_cmd().direction)
        elif any(waldo.molecule is not None and not waldo.is_stalled for waldo in self.waldos):
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
                waldos_moving_molecules = [self.waldos[0]]
            else:
                waldos_moving_molecules = [w for w in self.waldos if not w.is_stalled and w.molecule is not None]
                # (skipped if both waldos holding same molecule)
                # Check if a molecule being moved will bump into the back of another moving molecule
                if (len(waldos_moving_molecules) == 2 and self.waldos[0].direction != self.waldos[1].direction):
                    for waldo in self.waldos:
                        # Intersect the target positions of this waldo's molecule with the current positions of the
                        # other waldo's molecules
                        other_waldo = self.waldos[1 - waldo.idx]
                        target_posns = set(posn + waldo.direction for posn in waldo.molecule.atom_map)
                        if not target_posns.isdisjoint(other_waldo.molecule.atom_map):
                            raise ReactionError("Collision between molecules")

            # Move all molecules
            for waldo in waldos_moving_molecules:
                # If we're moving perpendicular to any quantum walls, check for collisions with them
                if ((self.quantum_walls_y and waldo.direction in (LEFT, RIGHT))
                        or (self.quantum_walls_x and waldo.direction in (UP, DOWN))):
                    # Move the molecule halfway, check for quantum wall collisions, then move the last half
                    waldo.molecule.move(waldo.direction, distance=0.5)
                    self.check_quantum_wall_collisions(waldo.molecule)
                    waldo.molecule.move(waldo.direction, distance=0.5)
                    waldo.molecule.round_posns()
                else:
                    waldo.molecule.move(waldo.direction)

            # Perform collision checks against the moved molecules
            for waldo in self.waldos:
                if waldo.molecule is not None and not waldo.is_stalled:
                    self.check_collisions_lazy(waldo.molecule)

        # Move waldos and mark them as no longer stalled. Note that is_rotated must be left alone to tell it not to
        # rotate twice
        for waldo in self.waldos:
            if not waldo.is_stalled:
                waldo.position += waldo.direction
            waldo.is_stalled = False

    def check_molecule_collisions_lazy(self, molecule):
        '''Raise an exception if the given molecule collides with any other molecules.
        Assumes integer co-ordinates in all molecules.
        '''
        for other_molecule in self.molecules:
            molecule.check_collisions_lazy(other_molecule)  # Implicitly ignores self

    def check_wall_collisions(self, molecule):
        '''Raise an exception if the given molecule collides with any walls.'''
        if not all(self.walls[UP] < p.row < self.walls[DOWN]
                   and self.walls[LEFT] < p.col < self.walls[RIGHT]
                   for p in molecule.atom_map):
            raise ReactionError("A molecule has collided with a wall")

    def check_quantum_wall_collisions(self, molecule):
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

    def check_collisions_lazy(self, molecule):
        '''Raise an exception if the given molecule collides with any other molecules or walls.
        Assumes integer co-ordinates in all molecules.
        '''
        self.check_molecule_collisions_lazy(molecule)
        self.check_wall_collisions(molecule)
        # Quantum wall collision checks may be skipped since they should only lie on grid edges

    def check_collisions(self, molecule):
        '''Check that the given molecule isn't colliding with any walls or other molecules.
        Raise an exception if it is.
        '''
        for other_molecule in self.molecules:
            molecule.check_collisions(other_molecule)  # Implicitly ignores self

        self.check_wall_collisions(molecule)
        self.check_quantum_wall_collisions(molecule)

    def exec_instrs(self, waldo):
        if waldo.position not in waldo.instr_map:
            return

        arrow_direction, cmd = waldo.instr_map[waldo.position]

        # Update the waldo's direction based on any arrow in this cell
        if arrow_direction is not None:
            waldo.direction = arrow_direction

        # Execute the non-arrow instruction
        if cmd is None:
            return
        elif cmd.type == InstructionType.INPUT:
            self.input(waldo, cmd.target_idx)
        elif cmd.type == InstructionType.OUTPUT:
            self.output(waldo, cmd.target_idx)
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

    def input(self, waldo, input_idx):
        # If there is no such pipe or it has no molecule available, stall the waldo
        if (input_idx > len(self.in_pipes) - 1
                or self.in_pipes[input_idx] is None
                or self.in_pipes[input_idx][-1] is None):
            waldo.is_stalled = True
            return

        # Grab the molecule from the appropriate pipe or stall if no such molecule (or no pipe)
        new_molecule = self.in_pipes[input_idx][-1]
        self.in_pipes[input_idx][-1] = None

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

    def output(self, waldo, output_idx):
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
            if self.out_pipes[output_idx][0] is None:
                # Put the molecule in the pipe
                self.out_pipes[output_idx][0] = molecule

                # Look for any other outputable molecule
                molecule = next(molecules_in_zone, None)  # Look for another outputable molecule

                # Remove the outputted molecule from the reactor (make sure not to use the iterator again now!)
                del self.molecules[self.out_pipes[output_idx][0]]

        # If there is any output(s) remaining in this zone (regardless of whether we outputted), stall this waldo
        waldo.is_stalled = molecule is not None

    def get_molecule(self, position):
        '''Select the molecule at the given grid position, or None if no such molecule.
        Used by Grab, Bond+/-, Fuse, etc.
        '''
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
        '''Given a molecule that has had some of its bonds broken from the given position, update reactor.molecules
        based on any molecules that broke off. Note that this always at least moves the molecule to the back of the
        priority queue, even if it did not break apart (this should be safe since defrag should only be called when the
        molecule is modified).
        '''
        # Update the reactor molecules based on how the molecule broke apart
        del self.molecules[molecule]
        for new_molecule in molecule.defrag(posn):
            self.molecules[new_molecule] = None  # Dummy value

            # Update the references of any waldos that were holding the molecule
            for waldo in self.waldos:
                if waldo.molecule is molecule and waldo.position in new_molecule:
                    waldo.molecule = new_molecule

    def delete_atom_bonds(self, posn):
        '''Helper used by fuse and swap to remove all bonds from an atom and break up its molecule if needed.
        If no atom at the given position, does nothing.
        '''
        molecule = self.get_molecule(posn)
        if molecule is None:
            return

        atom = molecule.atom_map[posn]
        for dirn in CARDINAL_DIRECTIONS:
            if dirn in atom.bonds:
                neighbor_atom = molecule.atom_map[posn + dirn]
                del atom.bonds[dirn]
                del neighbor_atom.bonds[dirn.opposite()]

        self.defrag_molecule(molecule, posn)

    def reduce_excess_bonds(self, posn):
        '''Helper used by fuse and split to reduce bonds on a mutated atom down to its new max count, and break up its
        molecule if needed.
        '''
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
        '''Swap atoms between swappers. Note that the order of operations here was carefully chosen to modify the
        internal priority order of reactor molecules the same way that SpaceChem does.
        '''
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
