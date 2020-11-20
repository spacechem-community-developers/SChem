#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import copy

import numpy as np

from spacechem.exceptions import InvalidOutputError
from spacechem.grid import Position
from spacechem.molecule import Molecule
from spacechem.spacechem_random import SpacechemRandom


# Dimensions of component types
COMPONENT_SHAPES = {
    # SpaceChem stores co-ordinates as col, row
    'reactor': (4, 4),
    'output': (2, 3),  # All production level outputs appear to be 2x3
    'recycler': (5, 5),
    'research-output': (1, 1),  # Hacky way to make sure research levels don't have colliding output components
    'research-input': (1, 1),   # ditto
    'drag-arbitrary-input': (2, 3),
    'drag-silo-input': (5, 5),
    'drag-atmospheric-input': (2, 2),
    'drag-oceanic-input': (2, 2),
    'drag-powerplant-input': (14, 15),
    'drag-mining-input': (3, 2)}


class Pipe(list):
    __slots__ = 'posns',

    def __init__(self, posns):
        super().__init__([None for _ in posns])
        self.posns = posns

    def move_contents(self):
        '''Shift all molecules in this pipe forward one if possible.'''
        # Iterate from the back to make clog-handling simple
        for i in range(len(self) - 2, -1, -1):  # Note that we don't need to shift the last element
            if self[i + 1] is None:
                self[i], self[i + 1] = None, self[i]

    @classmethod
    def from_export_str(cls, export_str):
        '''Note that a pipe's solution lines may not be contiguous. It is expected that the caller filters
        out the lines for a single pipe and passes them as a single string to this method.
        '''
        lines = export_str.strip().split('\n')
        assert all(s.startswith('PIPE:0,') for s in lines) or all(s.startswith('PIPE:1,') for s in lines), \
            "Invalid lines in pipe export string"

        # Extract and store the pipe's positions, checking for discontinuities in the given pipe positions
        posns = []
        for line in lines:
            fields = line.split(',')
            assert len(fields) == 3, f"Invalid num fields in PIPE line:\n{line}"

            posn = Position(col=int(fields[1]), row=int(fields[2]))
            if posns:
                assert (abs(posn[0] - posns[-1][0]), abs(posn[1] - posns[-1][1])) in ((0, 1), (1, 0)), \
                       "Pipe is not contiguous"
            posns.append(posn)

        assert posns, "Expected at least one PIPE line"
        assert len(posns) == len(set(posns)), "Pipe overlaps with itself"

        return Pipe(posns)

    def export_str(self, pipe_idx=0):
        '''Represent this pipe in solution export string format.'''
        return '\n'.join(f'PIPE:{pipe_idx},{posn.col},{posn.row}' for posn in self.posns)


class Component:
    '''Informal Interface class defining methods overworld objects will implement one or more of.'''
    __slots__ = 'type', 'posn', 'in_pipes', 'out_pipes'

    def __init__(self, type_, posn):
        self.type = type_
        self.posn = Position(*posn)
        self.in_pipes = []
        self.out_pipes = []

    def __str__(self):
        return f'{self.type},{self.posn}'

    def do_instant_actions(self, cur_cycle):
        ''''Do any instant actions (e.g. execute waldo instructions, spawn/consume molecules).'''
        pass

    def move_contents(self):
        '''Move the contents of this object (e.g. waldos/molecules), including its pipes.'''
        for pipe in self.out_pipes:
            pipe.move_contents()


class Input(Component):
    __slots__ = 'input_molecules', 'input_rate'

    # Convenience property for when we know we're dealing with an Input
    @property
    def out_pipe(self):
        return self.out_pipes[0]

    @out_pipe.setter
    def out_pipe(self, p):
        self.out_pipes[0] = p

    def __new__(cls, component_type, posn, input_dict, input_rate):
        '''Convert to a RandomInput if multiple molecules are specified.'''
        if len(input_dict['inputs']) > 1:
            return super().__new__(RandomInput)
        else:
            return super().__new__(cls)

    def __init__(self, component_type, posn, input_dict, input_rate):
        super().__init__(component_type, posn)
        assert len(input_dict['inputs']) != 0, "No molecules in input dict"
        self.input_molecules = [Molecule.from_json_string(input_mol_dict['molecule'])
                                for input_mol_dict in input_dict['inputs']]
        self.input_rate = input_rate

        dimensions = COMPONENT_SHAPES[component_type]
        # Initialize with a 1-long pipe
        self.out_pipes = [Pipe(posns=[Position(dimensions[0],
                                               (dimensions[1] - 1) // 2)])]

        # TODO: Should create a pipe of length 1 by default - which means this needs to know its own dimensions

    def do_instant_actions(self, cur_cycle):
        if cur_cycle % self.input_rate == 0 and self.out_pipe[0] is None:
            self.out_pipe[0] = copy.deepcopy(self.input_molecules[0])

    def export_str(self):
        '''Represent this input in solution export string format.'''
        # TODO: I'm still not sure what the 4th component field is used for. Custom reactor names maybe?
        return f"COMPONENT:'{self.type}',{self.posn[0]},{self.posn[1]},''" + '\n' + self.out_pipe.export_str()


class RandomInput(Input):
    __slots__ = 'random_generator', 'input_counts', 'random_bucket'

    def __init__(self, component_type, posn, input_dict, input_rate):
        super().__init__(component_type, posn, input_dict, input_rate)

        assert len(input_dict['inputs']) > 1, "Fixed input passed to RandomInput ctor"

        # Create a random generator with the given seed. Most levels default to seed 0
        seed = np.int32(input_dict['random-seed']) if 'random-seed' in input_dict else np.int32(0)
        self.random_generator = SpacechemRandom(seed=seed)
        self.random_bucket = []  # Bucket of indices for the molecules in the current balancing bucket

        # Construct one of each molecule from the input JSON
        # Input molecules have relative indices to within their zones, so let the ctor know if this is a beta input
        # zone molecule (will be initialized 4 rows downward)
        self.input_counts = [input_mol_dict['count'] for input_mol_dict in input_dict['inputs']]

    def get_input_molecule_idx(self):
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

    def do_instant_actions(self, cur_cycle):
        if cur_cycle % self.input_rate == 0 and self.out_pipe[0] is None:
            self.out_pipe[0] = copy.deepcopy(self.input_molecules[self.get_input_molecule_idx()])


class Output(Component):
    __slots__ = 'output_molecule', 'target_count', 'current_count'

    # Convenience property for when we know we're dealing with an Output
    @property
    def in_pipe(self):
        return self.in_pipes[0]

    @in_pipe.setter
    def in_pipe(self, p):
        self.in_pipes[0] = p

    def __init__(self, component_type, posn, output_dict):
        super().__init__(component_type, posn)
        self.output_molecule = Molecule.from_json_string(output_dict['molecule'])
        self.target_count = output_dict['count']
        self.current_count = 0
        self.in_pipes = [None]

    def do_instant_actions(self, cur_cycle):
        '''Check for and process any incoming molecule, and return True if this output just completed (in which case
        the caller should check if the other outputs are also done). This avoids checking all output counts every cycle.
        '''
        if self.in_pipe is None:
            return

        molecule = self.in_pipe[-1]
        if molecule is not None:
            if not molecule.isomorphic(self.output_molecule):
                raise InvalidOutputError(f"Invalid output molecule; expected {self.output_molecule} but got {molecule}")

            self.in_pipe[-1] = None
            self.current_count += 1
            if self.current_count == self.target_count:
                return True


class DisabledOutput(Output):
    '''Used by research levels, which actually crash if a wrong output is used unlike assembly reactors.'''
    __slots__ = ()

    def __init__(self, component_type, posn):
        # TODO: Should this even be a subclass of Output? I guess it gets the in_pipe property...
        Component.__init__(self, component_type, posn)
        self.in_pipes = [None]
        self.output_molecule = None
        self.target_count = None
        self.current_count = None

    def do_instant_actions(self, cur_cycle):
        # Technically should check for `in_pipe is None` first but I'd also be curious to see this crash since disabled
        # outputs are only used in research levels, where it should be impossible to not connect to the disabled output
        if self.in_pipe[-1] is not None:
            raise InvalidOutputError("A molecule was passed to a disabled output.")


class Recycler(Component):
    __slots__ = ()

    def __init__(self, component_type, posn):
        super().__init__(component_type, posn)
        self.in_pipes = [None, None, None]

    def do_instant_actions(self, cur_cycle):
        for pipe in self.in_pipes:
            if pipe is not None:
                pipe[-1] = None


# TODO: Ideally this would subclass both deque and Component but doing so gives me
#       "multiple bases have instance lay-out conflict". Need to investigate.
class StorageTank(Component):
    MAX_CAPACITY = 25
    __slots__ = 'contents',

    def __init__(self, component_type, posn, out_pipe=None):
        super().__init__(component_type, posn)
        self.in_pipes = [None]
        self.contents = collections.deque()

        dimensions = COMPONENT_SHAPES[component_type]
        if out_pipe is not None:
            self.out_pipes = [out_pipe]
        else:
            # Initialize with a 1-long pipe by default
            self.out_pipes = [Pipe(posns=[(dimensions[0], (dimensions[1] - 1) // 2)])]

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

    def do_instant_actions(self, cur_cycle):
        if self.in_pipe is None:
            return

        if self.in_pipe[-1] is not None and len(self.contents) < self.MAX_CAPACITY:
            self.contents.append(self.in_pipe[-1])
            self.in_pipe[-1] = None

        if self.out_pipe[0] is None and self.contents:
            self.out_pipe[0] = self.contents.popleft()

    @classmethod
    def from_export_str(self, export_str):
        # First line must be the COMPONENT line
        component_line, pipe_str = export_str.strip().split('\n', maxsplit=1)
        assert component_line.startswith('COMPONENT:'), "StorageTank.from_export_str expects COMPONENT line included"
        fields = component_line.split(',')
        assert len(fields) == 4, f"Unrecognized component line format:\n{component_line}"

        component_type = fields[0][len('COMPONENT:'):].strip("'")
        component_posn = Position(int(fields[1]), int(fields[2]))

        return StorageTank(component_type, component_posn, out_pipe=Pipe.from_export_str(pipe_str))


# TODO: Pass-through output
