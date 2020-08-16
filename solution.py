#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum

from spacechem.grid import Position, Direction
from spacechem.elements_data import elements_dict


# Solution (NN output) bits:
# 80 x Cell bits:
#    Red: 10 bits
#       Arrow Instruction: 3 bits
#           0 = None
#           1-4 = clockwise from 1 = Up.
#       Cmd Instruction: 7 bits
#       0 = None
#       1-109: Sensor
#       110 = Start
#       111 = Input Alpha
#       112 = Input Beta
#       113 = Output Psi
#       114 = Output Omega
#       115 = Grab
#       116 = Drop
#       117 = Grab-Drop
#       118 = Rotate clockwise
#       119 = Rotate C-Clockwise
#       120 = Bond+
#       121 = Bond-
#       122 = Sync
#       123 = Flip-Flop
#       124 = Fuse
#       125 = Split
#       126 = Swap
#   Blue (ditto): 10 bits
# = 20 bits per grid pos = 1600 bits
# Feature bits: 91 bits
#    8 x Bonder locations (0-7 row + 0-9 col = 7 bits) = 56 bits
#    1 x Sensor location = 7 bits
#    1 x Fuser location = 7 bits
#    1 x Splitter location = 7 bits
#    2 x Swapper locations = 14 bits
# Total: 1691 output bits

# If we choose to skip sensor levels, the cmd instruction can be reduced from 7 to 4 bits
# 0 = None
# 1 = Start
# 2, 3 = Input, Ouput
# 4, 5, 6 = Grab, Drop, Grab-Drop
# 7, 8 = Rotate c-wise / cc-wise
# 9, 10 = Bond+, Bond-
# 11 = Sync
# 12 = FF
# 13, 14 = Fuse, Split
# 15 = Swap

class InstructionType(Enum):
    '''Represents the various types of SpaceChem instruction.'''
    START = 'S'
    INPUT = 'i'
    OUTPUT = 'o'
    GRAB = 'g'
    DROP = 'd'
    GRAB_DROP = 'gd'
    ROTATE = 'r'
    SYNC = 'sy'
    BOND_PLUS = 'b+'
    BOND_MINUS = 'b-'
    SENSE = '?'
    FLIP_FLOP = 'f'
    FUSE = 'fus'
    SPLIT = 'spl'
    SWAP = 'swp'

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.value


class Instruction(namedtuple('Instruction', ('type', 'direction', 'target_idx'),
                             # Default direction and target_idx if unneeded
                             defaults=(None, None))):
    '''Represents a non-arrow SpaceChem instruction and any associated properties.
    Direction is used by rotates, flip-flops and sensor cmds. Target index represents either the zone idx for an input
    or output instruction, or an element's atomic # in the case of a sensor cmd.
    '''
    __slots__ = ()

    def __str__(self):
        return (f'{self.type}'
                f'{self.target_idx if self.target_idx is not None else ""}'
                f'{self.direction if self.direction is not None else ""}')

    def __repr__(self):
        return (f'Instruction({repr(self.type)}'
                + (f', target_idx={self.target_idx}' if self.target_idx is not None else '')
                + (f', direction={repr(self.direction)}' if self.direction is not None else '')
                + ')')


class Solution:
    __slots__ = ('level', 'author', 'name', 'symbols', 'bonders', 'sensors', 'waldo_starts', 'waldo_instr_maps',
                 'expected_score')

    def __init__(self, level, soln_export_string=None):
        self.level = level
        self.name = ''
        self.symbols = 0
        # Level Features
        # TODO: Now that we pre-compute bond firing orders in Reactor.__init__, just using a list here might be fine
        #       performance-wise and saves on index-handling code
        self.bonders = {}  # posn:idx dict for quick lookups
        self.sensors = []  # posns
        # Store waldo Starts as (posn, dirn) pairs for quick access when initializing reactor waldos
        self.waldo_starts = [None, None]
        # One map for each waldo, of positions to pairs of arrows (directions) and/or non-arrow instructions
        # TODO: usage might be cleaner if separate arrow_maps and instr_maps
        self.waldo_instr_maps = [{}, {}]

        if soln_export_string is None:
            return

        feature_posns = set()  # for verifying features were not placed illegally
        bonder_count = 0  # Track bonder priorities

        for line in soln_export_string.split('\n'):
            if line.startswith('MEMBER'):
                csv_values = line.split(',')
                if len(csv_values) != 8:
                    raise Exception(f"Unrecognized solution string line format:\n{line}")

                member_name = csv_values[0].split(':')[1].strip("'")

                # Game stores directions in degrees, with right = 0, up = -90 (reversed so sin math works on
                # the reversed vertical axis)
                direction = None if int(csv_values[1]) == -1 else Direction(1 + int(csv_values[1]) // 90)

                # Red has a field which is 64 for arrows, 128 for instructions
                # The same field in Blue is 16 for arrows, 32 for instructions
                waldo_idx = 0 if int(csv_values[3]) >= 64 else 1

                position = Position(int(csv_values[5]), int(csv_values[4]))

                if member_name == 'instr-start':
                    self.add_instruction(waldo_idx=waldo_idx, posn=position,
                                         instr=Instruction(InstructionType.START, direction=direction))
                    continue
                elif member_name.startswith('feature-'):
                    if position in feature_posns:
                        raise Exception(f"Solution contains overlapping features at {position}")
                    feature_posns.add(position)

                    if member_name == 'feature-bonder':
                        bonder_count += 1
                        self.bonders[position] = bonder_count
                    elif member_name == 'feature-sensor':
                        self.sensors.append(position)

                    continue

                # Given that this isn't a start instruction/feature, increment total symbols
                self.symbols += 1

                # All commands except start instructions get added to the instruction map
                if position not in self.waldo_instr_maps[waldo_idx]:
                    self.waldo_instr_maps[waldo_idx][position] = [None, None]

                # Note: Some similar instructions have the same name but are sub-typed by the
                #       second integer field
                instr_sub_type = int(csv_values[2])
                if member_name == 'instr-arrow':
                    self.waldo_instr_maps[waldo_idx][position][0] = direction
                elif member_name == 'instr-input':
                    self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.INPUT,
                                                                                target_idx=instr_sub_type)
                elif member_name == 'instr-output':
                    self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.OUTPUT,
                                                                                target_idx=instr_sub_type)
                elif member_name == 'instr-grab':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.GRAB_DROP)
                    elif instr_sub_type == 1:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.GRAB)
                    else:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.DROP)
                elif member_name == 'instr-rotate':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.ROTATE,
                                                                                    direction=Direction.CLOCKWISE)
                    else:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.ROTATE,
                                                                                    direction=Direction.COUNTER_CLOCKWISE)
                elif member_name == 'instr-sync':
                    self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SYNC)
                elif member_name == 'instr-bond':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.BOND_PLUS)
                    else:
                        self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.BOND_MINUS)
                elif member_name == 'instr-sensor':
                    # The last CSV field is used by the sensor for the target atomic number
                    atomic_num = int(csv_values[7])
                    if atomic_num not in elements_dict:
                        raise Exception(f"Invalid atomic number {atomic_num} on sensor command.")
                    self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SENSE,
                                                                                direction=direction,
                                                                                target_idx=atomic_num)
                elif member_name == 'instr-toggle':
                    self.waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.FLIP_FLOP,
                                                                                direction=direction)
            elif line.startswith('SOLUTION'):
                csv_values = line.split(',')
                soln_level_name = csv_values[0][len('SOLUTION:'):]
                if soln_level_name != level['name']:
                    print(f"Warning: Solution level name {repr(soln_level_name)} doesn't match expected name"
                          + f" {repr(level['name'])}.")
                self.author = csv_values[1]
                self.expected_score = tuple(int(i) for i in csv_values[2].split('-'))

                if len(csv_values) >= 4:
                    self.name = ','.join(csv_values[3:])  # Remainder is the soln name and may include commas

        # Sanity checks
        assert len(self.bonders) == level['bonder-count']

        # TODO: ('fuser', self.fusers, 1), ('splitter', self.splitters, 1), ('teleporter', self.swappers, 2)
        for feature, container, default_count in (('sensor', self.sensors, 1),):
            assert not container or level[f'has-{feature}']

            # Regular vs Community Edition sanity checks
            if f'{feature}-count' in level:
                assert len(container) == level[f'{feature}-count']
            elif level[f'has-{feature}']:
                assert len(container) == default_count

    def __repr__(self):
        return f'Solution(bonders={self.bonders}, starts={self.waldo_starts}, instrs={self.waldo_instr_maps})'

    def __str__(self):
        s = f'inputs={self.level.input_molecules}'
        s += f'\noutputs={self.level.output_molecules}'

        for waldo_idx, waldo_name in ((0, 'red'), (1, 'blue')):
            s += f'\n{waldo_name}:'

            # Show col indices
            s += '\n  '
            for col in range(10):
                s += f'  {col}  '

            for row in range(8):
                s += f'\n{row} '  # Show row index
                for col in range(10):
                    posn = Position(row, col)
                    if posn in self.waldo_instr_maps[waldo_idx]:
                        arrow, instr = self.waldo_instr_maps[waldo_idx][Position(row, col)]
                        s += f'{str(instr).rjust(3) if instr is not None else "   "} {arrow if arrow is not None else " "}'
                    else:
                        s += 5 * ' '

        return s

    def get_export_string(self):
        pass

    def add_instruction(self, waldo_idx, posn, instr):
        # TODO: Raise exception if start instr already exists for this waldo? Or else return False for performance...
        if instr.type == InstructionType.START:
            self.waldo_starts[waldo_idx] = (posn, instr.direction)
        else:
            self.symbols += 1

        if posn not in self.waldo_instr_maps[waldo_idx]:
            self.waldo_instr_maps[waldo_idx][posn] = [None, None]

        self.waldo_instr_maps[waldo_idx][posn][1] = instr

    def add_arrow(self, waldo_idx, posn, arrow_dirn):
        if posn not in self.waldo_instr_maps[waldo_idx]:
            self.waldo_instr_maps[waldo_idx][posn] = [None, None]

        self.waldo_instr_maps[waldo_idx][posn][0] = arrow_dirn
        self.symbols += 1

    def add_bonder(self, posn):
        self.bonders[posn] = len(self.bonders) + 1  # Index the bonders by order of insertion
