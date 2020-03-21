#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum

from spacechem.grid import Position, Direction

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
    INPUT = 0
    OUTPUT = 1
    GRAB = 2
    DROP = 3
    GRAB_DROP = 4
    ROTATE = 5
    SYNC = 6
    BOND_PLUS = 7
    BOND_MINUS = 8
    SENSE = 9
    FLIP_FLOP = 10
    FUSE = 11
    SPLIT = 12
    SWAP = 13

    def __str__(self):
        return self.name
    __repr__ = __str__


class Instruction(namedtuple('Instruction', ('instr_type', 'direction', 'target_idx'),
                             # Default direction and target_idx if unneeded
                             defaults=(None, None))):
    '''Represents a non-arrow SpaceChem instruction and any associated properties.
    Direction is used by rotates, flip-flops and sensor cmds. Target index represents either the zone idx for an input
    or output instruction, or an element's atomic # in the case of a sensor cmd.
    '''
    __slots__ = ()

    def __eq__(self, other):
        '''Allow direct comparison to instruction type for sanity.'''
        if isinstance(other, InstructionType):
            return self.instr_type == other
        else:
            return super().__eq__(self, other)


class Solution:
    __slots__ = 'name', 'symbols', 'bonders', 'waldo_starts', 'waldo_instr_maps', 'expected_score'

    def __init__(self, soln_export_string):
        self.name = ''
        self.symbols = 0
        # Level Features
        self.bonders = {}
         # Store waldo starts outside the instruction map for quick access when initializing
         # a solution (and since they don't need to appear in the instruction map)
        self.waldo_starts = [None, None]
        # Map of positions to pairs of arrows (directions) and/or non-arrow instructions
        self.waldo_instr_maps = [{}, {}]
        bonder_count = 0
        for line in soln_export_string.split('\n'):
            if line.startswith('MEMBER'):
                csv_values = line.split(',')
                if len(csv_values) != 8:
                    raise Exception(f"Unrecognized solution string line format:\n{line}")

                # Red has a field which is 64 for arrows, 128 for instructions
                # The same field in Blue is 16 for arrows, 32 for instructions
                waldo_idx = 0 if int(csv_values[3]) >= 64 else 1

                position = Position(int(csv_values[5]), int(csv_values[4]))

                # Game stores directions in degrees, with right = 0... but with up = -90? Should
                # really have been 90 if zach was going with math-defined...
                direction = None if int(csv_values[1]) == -1 else Direction(1 + int(csv_values[1]) // 90)

                member_name = csv_values[0].split(':')[1].strip("'")

                if member_name == 'feature-bonder':
                    bonder_count += 1
                    self.bonders[position] = bonder_count
                    continue
                elif member_name == 'instr-start':
                    self.waldo_starts[waldo_idx] = (position, direction)
                    continue

                # If this isn't a start instruction, increment total symbols
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
                    self.waldo_instr_maps[waldo_idx][position][1] = InstructionType.SYNC
                elif member_name == 'instr-bond':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[waldo_idx][position][1] = InstructionType.BOND_PLUS
                    else:
                        self.waldo_instr_maps[waldo_idx][position][1] = InstructionType.BOND_MINUS
            elif line.startswith('SOLUTION'):
                csv_values = line.split(',')
                self.expected_score = tuple(int(i) for i in csv_values[2].split('-'))

                if len(csv_values) >= 4:
                    self.name = csv_values[3]

    def __repr__(self):
        return f'Solution(bonders={self.bonders}, starts={self.waldo_starts}, instrs={self.waldo_instr_maps})'

    def get_export_string(self):
        pass

    def add_instruction(self, waldo_idx, posn, instr):
        if posn not in self.waldo_instr_maps[waldo_idx]:
            self.waldo_instr_maps[waldo_idx][posn] = [None, None]

        self.waldo_instr_maps[waldo_idx][1] = instr

    def add_arrow(self, waldo_idx, posn, arrow_dirn):
        if posn not in self.waldo_instr_maps[waldo_idx]:
            self.waldo_instr_maps[waldo_idx][posn] = [None, None]

        self.waldo_instr_maps[waldo_idx][0] = arrow_dirn
