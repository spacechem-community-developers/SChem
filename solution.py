#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

class Instruction(Enum):
    INPUT_ALPHA = 0
    INPUT_BETA = 1
    OUTPUT_PSI = 2
    OUTPUT_OMEGA = 3
    GRAB = 4
    DROP = 5
    GRAB_DROP = 6
    ROTATE_CWISE = 7
    ROTATE_CCWISE = 8
    BOND_PLUS = 9
    BOND_MINUS = 10
    SYNC = 11
    FLIP_FLOP = 12
    FUSE = 13
    SPLIT = 14
    SWAP = 15

    def __str__(self):
        return self.name
    __repr__ = __str__

class Solution:
    def __init__(self, soln_export_string):
        self.name=''
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
                color_idx = 0 if int(csv_values[3]) >= 64 else 1

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
                    self.waldo_starts[color_idx] = (position, direction)
                    continue

                # If this isn't a start instruction, increment total symbols
                self.symbols += 1

                # All commands except start instructions get added to the instruction map
                if position not in self.waldo_instr_maps[color_idx]:
                    self.waldo_instr_maps[color_idx][position] = [None, None]

                # Note: Some similar instructions have the same name but are sub-typed by the
                #       second integer field
                instr_sub_type = int(csv_values[2])
                if member_name == 'instr-arrow':
                    self.waldo_instr_maps[color_idx][position][0] = direction
                elif member_name == 'instr-input':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.INPUT_ALPHA
                    else:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.INPUT_BETA
                elif member_name == 'instr-output':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.OUTPUT_PSI
                    else:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.OUTPUT_OMEGA
                elif member_name == 'instr-grab':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.GRAB_DROP
                    elif instr_sub_type == 1:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.GRAB
                    else:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.DROP
                elif member_name == 'instr-rotate':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.ROTATE_CWISE
                    else:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.ROTATE_CCWISE
                elif member_name == 'instr-sync':
                    self.waldo_instr_maps[color_idx][position][1] = Instruction.SYNC
                elif member_name == 'instr-bond':
                    if instr_sub_type == 0:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.BOND_PLUS
                    else:
                        self.waldo_instr_maps[color_idx][position][1] = Instruction.BOND_MINUS
            elif line.startswith('SOLUTION'):
                csv_values = line.split(',')
                self.expected_score = tuple(int(i) for i in csv_values[2].split('-'))

                if len(csv_values) >= 4:
                    self.name = csv_values[3]

    def __repr__(self):
        return f'Solution(bonders={self.bonders}, starts={self.waldo_starts}, instrs={self.waldo_instr_maps})'

    def get_export_string(self):
        pass
