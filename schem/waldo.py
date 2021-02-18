#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum

from .grid import Direction


class InstructionType(Enum):
    '''Represents the various types of SpaceChem instruction.'''
    START = 'S'
    INPUT = 'i'
    OUTPUT = 'o'
    GRAB_DROP = 'g-d'
    GRAB = 'grb'
    DROP = 'drp'
    ROTATE = 'r'
    SYNC = 'sy'
    BOND_PLUS = 'b+'
    BOND_MINUS = 'b-'
    SENSE = '?'
    FLIP_FLOP = 'ff'
    FUSE = 'fus'
    SPLIT = 'spl'
    SWAP = 'swp'
    PAUSE = 'P'

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

    # TODO: Define from_export_str to simplify Reactor.from_export_str

    def export_str(self, waldo_idx, posn):
        # Handle instructions with names/sub-types that I deliberately differed from SC's implementation
        special_case_dict = {
            InstructionType.GRAB_DROP: ('grab', 0),
            InstructionType.GRAB: ('grab', 1),
            InstructionType.DROP: ('grab', 2),
            InstructionType.ROTATE: ('rotate', 0 if self.direction == Direction.CLOCKWISE else 1),
            InstructionType.BOND_PLUS: ('bond', 0),
            InstructionType.BOND_MINUS: ('bond', 1),
            InstructionType.SENSE: ('sensor', 0),
            InstructionType.FLIP_FLOP: ('toggle', 0)}
        if self.type in special_case_dict:
            name, sub_type = special_case_dict[self.type]
        else:
            name, sub_type = self.type.name.lower(), self.target_idx if self.target_idx is not None else 0

        # Direction stored in degrees with UP = -90, LEFT = 180
        if self.direction is None or self.type == InstructionType.ROTATE:
            dirn_degrees = -1
        else:
            dirn_degrees = ((self.direction.value - 1) * 90)

        waldo_int = 128 if waldo_idx == 0 else 32  # Instructions are 32 for blue, 128 for red
        target_element = self.target_idx if self.type == InstructionType.SENSE else 0

        # Dunno what second last field does
        return f"MEMBER:'instr-{name}',{dirn_degrees},{sub_type},{waldo_int},{posn.col},{posn.row},0,{target_element}"


class Waldo:
    __slots__ = 'idx', 'instr_map', 'flipflop_states', 'position', 'direction', 'molecule', 'is_stalled', 'is_rotating'

    def __init__(self, idx, instr_map):
        self.idx = idx
        self.instr_map = instr_map  # Position map of tuples containing arrow (direction) and 'command' (non-arrow) instructions
        self.flipflop_states = {}
        self.position = None  # Dummy init so we can check for duplicate Start
        for posn, (_, cmd) in instr_map.items():
            if cmd is not None:
                if cmd.type == InstructionType.FLIP_FLOP:
                    self.flipflop_states[posn] = False
                elif cmd.type == InstructionType.START:
                    assert self.position is None, "Duplicate waldo Start instruction"
                    self.position = posn
                    self.direction = cmd.direction
        assert self.position is not None, "Missing waldo Start instruction"


        self.molecule = None
        self.is_stalled = False  # Updated if the waldo is currently stalled on a sync, output, wall, etc.
        self.is_rotating = False  # Used to distinguish the first vs second cycle on a rotate command

    def __hash__(self):
        '''A waldo's run state is uniquely identified by its position, direction,
        whether it's holding a molecule, whether it's rotating, and its flip-flop states.
        '''
        return hash((self.position, self.direction, self.molecule is None, self.is_rotating,
                     tuple(self.flipflop_states.values())))

    def __repr__(self):
        return (f'Waldo({self.idx}, pos={self.position}, dir={self.direction}, is_stalled={self.is_stalled}'
                + f', instr_map={self.instr_map})')

    def __len__(self):
        return sum(1 for cell_instrs in self.instr_map.values()
                   for instr in cell_instrs
                   if (instr is not None
                       and not (isinstance(instr, Instruction) and instr.type == InstructionType.START)))

    def cur_cmd(self):
        '''Return a waldo's current 'command', i.e. non-arrow instruction, or None.'''
        return None if self.position not in self.instr_map else self.instr_map[self.position][1]

    def export_str(self):
        '''Represent this waldo's instructions in solution export string format.'''
        lines = []
        for posn, (arrow, instr) in self.instr_map.items():
            if arrow is not None:
                waldo_int = 64 if self.idx == 0 else 16
                dirn_degrees = (arrow.value - 1) * 90  # TODO: Should probably stop using .value for IntEnum's?
                lines.append(f"MEMBER:'instr-arrow',{dirn_degrees},0,{waldo_int},{posn.col},{posn.row},0,0")

            if instr is not None:
                lines.append(instr.export_str(waldo_idx=self.idx, posn=posn))

        return '\n'.join(lines)
