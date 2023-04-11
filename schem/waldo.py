#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum

from .grid import *


class InstructionType(Enum):
    """Represents the various types of SpaceChem non-arrow instruction."""
    START = 'S'
    INPUT = 'i'
    OUTPUT = 'o'
    GRAB_DROP = 'g-d'
    GRAB = 'grb'
    DROP = 'drp'
    ROTATE = 'r'  # ⭯ ⭮
    SYNC = 'sy'  # ⌛
    BOND_PLUS = 'b+'  # ⊕
    BOND_MINUS = 'b-'
    SENSE = '?'
    FLIP_FLOP = 'ff'  # F
    FUSE = 'fus'  # ⛯☢️
    SPLIT = 'spl'  # ☢️
    SWAP = 'swp'  # ⇄
    PAUSE = 'P'  # ⏸
    CONTROL = 'ctl'

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.value[0]


class Instruction(namedtuple('Instruction', ('type', 'direction', 'target_idx'),
                             # Default direction and target_idx if unneeded
                             defaults=(None, None))):
    """Represents a non-arrow SpaceChem instruction and any associated properties.
    Direction is used by rotates, flip-flops and sensor cmds. Target index represents either the zone idx for an input
    or output instruction, or an element's atomic # in the case of a sensor cmd.
    """
    __slots__ = ()

    STR_MAP = {InstructionType.START: 'S',
               InstructionType.INPUT: 'i',
               InstructionType.OUTPUT: 'o',
               InstructionType.GRAB_DROP: '&',
               InstructionType.GRAB: 'G',
               InstructionType.DROP: 'D',
               InstructionType.ROTATE: {Direction.CLOCKWISE: '⭮', Direction.COUNTER_CLOCKWISE: '⭯'},
               InstructionType.SYNC: '⌛',
               InstructionType.BOND_PLUS: '+',
               InstructionType.BOND_MINUS: '-',
               InstructionType.SENSE: '?',
               InstructionType.FLIP_FLOP: 'F',  # △ ▷ ▽ ◁ ▲ ▶ ◀ ▼
               InstructionType.FUSE: '⛯',  # ☢️ ?
               InstructionType.SPLIT: '☢',
               InstructionType.SWAP: '⇔',  # ⇄ ?
               InstructionType.PAUSE: 'P',
               InstructionType.CONTROL: 'C'}

    def __repr__(self):
        return (f'Instruction({repr(self.type)}'
                + (f', target_idx={self.target_idx}' if self.target_idx is not None else '')
                + (f', direction={repr(self.direction)}' if self.direction is not None else '')
                + ')')

    def __str__(self):
        s = self.STR_MAP[self.type]
        return s if isinstance(s, str) else s[self.direction]

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
            InstructionType.FLIP_FLOP: ('toggle', 0),
            InstructionType.PAUSE: ('debug', 0)}
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
    __slots__ = 'idx', 'arrows', 'commands', 'flipflop_states', 'position', 'direction', 'molecule', 'is_stalled', 'is_rotating'

    # Map of good unicode characters for displaying a waldo's paths
    dirns_to_char = {
        # https://www.compart.com/en/unicode/charsets/ISO_10367-box (light box chars): ┌,─,┐,└,│,┘,┼,┴,┬,┤,├,╴,╶,╵,╷
        frozenset((UP,)): '│',    # ╵ sadly doesn't work in common fonts
        frozenset((DOWN,)): '│',  # ╷
        frozenset((LEFT,)): '─',  # ╴
        frozenset((RIGHT,)): '─', # ╶
        frozenset((UP, RIGHT)): '└',
        frozenset((UP, DOWN)): '│',
        frozenset((UP, LEFT)): '┘',
        frozenset((RIGHT, DOWN)): '┌',
        frozenset((RIGHT, LEFT)): '─',
        frozenset((DOWN, LEFT)): '┐',
        frozenset((UP, RIGHT, DOWN)): '├',
        frozenset((UP, RIGHT, LEFT)): '┴',
        frozenset((UP, DOWN, LEFT)): '┤',
        frozenset((RIGHT, DOWN, LEFT)): '┬',
        frozenset((UP, RIGHT, DOWN, LEFT)): '┼'}

    def __init__(self, idx, arrows, commands):
        self.idx = idx
        self.arrows = arrows  # Position map of tuples containing arrow instructions
        self.commands = commands  # Position map of tuples containing non-arrow instructions
        self.flipflop_states = {}
        self.position = None  # Dummy init so we can check for duplicate Start
        for posn, cmd in commands.items():
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
        """A waldo's run state is uniquely identified by its position, direction,
        whether it's holding a molecule, whether it's rotating, and its flip-flop states.
        """
        return hash((self.position, self.direction, self.molecule is None, self.is_rotating,
                     tuple(self.flipflop_states.values())))

    def __repr__(self):
        return (f'Waldo({self.idx}, pos={self.position}, dir={self.direction}, is_stalled={self.is_stalled}'
                + f', arrows={self.arrows}, cmds={self.commands})')

    def __len__(self):
        return len(self.arrows) + len(self.commands) - 1  # Should be one Start instruction

    def cur_cmd(self):
        """Return a waldo's current 'command', i.e. non-arrow instruction, or None."""
        return self.commands[self.position] if self.position in self.commands else None

    def export_str(self):
        """Represent this waldo's instructions in solution export string format."""
        start_line = None
        lines = []
        # Merge arrows and commands and sort them all by position (col then row), but separating the start command
        for posn, instr in sorted(list(self.arrows.items()) + list(self.commands.items()),  # Arrows before commands
                                  key=lambda x: x[0]):  # Tie-break will keep arrows in front of non-arrows
            if isinstance(instr, Direction):  # Arrow
                waldo_int = 64 if self.idx == 0 else 16
                dirn_degrees = (instr.value - 1) * 90  # TODO: Should probably stop using .value for IntEnum's?
                lines.append(f"MEMBER:'instr-arrow',{dirn_degrees},0,{waldo_int},{posn.col},{posn.row},0,0")
            elif instr.type == InstructionType.START:
                start_line = instr.export_str(waldo_idx=self.idx, posn=posn)
            else:  # Command
                lines.append(instr.export_str(waldo_idx=self.idx, posn=posn))

        # Put the start instr line at the front
        return '\n'.join([start_line, *lines])

    def trace_path(self, num_cols=10, num_rows=8):
        """Return a dict of position: directions representing paths this waldo visually traces.
        Useful for representing the waldo visually and for calculating e.g. waldopath.
        """
        def is_valid_posn(posn):
            return 0 <= posn.col < num_cols and 0 <= posn.row < num_rows

        branching_instr_types = {InstructionType.SENSE, InstructionType.FLIP_FLOP}

        # Override the start direction with any arrow since unlike other directional commands it can't branch
        start_posn, start_dirn = next((posn, (self.arrows[posn] if posn in self.arrows else cmd.direction))
                                      for posn, cmd in self.commands.items()
                                      if cmd.type == InstructionType.START)
        visited_posn_dirns = {}  # posn: directions to catch when we're looping
        traced_posn_dirns = {}  # posn: directions that were visually traced (separate from visited since
                                # a cell that is passed through in only one direction may be traced in both)
        unexplored_branches_stack = [(start_posn, start_dirn)]
        while unexplored_branches_stack:
            cur_posn, incoming_dirn = unexplored_branches_stack.pop()

            # Trace the path coming into this cell
            if cur_posn not in traced_posn_dirns:
                traced_posn_dirns[cur_posn] = set()
            traced_posn_dirns[cur_posn].add(incoming_dirn.opposite())

            # Arrows update the direction of the current branch but don't create a new one
            outgoing_dirn = self.arrows[cur_posn] if cur_posn in self.arrows else incoming_dirn

            # Check the current position/direction against the visit map. We do this after evaluating the arrow to
            # reduce excess visits (since the original direction of a waldo never matters to its outgoing path if an
            # arrow is present, unlike with branching commands)
            if cur_posn not in visited_posn_dirns:
                visited_posn_dirns[cur_posn] = set()
            elif outgoing_dirn in visited_posn_dirns[cur_posn]:
                # We've already explored this cell in the current direction and must have already added any branches
                # starting from this cell, so end this branch
                continue

            traced_posn_dirns[cur_posn].add(outgoing_dirn)
            visited_posn_dirns[cur_posn].add(outgoing_dirn)

            # Add any new branch
            cmd = self.commands[cur_posn] if cur_posn in self.commands else None
            if cmd is not None and cmd.type in branching_instr_types:
                traced_posn_dirns[cur_posn].add(cmd.direction)
                next_branch_posn = cur_posn + cmd.direction
                if is_valid_posn(next_branch_posn):
                    unexplored_branches_stack.append((next_branch_posn, cmd.direction))

            # Put the current branch back on top of the stack
            next_posn = cur_posn + outgoing_dirn
            if is_valid_posn(next_posn):
                unexplored_branches_stack.append((next_posn, outgoing_dirn))

        return traced_posn_dirns

    def reset(self):
        for posn, cmd in self.commands.items():
            if cmd.type == InstructionType.START:
                self.position = posn
                self.direction = cmd.direction
                break

        self.molecule = None
        self.is_stalled = False
        self.is_rotating = False

        for k in self.flipflop_states:
            self.flipflop_states[k] = False

        return self
