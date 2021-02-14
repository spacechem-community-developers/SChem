#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import IntEnum
import math


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    CLOCKWISE = 5  # 1 mod 4
    COUNTER_CLOCKWISE = 7  # 3 mod 4

    def __repr__(self):
        return self.name

    def __str__(self):
        return {0: '^', 1: '>', 2: 'v', 3: '<', 5: '<', 7: '>'}[self.value]

    def __add__(self, other):
        '''E.g. UP + CLOCKWISE == RIGHT.'''
        return Direction((self.value + other.value) % 4)

    def __sub__(self, other):
        '''E.g. UP - CLOCKWISE == LEFT.'''
        return Direction((self.value - other.value) % 4)

    def opposite(self):
        if self.value < 4:
            return Direction((self.value + 2) % 4)
        else:
            # rotational directions, this is a bit hacky since I stuffed them in what should really be separate enums...
            return Direction(4 + ((self.value + 2) % 4))


# Convenience
CARDINAL_DIRECTIONS = UP, RIGHT, DOWN, LEFT = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)


class Position(namedtuple("Position", ('col', 'row'))):
    '''Grid position, with (col, row) 0-indexed from the top left of the reactor or overworld.'''

    dirn_to_delta = {UP: (0, -1),
                     RIGHT: (1, 0),
                     DOWN: (0, 1),
                     LEFT: (-1, 0)}

    __slots__ = ()

    def __str__(self):
        return f'({self.col}, {self.row})'
    __repr__ = __str__

    def __add__(self, other):
        if isinstance(other, Direction):
            c_delta, r_delta = self.dirn_to_delta[other]
            return Position(self.col + c_delta, self.row + r_delta)
        else:
            return Position(self.col + other[0], self.row + other[1])

    def __sub__(self, other):
        return Position(self.col - other[0], self.row - other[1])

    def __abs__(self):
        return Position(abs(self[0]), abs(self[1]))

    def move(self, direction, distance=1):
        c_delta, r_delta = self.dirn_to_delta[direction]
        return Position(self.col + distance * c_delta, self.row + distance * r_delta)

    def rotate_fine(self, pivot_pos, direction, radians):
        '''Rotate in sub-quarter-turn increments.'''
        # In normal cartesian math clockwise would be negative, but our vertical axis is reversed
        if direction == Direction.COUNTER_CLOCKWISE:
            radians *= -1

        sin_theta, cos_theta = math.sin(radians), math.cos(radians)
        return Position(pivot_pos.col + cos_theta * (self.col - pivot_pos.col) - sin_theta * (self.row - pivot_pos.row),
                        pivot_pos.row + sin_theta * (self.col - pivot_pos.col) + cos_theta * (self.row - pivot_pos.row))

    def round(self):
        '''Return from float-precision co-ordinates to the integer grid.'''
        return Position(round(self.col), round(self.row))
