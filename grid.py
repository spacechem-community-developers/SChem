#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import IntEnum
import math


class Direction(IntEnum):  # TODO: Need to research Enum vs IntEnum more... Enum lacked a consistent __hash__
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
        if self.value >= 4:  # rotational directions, this is a bit hacky since I stuffed them in
                             # what should really be two separate enums...
            return Direction(4 + ((self.value + 2) % 4))
        else:
            return Direction((self.value + 2) % 4)


class Position(namedtuple("Position", ('row', 'col'))):
    '''Grid position, with row and column 0-indexed from the top left of the reactor.'''

    dirn_to_delta = {Direction.UP: (-1, 0),
                     Direction.RIGHT: (0, 1),
                     Direction.DOWN: (1, 0),
                     Direction.LEFT: (0, -1)}

    __slots__ = ()  # Apparently necessary for preserving namedtuple performance in a subclass

    def __str__(self):
        return f'({self.row}, {self.col})'
    __repr__ = __str__

    def __add__(self, direction):
        r_delta, c_delta = self.dirn_to_delta[direction]
        return Position(self.row + r_delta, self.col + c_delta)

    def move(self, direction, distance):
        r_delta, c_delta = self.dirn_to_delta[direction]
        return Position(self.row + distance * r_delta, self.col + distance * c_delta)

    def rotate_fine(self, pivot_pos, direction, radians):
        '''Rotate in sub-quarter-turn increments.'''
        # In normal cartesian math clockwise would be negative, but our vertical axis is reversed
        if direction == Direction.COUNTER_CLOCKWISE:
            radians *= -1

        sin_theta, cos_theta = math.sin(radians), math.cos(radians)
        return Position(pivot_pos.row + sin_theta * (self.col - pivot_pos.col) + cos_theta * (self.row - pivot_pos.row),
                        pivot_pos.col + cos_theta * (self.col - pivot_pos.col) - sin_theta * (self.row - pivot_pos.row))

    def round(self):
        '''Return from float-precision co-ordinates to the integer grid.'''
        return Position(round(self.row), round(self.col))
