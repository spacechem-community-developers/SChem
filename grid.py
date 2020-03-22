#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from enum import Enum


class Direction(Enum):
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

    __slots__ = ()  # Apparently necessary for preserving namedtuple performance in a subclass

    def __str__(self):
        return f'({self.row}, {self.col})'
    __repr__ = __str__

    # TODO: Check for wall collisions implicitly here?
    def __add__(self, direction):
        if direction == Direction.UP:
            return Position(self.row - 1, self.col)
        elif direction == Direction.RIGHT:
            return Position(self.row, self.col + 1)
        elif direction == Direction.DOWN:
            return Position(self.row + 1, self.col)
        elif direction == Direction.LEFT:
            return Position(self.row, self.col - 1)

    def rotate(self, pivot_pos, direction):
        '''Return the position obtained by rotating this position around a pivot point.'''
        if direction == Direction.CLOCKWISE:
            return Position(pivot_pos.row + (self.col - pivot_pos.col),
                            pivot_pos.col - (self.row - pivot_pos.row))
        else:
            return Position(pivot_pos.row - (self.col - pivot_pos.col),
                            pivot_pos.col + (self.row - pivot_pos.row))
