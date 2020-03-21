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

    def __str__(self):
        return self.name
    __repr__ = __str__

    def __add__(self, other):
        '''The sum of UP and CLOCKWISE is RIGHT.
        Also functions as a mapper from internal bond directions to the external grid's directions.
        (a molecule that has relative orientation RIGHT needs to transform each of its bond
        directions to be direction + RIGHT (clockwise) (UP is the identity direction).
        '''
        return Direction((self.value + other.value) % 4)

    def __sub__(self, other):
        '''The UP minus CLOCKWISE is LEFT.
        Also functions as a mapper from the external grid's directions to internal bond directions.
        (a molecule that has relative orientation RIGHT needs to transform an added atom's bonds
        to be direction - RIGHT (counter-clockwise) (UP is the identity direction).
        '''
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
