#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product

MAX_TERRAIN_INT = 7

# Terrain obstacles and default component positions. The positions of obstacles within a level is uniquely identified by
# the terrain type for ResNet and custom levels, but each main game level has a unique terrain layout.
# This dict will store, for each type indexed by either its level name (main game) or its
# integer terrain type (ResearchNet/custom), the following:
# * Set containing all co-ordinates obstructed by terrain obstacles (rocks, etc.)
# * lists of where inputs, outputs, and other overworld buildings will be spawned if present

# itertools.product used to somewhat more compactly specify the co-ordinates in obstacles
terrains = {
    'research':
        {'obstructed': {},
         'input-zones': (('research-input', (0, 1)),  # actually these have empty type in SC but fuck that
                         ('research-input', (0, 2))),
         'output-zones': (('research-output', (7, 1)),
                          ('research-output', (7, 2)),)},
    0: {  # Sernimir IV
        'obstructed': {
            # Top-left rock
            *product(range(0, 10), (0,)),
            *product(range(0, 9), (1,)),
            *product(range(0, 8), (2,)),
            *product(range(2, 7), (3,)),
            # Mid-left rock
            *product(range(4, 6), (8,)),
            *product(range(3, 7), (9,)),
            *product(range(2, 8), range(10, 13)),
            *product(range(5, 7), (13,)),
            # Bottom rock
            *product(range(18, 21), (16,)),
            *product(range(14, 22), range(17, 19)),
            *product(range(15, 22), (19,)),
            *product(range(17, 21), (20,)),
            # Top-right rock
            *product(range(26, 32), range(0, 2)),
            *product(range(27, 32), (2,)),
            *product(range(28, 32), (3,)),
            *product(range(29, 32), (4,)),
            # Mid-right rock
            *product(range(30, 32), range(13, 19)),
            *product((29,), range(14, 18)),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (24, 17))},
    1: {  # Danopth
        'obstructed': {
            # Top-left rock/pit
            *product(range(3, 5), (0,)),
            *product(range(2, 6), (1,)),
            *product(range(1, 12), range(2, 4)),
            *product(range(1, 11), (4,)),
            *product(range(3, 5), (5,)),
            # Top-middle rock
            *product(range(19, 23), range(0, 2)),
            # Bottom-left rock
            *product(range(0, 2), range(15, 19)),
            *product((0,), range(19, 21)),
            *product(range(0, 7), (21,)),
            # Bottom-right rock
            *product((31,), (15,)),
            *product(range(29, 32), range(16, 19)),
            *product(range(30, 32), (19,)),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (24, 17))},
    2: {  # Alkonost
        'obstructed': {
            # Left tree column
            *product((0,), range(0, 22)),
            # Top-left trees
            *product(range(1, 6), range(0, 3)),
            *product(range(1, 5), (3,)),
            # Bottom-left trees
            *product((1,), range(14, 18)),
            *product((2,), range(15, 17)),
            *product((1,), range(20, 22)),
            # Middle trees
            (14, 10),
            *product(range(12, 16), (11,)),
            *product(range(9, 16), range(12, 15)),
            *product(range(10, 16), (15,)),
            # Top-mid trees
            *product(range(13, 19), (0,)),
            *product(range(14, 18), (1,)),
            # Right trees
            *product((29,), range(4, 6)),
            *product((30,), range(2, 7)),
            *product((31,), range(2, 10)),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (24, 17))},
    3: {  # Sikutar
        'obstructed': {
            # Left ravine
            *product((0,), range(6, 8)),
            *product(range(0, 4), (8,)),
            *product(range(7, 9), (8,)),
            *product(range(0, 9), range(9, 12)),
            *product(range(0, 2), (12,)),
            # Middle snow
            *product(range(13, 16), range(7, 10)),
            # Top-right water
            *product(range(24, 32), (0,)),
            *product(range(26, 32), (0,)),
            # Right rocks
            (31, 11),
            *product(range(30, 32), range(12, 15)),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (24, 17))},
    4: {  # Hephaestus IV
        'obstructed': {
            # Bottom-left lava
            *product((0,), range(14, 20)),
            *product((1,), range(15, 19)),
            # Middle lava
            *product(range(10, 13), range(4, 8)),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (24, 17))},
    5: {  # Flidais
        'obstructed': {
            # Bottom-left creeper
            *product(range(0, 2), range(16, 18)),
            # Middle creeper
            *product(range(10, 17), range(7, 10)),
            *product(range(10, 16), (10,)),
            # Right creeper
            *product(range(26, 29), (11,)),
            *product((26,), range(12, 14)),
            *product(range(27, 32), range(12, 16)),
            (31, 16),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (24, 17))},
    6: {  # 63 Corvi
        'obstructed': {
            # Top wall
            *product(range(0, 32), (0,)),
            *product(range(0, 8), (1,)),
            # Raised platform's edges
            *product(range(9, 11), range(1, 8)),
            *product(range(9, 32), range(8, 10)),
            # Right end of platform
            *product(range(28, 32), range(1, 8)),
        }},
    7: {  # Sandbox
        'obstructed': {
            # Top-left rock
            *product(range(10), (0,)),
            *product(range(9), (1,)),
            *product(range(8), (2,)),
            *product(range(6), (3,)),
            (0, 4),
            # Top-right rock
            *product(range(25, 32), (0,)),
            *product(range(26, 32), (1,)),
            *product(range(27, 32), (2,)),
            *product(range(28, 32), (3,)),
            *product(range(29, 32), (4,)),
            # Bottom-left rock
            *product((0,), range(19, 21)),
        },
        'random-input-zones': (('drag-arbitrary-input', (1, 4)),),
        'programmed-input': ('drag-programmed-input', (1, 13)),
        'fixed-input-zones': (('drag-arbitrary-input', (2, 18)),),
        'recycler': ('drag-recycler', (27, 17))},
    # TODO: I only did the most central terrain obstacles on my first pass through the main game; add the edge bits
    "An Introduction to Pipelines": {
        'obstructed': {
            # Middle rock
            *product(range(18, 21), (16,)),
            *product(range(14, 22), range(17, 19)),
            *product(range(15, 22), (19,)),
            *product(range(17, 21), (20,)),
        },
        'fixed-input-zones': (('drag-silo-input', (9, 4)),
                              ('drag-silo-input', (8, 12))),
        'output-zones': (('output', (23, 6)),
                         ('output', (23, 12)),)},
    "There's Something in the Fishcake": {
        'obstructed': {
            # Top-left rock
            *product(range(4, 6), (1,)),
            *product(range(3, 7), (2,)),
            *product(range(2, 8), range(3, 6)),
            *product(range(3, 7), (6,)),
            # Middle rock
            *product(range(17, 21), (9,)),
            *product(range(15, 22), (10,)),
            *product(range(14, 22), range(11, 13)),
            *product(range(16, 22), (13,)),
            # Wreckage
            *product(range(24, 27), range(18, 20)),
            # TODO: Edge bits
        },
        'fixed-input-zones': (('drag-silo-input', (6, 8)),
                              ('drag-silo-input', (7, 16))),
        'output-zones': (('output', (23, 6)),)},
    "Sleepless on Sernimir IV": {
        'obstructed': {
            # Top rock
            *product(range(17, 21), (2,)),
            *product(range(15, 22), (3,)),
            *product(range(14, 22), range(4, 6)),
            *product(range(15, 22), (6,)),
        },
        'fixed-input-zones': (('drag-silo-input', (4, 2)),
                              ('drag-silo-input', (5, 9)),
                              ('drag-silo-input', (6, 16))),
        'output-zones': (('output', (23, 8)),)},
    "Settling into the Routine": {
        'obstructed': {},
        'fixed-input-zones': (('drag-oceanic-input', (8, 3)),),
        'output-zones': (('output', (23, 12)),
                         ('output', (23, 18)))},
    "Nothing Works": {
        'obstructed': {
            # Rock
            *product(range(20, 23), (9,)),
            *product(range(19, 23), range(10, 12)),
            *product(range(20, 23), (12,)),
        },
        'fixed-input-zones': (('drag-oceanic-input', (8, 10)),
                              ('drag-atmospheric-input', (5, 19))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 18)))},
    "Challenge: In-Place Swap": {
        'obstructed': {
            # Bottom-left stuff
            *product(range(3, 6), (17,)),
            *product(range(0, 7), range(18, 20)),
            *product(range(0, 8), (20,)),
            *product(range(0, 9), (21,)),
            # Bottom-middle rock
            *product(range(13, 16), (18,)),
            *product(range(12, 17), (19,)),
            *product(range(11, 19), (20,)),
            *product(range(10, 22), (21,)),
        },
        'fixed-input-zones': (('drag-atmospheric-input', (5, 4)),
                              ('drag-oceanic-input', (1, 16))),
        'output-zones': (('output', (23, 12)),
                         ('output', (23, 18),))},
    "No Ordinary Headache": {
        'obstructed': {
            # Middle trees
            *product(range(12, 15), (3,)),
            *product(range(11, 16), (4,)),
            *product(range(9, 16), range(5, 8)),
            *product(range(10, 16), (8,)),
        },
        'random-input-zones': (('drag-atmospheric-input', (8, 18)),),
        'output-zones': (('output', (23, 2)),),
        'recycler': ('drag-recycler', (25, 14))},
    "No Thanks Necessary": {
        'obstructed': {},
        'random-input-zones': (('drag-atmospheric-input', (4, 5)),
                               ('drag-atmospheric-input', (7, 18)),),
        'output-zones': (('output', (22, 7)),
                         ('output', (21, 17)),),
        'recycler': ('drag-recycler', (26, 1))},
    "Challenge: Going Green": {
        'obstructed': {
            # Middle trees
            *product(range(12, 15), (10,)),
            *product(range(10, 16), range(11, 16)),
            *product((9,), range(12, 15)),
        },
        'random-input-zones': (('drag-powerplant-input', (-7, 4)),),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 18)))},
    "Falling": {
        'obstructed': {
            # Top-left water
            *product(range(0, 2), (4,)),
            *product(range(0, 3), (5,)),
            *product(range(0, 9), range(6, 10)),
            *product(range(0, 2), (10,)),
            # Middle ice
            *product(range(13, 16), range(5, 8)),
        },
        'fixed-input-zones': (('drag-oceanic-input', (4, 4)),
                              ('drag-silo-input', (0, 15))),
        'output-zones': (('output', (23, 7)),
                         ('output', (23, 12)),
                         ('output', (23, 17))),
        'recycler': ('drag-recycler', (26, 1))},
    "Challenge: Applied Fusion": {
        'obstructed': {
            # Top-left water
            # You can tell Zach did these by hand since the same ravine as in falling is traced slightly differently
            *product(range(0, 3), range(11, 13)),
            *product(range(0, 9), range(13, 17)),
            *product(range(0, 2), (17,)),
            # Middle ice
            *product(range(13, 16), range(12, 8)),
        },
        'fixed-input-zones': (('drag-oceanic-input', (4, 11)),),
        'output-zones': (('output', (23, 18)),),
        'recycler': ('drag-recycler', (26, 1))},
    "Molecular Foundry": {
        'obstructed': {
            # Middle rock
            *product(range(10, 13), range(11, 15)),
            # Wreckage
            *product(range(2, 5), range(18, 20)),
        },
        'random-input-zones': (('drag-mining-input', (1, 7)),),
        'output-zones': (('output', (23, 18)),),
        'recycler': ('drag-recycler', (22, 3))},
    "Gas Works Park": {
        'obstructed': {
            # Middle rock
            *product(range(10, 13), range(4, 8)),
        },
        'random-input-zones': (('drag-mining-input', (1, 7)),),
        'fixed-input-zones': (('drag-silo-input', (3, 17)),),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 18)))},
    "Challenge: KOHCTPYKTOP": {
        'obstructed': {
            # Middle rock
            *product(range(10, 13), range(4, 8)),
        },
        'fixed-input-zones': (('drag-mining-input', (3, 3)),
                              ('drag-silo-input', (2, 10)),
                              ('drag-silo-input', (2, 17))),
        'output-zones': (('output', (23, 2)),
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('drag-recycler', (26, 17))},
    "Ω-Pseudoethyne": {'obstructed': {}},  # Flidais components can't be added/modified so terrain may be empty
    "Σ-Ethylene": {'obstructed': {}},
    "Teleporters": {
        'obstructed': {
            # Top wall
            *product(range(0, 32), (0,)),
            *product(range(0, 8), (1,)),
            # Raised platform's edges
            *product(range(9, 11), range(1, 8)),
            *product(range(9, 32), range(8, 10)),
            # Right end of platform
            *product(range(28, 32), range(1, 8)),
            # TODO: Very hacky
            # Due to our simplifying assumptions aboout component pipe positions, we made drag-spaceship-input one row
            # shorter than it should be to handle its pipe being in its top row
            # Add terrain where the rest of it should be
            *product(range(4, 6), (20,)),
        },
        'fixed-input-zones': (('drag-spaceship-input', (4, 18)),),
        'output-zones': (('output', (23, 3)),),
        'recycler': ('drag-recycler', (26, 17))},
    "Precursor Compounds": {
        'obstructed': {
            # Top wall
            *product(range(0, 32), (0,)),
            *product(range(0, 8), (1,)),
            # Raised platform's edges
            *product(range(9, 11), range(1, 8)),
            *product(range(9, 32), range(8, 10)),
            # Right end of platform
            *product(range(28, 32), range(1, 8)),
            # TODO: Very hacky
            # See above note about drag-spaceship-input
            *product(range(1, 3), (11,)),
            *product(range(4, 6), (21,)),
        },
        'random-input-zones': (('drag-spaceship-input', (1, 9)),),
        'fixed-input-zones': (('drag-spaceship-input', (4, 19)),),
        'output-zones': (('output', (23, 12)),
                         ('output', (23, 18)))},
}
