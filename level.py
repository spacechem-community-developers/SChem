#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import gzip
import io
from itertools import product
import json
import zlib

from spacechem.molecule import Molecule
from spacechem.spacechem_random import SpacechemRandom


OVERWORLD_ROWS = 22
OVERWORLD_COLS = 32

# Terrain maps. The positions of obstacles within the level is uniquely identified by the terrain type for
# all ResearchNet and custom levels, but each main game level has a unique terrain layout.
# This dict will store, for each type indexed by either its level name (main game) or its
# integer terrain type (ResearchNet/custom), the following:
# * Set containing all co-ordinates obstructed by terrain obstacles (rocks, etc.)
# * lists of where inputs, outputs, and other overworld buildings will be spawned if present


# itertools.product used to somewhat more compactly specify the co-ordinates in every obstacle
TERRAIN_MAPS = {
    'research':
        {'obstructed': {},
         'input-zones': (('research-input', (0, 1)),  # hacks hacks hacks
                         ('research-input', (0, 2))),
         'output-zones': (('research-output', (7, 0)),
                          ('research-output', (7, 1)),)},  # :thonk:
    0: {'obstructed': {
            # Mid-left rock
            *product(range(4, 6), (8,)),
            *product(range(3, 7), (9,)),
            *product(range(2, 8), range(10, 13)),
            *product(range(5, 7), (13,)),
            # Bottom rock
            },
        'fixed-input-zones': (('drag-arbitrary-input', (3, 13)),
                              ('drag-arbitrary-input', (3, 18))),
        'random-input-zones': (('drag-arbitrary-input', (1, 5)),),
        'output-zones': (('output', (23, 2)),  # TODO: Might be able to hardcode outputs on column 23...
                         ('output', (23, 7)),
                         ('output', (23, 12))),
        'recycler': ('recycler', (24, 17))},
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
    "Molecular Foundry": {
        'obstructed': {},
        'random-input-zones': (('drag-mining-input', (1, 7)),),
        'output-zones': (('output', (23, 18)),),
        'recycler': ('recycler', (22, 3))}}


class Level:
    '''Parent class for Research and Production levels. Level(code) will return an instance of whichever subclass
    is appropriate.
    '''
    __slots__ = ('dict', 'input_molecules', 'output_molecules', 'output_counts', 'input_random_generators',
                 'input_random_buckets')

    def __new__(cls, code):
        '''Return an instance of ResearchLevel or ProductionLevel as appropriate based on the given level code.'''
        d = json.loads(zlib.decompress(base64.b64decode(code), wbits=16+15).decode('utf-8'))
        # TODO: There doesn't seem to be a clean way to have Level(code) return an instance of the appropriate subclass,
        #       while also allowing ResearchLevel(code) and ProductionLevel(code) to work, without duplicating some
        #       conversion to/from the level code or else duplicating a lot of init logic.
        if d['type'] == 'research':
            return super().__new__(ResearchLevel)
        elif d['type'] == 'production':
            return super().__new__(ProductionLevel)
        else:
            raise ValueError(f"Unrecognized level type {d['type']}")

    def __init__(self, code=None):
        self.dict = json.loads(zlib.decompress(base64.b64decode(code), wbits=16+15).decode('utf-8'))

        if code is None:
            self['name'] = 'Unknown'
            self['author'] = 'Unknown'
            self['difficulty'] = 0

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, item, val):
        self.dict[item] = val

    def __contains__(self, item):
        return item in self.dict

    def __str__(self):
        return json.dumps(self.dict)

    def get_code(self):
        '''Export to mission code string; gzip then b64 the level json.'''
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w") as f:
            f.write(json.dumps(self.dict).encode('utf-8'))
        return base64.b64encode(out.getvalue()).decode()

    def get_name(self):
        return self.dict['name']


class ResearchLevel(Level):
    __slots__ = ()

    # Basic __new__ implementation required so that Level.__new__ can be smart
    def __new__(cls, code=None):
        return object.__new__(cls)

    def __init__(self, code=None):
        super().__init__(code)

        if code is None:
            self['type'] = 'research'
            self['input-zones'] = {}
            self['output-zones'] = {}

            self['has-large-output'] = False

            # Features of the level
            self['bonder-count'] = 0
            self['has-sensor'] = False
            self['has-fuser'] = False
            self['has-splitter'] = False
            self['has-teleporter'] = False

        assert self['type'] == 'research'

        self.input_molecules = {}
        self.input_random_generators = {}
        self.input_random_buckets = {}
        for i, input_zone_dict in self['input-zones'].items():
            i = int(i)

            if len(input_zone_dict['inputs']) > 1:
                # Create a random generator for this zone (note that all zones use the same seed and thus sequence)
                self.input_random_generators[i] = SpacechemRandom()
                self.input_random_buckets[i] = []  # Bucket of indices for the molecules in the current balancing bucket

            # Construct one of each molecule from the input JSON
            # Input molecules have relative indices to within their zones, so let the ctor know if this is a beta input
            # zone molecule (will be initialized 4 rows downward)
            self.input_molecules[i] = [Molecule.from_json_string(input_dict['molecule'])
                                       for input_dict in input_zone_dict['inputs']]

        self.output_molecules = {}
        self.output_counts = {}

        for i, output_dict in self['output-zones'].items():
            i = int(i)
            self.output_molecules[i] = Molecule.from_json_string(output_dict['molecule'])
            self.output_counts[i] = output_dict['count']

    def get_bonder_count(self):
        return self.dict['bonder-count']


class ProductionLevel(Level):
    __slots__ = ()

    # Basic __new__ implementation required so that Level.__new__ can be smart
    def __new__(cls, code=None):
        return object.__new__(cls)

    def __init__(self, code=None):
        super().__init__(code)

        if code is None:
            self['type'] = 'production'
            self['random-input-zones'] = {}
            self['fixed-input-zones'] = {}
            self['output-zones'] = {}
            self['terrain'] = 0
            self['max-reactors'] = 0
            self['has-starter'] = False
            self['has-assembly'] = False
            self['has-disassembly'] = False
            self['has-advanced'] = False
            self['has-nuclear'] = False
            self['has-recycler'] = False

        assert self['type'] == 'production'
