#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import copy
import gzip
import io
import json
import zlib

from spacechem.grid import Position
from spacechem.molecule import Molecule

# Level (NN input) bits:
# Features: 7 bits
#   Allowed instrs: 1 bit for each disable-able instruction (sensor, FF, fuse, split, swap): 5 bits
#      Also double as flags for whether sensor, fuser, splitter, and/or swappers are present.
#   0-3: 0, 2, 4, or 8 bonders (2 bits)
# Inputs: 480 bits
# 32 x Cell bits:
#    0-109 element or 0 if no atom = 7 bits
#    0-12: element's max bond count = 4 bits
#    0-3: rightward bonds = 2 bits
#    0-3: downward bonds = 2 bits
# Outputs: 501 bits
#   large_output bit: 1 bit for if it's a large output zone (cell def's unchanged if so)
#   each:
#       16 x cell bits = 240 bits
#       num outputs bits = 10 bits (allow up to 1024 outputs)

# Total level bits: 988 bits

class Level:
    __slots__ = 'dict', 'input_molecules', 'output_molecules', 'output_counts'

    '''Parent class for Research and Production levels.'''
    def __init__(self):
        self.dict = {}

    def __init__(self, level_export_string):
        '''Import a mission code string; decode b64 then uncompress gzip.'''
        self.dict = json.loads(zlib.decompress(base64.b64decode(level_export_string),
                                               wbits=16+15).decode('utf-8'))

    def __getitem__(self, item):
        return self.dict[item]

    def __setitem__(self, item, val):
        self.dict[item] = val

    def __str__(self):
        return json.dumps(self.dict)

    # def __init__(self, input_molecules, output_molecules,
    #              num_bonders=0,
    #              has_sensor=False,
    #              has_fuser=False, has_splitter=False,
    #              has_swapper=False,
    #              has_flip_flops=True):
    #     self.input_molecules = input_molecules # None as needed
    #     self.output_molecules = output_molecules
    #
    #     self.num_bonders = num_bonders
    #     self.has_sensor = has_sensor
    #     self.has_fuser = has_fuser
    #     self.has_splitter = has_splitter
    #     self.has_swapper = has_swapper
    #     self.has_flip_flops = has_flip_flops

    def get_code(self):
        '''Export to mission code string; gzip then b64 the level json.'''
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w") as f:
            f.write(json.dumps(self.dict).encode('utf-8'))
        return base64.b64encode(out.getvalue()).decode()

    def get_name(self):
        return self.dict['name']

    def get_input_molecule(self, input_idx):
        '''Return a new copy of the given input index's molecule, or None if the input is unused.'''
        return copy.deepcopy(self.input_molecules[input_idx])

    def get_output_molecule(self, output_idx):
        '''Return the given output index's molecule, or None if the output is unused.'''
        return self.output_molecules[output_idx]  # Promise not to mutate it please thx


class ResearchLevel(Level):
    __slots__ = ()

    def __init__(self):
        super().__init__()
        self['input-zones'] = {}
        self['output-zones'] = {}

        self['has-large-output'] = False

        # Features of the level
        self['bonder-count'] = 0
        self['has-sensor'] = False
        self['has-fuser'] = False
        self['has-splitter'] = False
        self['has-teleporter'] = False

        self['type'] = 'research'
        self['name'] = 'Unknown'
        self['author'] = "Unknown"
        self['difficulty'] = 0

    def __init__(self, level_export_string):
        '''Import a mission code string; decode b64 then uncompress gzip.'''
        super().__init__(level_export_string)
        assert self['type'] == 'research'

        self.output_counts = [None, None]

        # Prepare input and output molecule objects
        self.input_molecules = [None, None]
        for i, input_dict in self['input-zones'].items():
            i = int(i)
            # TODO: Assuming non-random level for now (only one input molecule)

            # Input molecules have relative indices to within their zones, so let the ctor know if this is a beta input
            # zone molecule (will be initialized 4 rows downward)
            self.input_molecules[i] = Molecule.from_json_string(input_dict['inputs'][0]['molecule'], is_beta=(i == 1))

        # Even in a large output level we'll leave the second field as None in each case, to
        # simplify downstream code handling omega output commands (which can be put in large
        # output levels)
        self.output_molecules = [None, None]
        self.output_counts = [None, None]

        for i, output_dict in self['output-zones'].items():
            i = int(i)
            self.output_molecules[i] = Molecule.from_json_string(output_dict['molecule'])
            self.output_counts[i] = output_dict['count']

    def get_bonder_count(self):
        return self.dict['bonder-count']
