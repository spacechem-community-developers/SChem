#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import copy
import gzip
import io
import json
import zlib

from spacechem.molecule import Molecule
from spacechem.spacechem_random import SpacechemRandom


class Level:
    '''Parent class for Research and Production levels.'''
    __slots__ = ('dict', 'input_molecules', 'output_molecules', 'output_counts', 'input_random_generators',
                 'input_random_buckets')

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

    def get_input_molecule_idx(self, input_idx):
        '''Given an input zone index, get the next input molecule's index. Exposed to allow for tracking branches in
        random level states.
        '''
        if len(self.input_molecules[input_idx]) > 1:
            # Create the next balance bucket if we've run out.
            # The bucket stores an index identifying one of the 2-3 molecules
            if not self.input_random_buckets[input_idx]:
                for mol_idx, mol_dict in enumerate(self['input-zones'][str(input_idx)]['inputs']):
                    self.input_random_buckets[input_idx].extend(mol_dict['count'] * [mol_idx])

            # Randomly remove one entry from the bucket and return it
            bucket_idx = self.input_random_generators[input_idx].next(len(self.input_random_buckets[input_idx]))
            return self.input_random_buckets[input_idx].pop(bucket_idx)
        else:
            return 0

    def get_input_molecule(self, input_idx):
        '''Return a new copy of the given input index's molecule, or None if the input is unused.
        Randomly choose from the input's molecules if there are multiple.
        '''
        return copy.deepcopy(self.input_molecules[input_idx][self.get_input_molecule_idx(input_idx)])

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
            self.input_molecules[i] = [Molecule.from_json_string(input_dict['molecule'], zone_idx=i)
                                       for input_dict in input_zone_dict['inputs']]

        self.output_molecules = {}
        self.output_counts = {}

        for i, output_dict in self['output-zones'].items():
            i = int(i)
            self.output_molecules[i] = Molecule.from_json_string(output_dict['molecule'], zone_idx=i)
            self.output_counts[i] = output_dict['count']

    def get_bonder_count(self):
        return self.dict['bonder-count']
