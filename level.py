#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import gzip
import io
import json
import zlib

from .molecule import Molecule

OVERWORLD_ROWS = 22
OVERWORLD_COLS = 32


class Level:
    '''Parent class for Research and Production levels. Level(code) will return an instance of whichever subclass
    is appropriate.
    '''
    export_line_len = 74
    __slots__ = 'dict',

    def __new__(cls, code):
        '''Return an instance of ResearchLevel or ProductionLevel as appropriate based on the given level code.'''
        d = json.loads(zlib.decompress(base64.b64decode(code), wbits=16+15).decode('utf-8'))
        # TODO: There doesn't seem to be a clean way to have Level(code) return an instance of the appropriate subclass,
        #       while also allowing ResearchLevel(code) and ProductionLevel(code) to work, without duplicating some
        #       conversion to/from the level code or else duplicating a lot of init logic.
        if d['type'].startswith('research'):
            return super().__new__(ResearchLevel)
        elif d['type'].startswith('production'):
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

    @property
    def code(self):
        '''Export to mission code string; gzip then b64 the level json.'''
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w") as f:
            f.write(json.dumps(self.dict).encode('utf-8'))
        code = base64.b64encode(out.getvalue()).decode()
        # Line-wrap the export code for readability
        return '\n'.join(code[i:i+self.export_line_len] for i in range(0, len(code), self.export_line_len))

    @property
    def name(self):
        return self.dict['name']
    @name.setter
    def name(self, s):
        assert isinstance(s, str), "Level name must be a string"
        self.dict['name'] == s


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

        assert self['type'].startswith('research')


    def get_bonder_count(self):
        return self.dict['bonder-count']

    def output_molecules(self):
        '''Return a list of Molecule objects demanded by this level.'''
        return [Molecule.from_json_string(output_dict['molecule'])
                for _, output_dict in sorted(self['output-zones'].items())]


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

        assert self['type'].startswith('production')
