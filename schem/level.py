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


# TODO: TO reduce the complexity of Solution.__init__, provide something like a Level.default_components() method
class Level:
    """Parent class for Research and Production levels. Level(code) will return an instance of whichever subclass
    is appropriate.
    """
    export_line_len = 74
    __slots__ = 'dict', 'resnet_id'

    @classmethod
    def code_to_json(cls, code):
        try:
            return json.loads(zlib.decompress(base64.b64decode(code), wbits=16+15).decode('utf-8'))
        except Exception as e:
            raise ValueError("String is not a valid-format SpaceChem level code") from e

    def __new__(cls, code):
        """Return an instance of ResearchLevel or ProductionLevel as appropriate based on the given level code."""
        # If this is being called from a child class, behave like a normal __new__ implementation (to avoid recursion)
        if cls != Level:
            return object.__new__(cls)

        d = cls.code_to_json(code)
        # TODO: There doesn't seem to be a clean way to have Level(code) return an instance of the appropriate subclass,
        #       while also allowing ResearchLevel(code) and ProductionLevel(code) to work, without duplicating some
        #       conversion to/from the level code or else duplicating a lot of init logic.
        if d['type'].startswith('research'):
            return super().__new__(ResearchLevel)
        elif d['type'].startswith('production'):
            return super().__new__(ProductionLevel)
        elif d['type'].startswith('sandbox'):
            return super().__new__(SandboxLevel)
        else:
            raise ValueError(f"Unrecognized level type {d['type']}")

    def __init__(self, code=None):
        if code is None:
            self.dict = {}
            self['name'] = 'Unknown'
            self['author'] = 'Unknown'
            self['difficulty'] = 0
        else:
            self.dict = self.code_to_json(code)

        # Since unfortunately some official levels share names, sometimes we want to track which volume/issue/puzzle the
        # level is, if it's a ResNet level.
        self.resnet_id = None

    def __eq__(self, other):
        return isinstance(other, Level) and self.dict == other.dict

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
        """Export to mission code string; gzip then b64 the level json."""
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
        self.dict['name'] = s

    @property
    def type(self):
        return self.dict['type']
    @type.setter
    def type(self, s):
        assert isinstance(s, str), "Level type must be a string"
        self.dict['type'] = s

    # TODO: More properties


class ResearchLevel(Level):
    __slots__ = ()

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

    def output_molecules(self):
        """Return a list of Molecule objects demanded by this level."""
        return [Molecule.from_json_string(output_dict['molecule'])
                for _, output_dict in sorted(self['output-zones'].items())]


class ProductionLevel(Level):
    __slots__ = ()

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


class SandboxLevel(Level):
    __slots__ = ()

    def __init__(self, code=None):
        super().__init__(code)

        if code is None:
            self['type'] = 'sandbox'
            self['random-input-zones'] = {}
            self['fixed-input-zones'] = {}
            self['programmed-input-start'] = []
            self['programmed-input-repeat'] = []
            self['programmed-input-molecules'] = {}

        assert self['type'] == 'sandbox'
