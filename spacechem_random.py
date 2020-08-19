#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Implementation of the .NET PRNG used by Spacechem. Copied with slight adaptation from:
https://github.com/csaboka/spacechempatch/blob/master/SpacechemPatch/Patches/AbstractForcedRandom.cs
whose license is included here as required:

"MIT License

Copyright (c) 2018 Csaba Varga

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."
'''

from typing import List

import numpy as np


class SpacechemRandom:
    '''Class for providing random values, which mimics the windows PRNG used by Spacechem.'''
    inext: int
    inextp: int
    SeedArray: List[np.int32]

    def __init__(self, seed: np.int32 = 0):
        '''Uses Spacechem's default seed of 0 if seed not specified.'''
        mj = np.int32(161803398) - (abs(seed) if (seed != np.iinfo(np.int32).min) else np.iinfo(np.int32).max)
        self.SeedArray = 56 * [np.int32(0)]
        self.SeedArray[55] = mj
        mk = np.int32(1)
        for i in range(1, 55):
            ii = (21 * i) % 55
            self.SeedArray[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += np.iinfo(np.int32).max
            mj = self.SeedArray[ii]

        for k in range(1, 5):
            for i in range(1, 56):
                self.SeedArray[i] -= self.SeedArray[1 + ((i + 30) % 55)]
                if self.SeedArray[i] < 0:
                    self.SeedArray[i] += np.iinfo(np.int32).max

        self.inext = 0
        self.inextp = 21  # Mac used to use a different value here, unrelated to the 21 above

    def next(self, max_value) -> int:
        '''Given an int N, return a random value from 0 to N-1 inclusive.'''
        # This is also coming from the .NET Reference Source, merging the logic of Next() and InternalSample().

        # Increment the SeedArray indices while wrapping to keep them in [1,55]
        self.inext += 1
        if self.inext >= 56:
            self.inext = 1
        self.inextp += 1
        if self.inextp >= 56:
            self.inextp = 1

        ret_val: np.int32 = self.SeedArray[self.inext] - self.SeedArray[self.inextp]

        # The following line isn't present in the old Mono sources, but without it, it's possible to return the maximum
        # value and violate the contract.
        if ret_val == np.iinfo(np.int32).max:
            ret_val -= 1

        if ret_val < 0:
            ret_val += np.iinfo(np.int32).max

        self.SeedArray[self.inext] = ret_val

        return int(ret_val * (1 / np.iinfo(np.int32).max) * max_value)
