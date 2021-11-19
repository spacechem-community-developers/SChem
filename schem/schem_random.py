#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Implementation of the .NET PRNG used by Spacechem. Copied with slight adaptation from:
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
"""

from typing import List

import numpy as np


class SChemRandom:
    """Class for providing random values, which mimics the windows PRNG used by Spacechem."""
    __slots__ = 'inext', 'inextp', 'SeedArray'

    inext: int
    inextp: int
    SeedArray: List[np.int32]

    MAX_SEED = 161_803_398  # If seeds higher than this are given the PRNG is liable to produce overflow errors

    INT32_MIN = np.iinfo(np.int32).min  # -2147483648
    INT32_MAX = np.iinfo(np.int32).max  # 2147483647

    def __init__(self, seed=0):
        """Initialize the PRNG. Seed must be from 0 to 161,803,398 (inclusive); default 0."""
        # The PRNG originally accepted negative seeds, but since all it did was use their absolute value, I think it's
        # less misleading to just explicitly disallow them. The upper bound here is because the below constructor math
        # can overflow if given values above the constant used to set `mj`.
        if not (0 <= seed <= self.MAX_SEED):
            raise ValueError(f"Seed must be from 0 to {self.MAX_SEED}.")

        seed = np.int32(seed)
        mj = np.int32(self.MAX_SEED) - seed
        self.SeedArray = 55 * [np.int32(0)]
        self.SeedArray[54] = mj
        mk = np.int32(1)
        for n in range(1, 55):
            i = ((21 * n) % 55) - 1  # Note: 21n % 55 can never be 0 for 1 <= n <= 54 since 21 and 55 are coprime
            self.SeedArray[i] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.INT32_MAX
            mj = self.SeedArray[i]

        for _ in range(4):
            for i in range(55):
                self.SeedArray[i] -= self.SeedArray[(i + 31) % 55]
                if self.SeedArray[i] < 0:
                    self.SeedArray[i] += self.INT32_MAX

        self.inext = 54
        self.inextp = 20  # Mac used to use a different value here (note: was 21 before I cleaned the array indices)

    def next(self, max_value) -> int:
        """Given an int N, return a random value from 0 to N-1 inclusive."""
        # This is also coming from the .NET Reference Source, merging the logic of Next() and InternalSample().

        # Increment the SeedArray indices while wrapping to keep them in [0,54]
        self.inext = (self.inext + 1) % 55
        self.inextp = (self.inextp + 1) % 55

        ret_val: np.int32 = self.SeedArray[self.inext] - self.SeedArray[self.inextp]

        # The following line isn't present in the old Mono sources, but without it, it's possible to return the maximum
        # value and violate the contract.
        if ret_val == self.INT32_MAX:
            ret_val -= 1

        if ret_val < 0:
            ret_val += self.INT32_MAX

        self.SeedArray[self.inext] = ret_val

        return int(ret_val * (1 / self.INT32_MAX) * max_value)
