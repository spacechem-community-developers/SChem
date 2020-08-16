#!/usr/bin/env python
# -*- coding: utf-8 -*-

class RunSuccess(Exception):
    pass


class InfiniteLoopError(Exception):
    pass


class InvalidOutputError(Exception):
    pass


class ReactionError(Exception):
    pass
