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

class PauseException(Exception):
    """Raised when a Pause command is encountered by a waldo"""
    pass
