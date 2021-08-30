#!/usr/bin/env python
# -*- coding: utf-8 -*-


# __module__ set in class definitions so they show as e.g. "ReactionError" instead of "schem.exceptions.ReactionError"
# in exception messages

class InfiniteLoopError(Exception):
    __module__ = Exception.__module__


class InvalidOutputError(Exception):
    __module__ = Exception.__module__


class ReactionError(Exception):
    __module__ = Exception.__module__


class PauseException(Exception):
    """Raised when a Pause command is encountered by a waldo"""
    __module__ = Exception.__module__


class ScoreError(Exception):
    """Raised during validations if solution's expected score does not match."""
    __module__ = Exception.__module__
