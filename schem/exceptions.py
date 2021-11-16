#!/usr/bin/env python
# -*- coding: utf-8 -*-


# __module__ set in class definitions so they show as e.g. "ReactionError" instead of "schem.exceptions.ReactionError"
# in exception messages


class SolutionImportError(Exception):
    """Raised when a solution string cannot be successfully loaded into a level."""
    __module__ = Exception.__module__


class SolutionRunError(RuntimeError):
    """Raised when a solution encounters an error while being run."""
    __module__ = Exception.__module__


class ReactionError(SolutionRunError):
    """Raised if there is a molecule collision or a molecule is pulled apart in a reactor."""
    __module__ = Exception.__module__


class PauseException(SolutionRunError):
    """Raised when a Pause command is encountered by a waldo."""
    __module__ = Exception.__module__


class InvalidOutputError(SolutionRunError):
    """Raised when an invalid molecule is passed to a level output zone."""
    __module__ = Exception.__module__


class ScoreError(Exception):
    """Raised during validations if a solution's expected score does not match its actual score."""
    __module__ = Exception.__module__
