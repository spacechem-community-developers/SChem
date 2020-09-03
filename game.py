#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import math
import time

from spacechem.exceptions import *
from spacechem.grid import Position, Direction
from spacechem.level import ResearchLevel
from spacechem.molecule import ATOM_RADIUS
from spacechem.spacechem_random import SpacechemRandom
from spacechem.solution import InstructionType, Solution
from spacechem.tests import test_data

NUM_MOVE_CHECKS = 10  # Number of times to check for collisions during molecule movement


class Waldo:
    __slots__ = 'idx', 'instr_map', 'flipflop_states', 'position', 'direction', 'molecule', 'is_stalled', 'is_rotating'

    def __init__(self, idx, position, direction, instr_map):
        self.idx = idx
        self.instr_map = instr_map  # Position map of tuples containing arrow and 'command' (non-arrow) instructions
        self.flipflop_states = {posn: False for posn, (_, cmd) in instr_map.items()
                                if cmd is not None and cmd.type == InstructionType.FLIP_FLOP}
        self.position = position
        self.direction = direction
        self.molecule = None
        self.is_stalled = False  # Updated if the waldo is currently stalled on a sync, output, wall, etc.
        self.is_rotating = False  # Used to distinguish the first vs second cycle on a rotate command

    def __hash__(self):
        '''A waldo's run state is uniquely identified by its position, direction,
        whether it's holding a molecule, whether it's rotating, and its flip-flop states.
        '''
        # TODO: It is preferable if a hash not only uniquely identifies a cycle's behavior, but (at least in a
        #       deterministic solution) a cycle's behaviour will only be the same if it started with the same hash.
        #       Having waldo direction in here breaks that nicety, as e.g. ending on an arrow will not have the same
        #       hash as ending on that arrow when arriving from a different direction (even though for both the next
        #       cycle will have the waldo moving in the same direction). To avoid needing to jerry-rig the hash value,
        #       the simplest fix is to do hash checks after updating the direction each cycle (maybe even after instant
        #       instructions? need to think about how this affects the random input state tree. I think we want to
        #       snapshot right before calling any inputs so that we can guarantee we will encounter the hash before
        #       a certain input no matter what the previous cycle's position/direction was, if it ends on that input
        #       command and facing the same way)
        #       That said the current code *should* work, it's just making more random input nodes/segments than it
        #       needs to and/or detecting loops 1 cycle later than necessary...
        return hash((self.position, self.direction, self.molecule is None, self.is_rotating,
                     tuple(self.flipflop_states.values())))

    def __repr__(self):
        return f'Waldo({self.idx}, pos={self.position}, dir={self.direction}, is_stalled={self.is_stalled})'

    def cur_cmd(self):
        '''Return a waldo's current 'command', i.e. non-arrow instruction, or None.'''
        return None if self.position not in self.instr_map else self.instr_map[self.position][1]


class StateTreeSegment:
    __slots__ = 'cycles', 'output_cycles', 'next_node'

    def __init__(self):
        self.cycles = 0
        self.output_cycles = [[], []]
        self.next_node = None


class Reactor:
    '''Represents the current state of a running solution.
    To track state properly, requires that exec_cmd(waldo) be called with each of its waldos on each
    cycle for which that waldo has a cmd, followed by move(waldo) for each waldo each cycle.
    Might be able to skip ahead to cycles when either waldo has a command... but on the other hand
    if this is coded right it won't make a huge difference.
    '''
    # For convenience during float-precision rotation co-ordinates, we consider the center of the
    # top-left cell to be at (0,0), and hence the top-left reactor corner is (-0.5, -0.5).
    # Further, treat the walls as being one atom radius closer, so that we can efficiently check if an atom will collide
    # with them given only the atom's center co-ordinates
    walls = {Direction.UP: -0.5 + ATOM_RADIUS, Direction.DOWN: 7.5 - ATOM_RADIUS,
             Direction.LEFT: -0.5 + ATOM_RADIUS, Direction.RIGHT: 9.5 - ATOM_RADIUS}

    __slots__ = ('level', 'solution', 'cycle', 'waldos', 'molecules', 'did_input_this_cycle',
                 'did_output_this_cycle', 'completed_output_counts', 'bonder_pairs',
                 'prior_states', 'state_tree_segments', 'state_tree_nodes',
                 'cur_state_tree_segment_idx', 'cycles_to_new_state',
                 'incoming_inputs', 'last_random_molecule_indices',
                 'debug')

    def __init__(self, solution, debug=None):
        self.debug = debug  # For convenience

        self.level = solution.level
        self.solution = solution
        self.cycle = 0

        # Track whether each input zone has already done its max of 1 input per cycle
        self.did_input_this_cycle = [False, False]

        # Global output vars
        self.completed_output_counts = {i: 0 for i in self.level.output_counts.keys()}
        # Track whether each input zone has already done its max of 1 output per cycle
        self.did_output_this_cycle = [False, False]

        # Store molecules as dict keys to be ordered (preserving Spacechem's hidden
        # 'least recently modified' rule) and to have O(1) add/delete.
        # Values are ignored.
        self.molecules = {}

        # Pre-process the soln into red and blue instr maps that allow for skipping along rows/columns
        # to the next arrow/instruction. Ignores Start commands.
        self.waldos = [Waldo(i,
                             solution.waldo_starts[i][0],
                             solution.waldo_starts[i][1],
                             solution.waldo_instr_maps[i])
                       for i in range(len(solution.waldo_starts))]

        # For convenience/performance, pre-compute a list of (bonder_A, bonder_B, direction) triplets, sorted in the
        # order that bonds or debonds should occur in
        self.bonder_pairs = tuple((posn, neighbor_posn, direction)
                                  for posn in solution.bonders
                                  for neighbor_posn, direction in
                                  sorted([(posn + direction, direction)
                                          for direction in (Direction.RIGHT, Direction.DOWN)
                                          if posn + direction in solution.bonders],
                                         key=lambda x: solution.bonders[x[0]]))

        # State fast-forwarding related vars
        self.prior_states = {}
        self.state_tree_segments = [StateTreeSegment()]  # Tree starts with one segment
        self.cur_state_tree_segment_idx = 0
        self.state_tree_nodes = []
        self.last_random_molecule_indices = [None, None]
        self.cycles_to_new_state = 0
        # To allow the state fast-forwarding algorithm to look ahead for known loops, we store any random input
        # indices it drew early and use them up first
        self.incoming_inputs = [[], []]

    def __hash__(self):
        '''Hash of the current reactor state. Ignores cycle/output counts.'''
        return hash((tuple(molecule.hashable_repr() for molecule in self.molecules),
                     tuple(self.waldos)))

    def __str__(self):
        '''Pretty-print this reactor.'''
        num_cols = 10
        num_rows = 8

        # 2 characters per atom + 1 space between atoms/walls (we'll use that space to show waldos)
        grid = [['   ' for _ in range(num_cols)] + [' '] for _ in range(num_rows)]

        # Map out the molecules in the reactor
        for molecule in self.molecules:
            for (r, c), atom in molecule.atom_map.items():
                # Round co-ordinates in case we are mid-rotate
                r, c = round(r), round(c)
                if grid[r][c] != '   ':
                    grid[r][c] = ' XX'  # Colliding atoms
                else:
                    grid[r][c] = f' {atom.element.symbol.rjust(2)}'

        # Represent waldos as |  | when not grabbing and (  ) when grabbing. Distinguishing red/blue isn't too important
        for waldo in self.waldos:
            r, c = waldo.position
            # This will overwrite the previous waldo sometimes but its not a noticeable issue
            if waldo.molecule is None:
                grid[r][c] = f'|{grid[r][c][1:]}'
                grid[r][c + 1] = f'|{grid[r][c + 1][1:]}'
            else:
                grid[r][c] = f'({grid[r][c][1:]}'
                grid[r][c + 1] = f'){grid[r][c + 1][1:]}'

        result = f" {num_cols * '___'}_ \n"
        for row in grid:
            result += f"|{''.join(row)}|\n"
        result += f" {num_cols * '‾‾‾'}‾ \n"
        result += f"Cycle {self.cycle}"

        return result

    def check_molecule_collisions(self, molecule):
        '''Raise an exception if the given molecule collides with any other molecules.
        Assumes integer co-ordinates in all molecules.
        '''
        for other_molecule in self.molecules.keys():
            molecule.check_collisions(other_molecule)  # Implicitly ignores self

    def check_wall_collisions(self, molecule):
        '''Raise an exception if the given molecule collides with any walls.'''
        if not all(self.walls[Direction.UP] < p.row < self.walls[Direction.DOWN]
                   and self.walls[Direction.LEFT] < p.col < self.walls[Direction.RIGHT]
                   for p in molecule.atom_map):
            raise ReactionError("A molecule has collided with a wall")

    def check_collisions(self, molecule):
        '''Raise an exception if the given molecule collides with any other molecules or walls.
        Assumes integer co-ordinates in all molecules.
        '''
        self.check_molecule_collisions(molecule)
        self.check_wall_collisions(molecule)

    def check_collisions_fine(self, molecule):
        '''Check that the given molecule isn't colliding with any walls or other molecules.
        Raise an exception if it does.
        '''
        for other_molecule in self.molecules:
            molecule.check_collisions_fine(other_molecule)  # Implicitly ignores self

        self.check_wall_collisions(molecule)

    def exec_instrs(self, waldo):
        if waldo.position not in waldo.instr_map:
            return

        arrow_direction, cmd = waldo.instr_map[waldo.position]

        # Update the waldo's direction based on any arrow in this cell
        if arrow_direction is not None:
            waldo.direction = arrow_direction

        # Execute the non-arrow instruction
        if cmd is None:
            return
        elif cmd.type == InstructionType.INPUT:
            self.input(waldo, cmd.target_idx)
        elif cmd.type == InstructionType.OUTPUT:
            self.output(waldo, cmd.target_idx)
        elif cmd.type == InstructionType.GRAB:
            self.grab(waldo)
        elif cmd.type == InstructionType.DROP:
            self.drop(waldo)
        elif cmd.type == InstructionType.GRAB_DROP:
            if waldo.molecule is None:
                self.grab(waldo)
            else:
                self.drop(waldo)
        elif cmd.type == InstructionType.ROTATE:
            # If we are holding a molecule and weren't just rotating, start rotating
            # In all other cases, stop rotating
            waldo.is_rotating = waldo.molecule is not None and not waldo.is_rotating
            waldo.is_stalled = waldo.is_rotating
        elif cmd.type == InstructionType.BOND_PLUS:
            self.bond_plus()
        elif cmd.type == InstructionType.BOND_MINUS:
            self.bond_minus()
        elif cmd.type == InstructionType.SYNC:
            # Mark this waldo as stalled if both waldos aren't on a Sync
            other_waldo = self.waldos[1 - waldo.idx]
            waldo.is_stalled = other_waldo.cur_cmd() is None or other_waldo.cur_cmd().type != InstructionType.SYNC
        elif cmd.type == InstructionType.FUSE:
            pass
        elif cmd.type == InstructionType.SPLIT:
            pass
        elif cmd.type == InstructionType.SENSE:
            for posn in self.solution.sensors:
                molecule = self.get_molecule(posn)
                if molecule is not None and molecule.atom_map[posn].element.atomic_num == cmd.target_idx:
                    waldo.direction = cmd.direction
                    break
        elif cmd.type == InstructionType.FLIP_FLOP:
            # Update the waldo's direction if the flip-flop is on
            if waldo.flipflop_states[waldo.position]:
                waldo.direction = cmd.direction

            waldo.flipflop_states[waldo.position] = not waldo.flipflop_states[waldo.position]  # ...flip it
        elif cmd.type == InstructionType.SWAP:
            pass

    def input(self, waldo, input_idx):
        # If this input has already been called this cycle (i.e. blue inputs after red), stall this
        # waldo and do nothing
        if self.did_input_this_cycle[input_idx]:
            waldo.is_stalled = True
            return

        if self.incoming_inputs[input_idx]:
            # If our state fast-forwarder pulled any random input indices early, we have to use them up first
            molecule_idx = self.incoming_inputs[input_idx].pop(0)
        else:
            molecule_idx = self.level.get_input_molecule_idx(input_idx)

        # Store the last random input index we received for the convenience of the state-handling code
        if len(self.level.input_molecules[input_idx]) >= 2:
            self.last_random_molecule_indices[input_idx] = molecule_idx

        new_molecule = copy.deepcopy(self.level.input_molecules[input_idx][molecule_idx])
        self.molecules[new_molecule] = None  # Dummy value

        self.check_molecule_collisions(new_molecule)

        # Indicate that we inputted this cycle
        self.did_input_this_cycle[input_idx] = True

    def output(self, waldo, output_idx):
        # If the output is disabled, do nothing
        if self.level.output_counts[output_idx] is None:
            return

        # TODO: It'd be nice to only have to calculate this for molecules that have been
        #       debonded or dropped, etc. However, the cost of pre-computing it every time
        #       we do such an action is probably not worth the cost of just doing it once
        #       over all molecules whenever output is called.
        # TODO 2: On the other hand, solutions with a waldo wall-stalling on output just
        #         got fucked
        molecules_in_zone = iter(molecule for molecule in self.molecules
                                 # Ignore grabbed molecules
                                 if not any(waldo.molecule is molecule for waldo in self.waldos)
                                 and molecule.output_zone_idx(large_output=self.level['has-large-output']) == output_idx)
        molecule = next(molecules_in_zone, None)

        # Try to output the first molecule in the zone if an output hasn't already been done this cycle
        if not self.did_output_this_cycle[output_idx] and molecule is not None:
            if not molecule.isomorphic(self.level.get_output_molecule(output_idx)):
                raise InvalidOutputError("Invalid output molecule.")

            self.completed_output_counts[output_idx] += 1
            self.did_output_this_cycle[output_idx] = True

            # Check if we've won
            if all(self.completed_output_counts[i] >= self.level.output_counts[i]
                   for i in self.level.output_counts):
                raise RunSuccess()

            # Delete the molecule and check if there's another molecule in the zone (so we know whether to stall)
            outputted_molecule = molecule  # This awkward shuffle is to avoid deleting from the dict while iterating it
            molecule = next(molecules_in_zone, None)
            del self.molecules[outputted_molecule]

        # If there is any output(s) remaining in this zone, stall this waldo
        waldo.is_stalled = molecule is not None

    def grab(self, waldo):
        if waldo.molecule is None:
            waldo.molecule = self.get_molecule(waldo.position)

    def get_molecule(self, position):
        '''Select the molecule at the given grid position, or None if no such molecule.
        Used by Grab, Bond+/-, Fuse, etc.
        '''
        return next((molecule for molecule in self.molecules if position in molecule), None)

    def drop(self, waldo):
        waldo.molecule = None  # Remove the reference to the molecule

    def move_contents(self):
        '''Move all waldos in this reactor and any molecules they are holding.'''
        # If the waldo is facing a wall, mark it as stalled (may also be stalled due to sync, input, etc.)
        for waldo in self.waldos:
            if ((waldo.direction == Direction.UP and waldo.position.row == 0)
                    or (waldo.direction == Direction.DOWN and waldo.position.row == 7)
                    or (waldo.direction == Direction.LEFT and waldo.position.col == 0)
                    or (waldo.direction == Direction.RIGHT and waldo.position.col == 9)):
                waldo.is_stalled = True

        # If any waldo is about to rotate a molecule, don't skimp on collision checks
        if any(waldo.is_rotating for waldo in self.waldos):
            # If both waldos are holding the same molecule and either of them is rotating, a crash occurs
            # (even if they're in the same position and rotating the same direction)
            if self.waldos[0].molecule is self.waldos[1].molecule:
                raise ReactionError("Molecule pulled apart")

            # Otherwise, move each waldo's molecule partway at a time and check for collisions each time
            step_radians = math.pi / (2 * NUM_MOVE_CHECKS)
            step_distance = 1 / NUM_MOVE_CHECKS
            for i in range(NUM_MOVE_CHECKS):
                # Move all molecules currently being held by a waldo forward a step
                for waldo in self.waldos:
                    if waldo.molecule is not None and not waldo.is_stalled:
                        waldo.molecule.move_fine(waldo.direction, distance=step_distance)
                    elif waldo.is_rotating:
                        waldo.molecule.rotate_fine(pivot_pos=waldo.position,
                                                   direction=waldo.cur_cmd().direction,
                                                   radians=step_radians)

                # After moving all molecules, check each rotated molecule for collisions with walls or other molecules
                # Though all molecules had to move, only the rotating one(s) needs to do checks at each step, since we
                # know the other waldo will only have static molecules left to check against, and translation movements
                # can't clip through a static atom without ending on top of it
                # Note: This only holds true for <= 2 waldos and since we checked that at least one waldo is rotating
                for waldo in self.waldos:
                    if waldo.is_rotating:
                        self.check_collisions_fine(waldo.molecule)

            # After completing all steps of the movement, convert moved molecules back to integer co-ordinates and do
            # any final checks/updates
            for waldo in self.waldos:
                if waldo.molecule is not None and not waldo.is_stalled:
                    waldo.molecule.round_posns()
                    # Do the final check we skipped for non-rotating molecules
                    self.check_collisions(waldo.molecule)
                elif waldo.is_rotating:
                    waldo.molecule.round_posns()
                    # Rotate atom bonds
                    waldo.molecule.rotate_bonds(waldo.cur_cmd().direction)
        elif any(waldo.molecule is not None and not waldo.is_stalled for waldo in self.waldos):
            # If we are not doing any rotates, we can skip the full collision checks
            # Non-rotating molecules can cause collisions/errors if:
            # * The waldos are pulling a molecule apart
            # * OR The final destination of a moved molecule overlaps any other molecule after the move
            # * OR The final destination of a moved molecule overlaps the initial position of another moving molecule,
            #      and the offending waldos were not moving in the same direction

            # Given that either is moving, if the waldos share a molecule they must move in the same direction
            if (self.waldos[0].molecule is self.waldos[1].molecule
                    and (any(waldo.is_stalled for waldo in self.waldos)
                         or self.waldos[0].direction != self.waldos[1].direction)):
                raise ReactionError("A molecule has been grabbed by both waldos and pulled apart.")

            # Check if any molecule being moved will bump into the back of another moving molecule
            if (all(waldo.molecule is not None and not waldo.is_stalled
                    for waldo in self.waldos)
                    and self.waldos[0].direction != self.waldos[1].direction):
                for waldo in self.waldos:
                    # Intersect the target positions of this waldo's molecule with the current positions of the other
                    # waldo's molecules
                    other_waldo = self.waldos[1 - waldo.idx]
                    target_posns = set(posn + waldo.direction for posn in waldo.molecule.atom_map)
                    if not target_posns.isdisjoint(other_waldo.molecule.atom_map):
                        raise ReactionError("Molecule collision")

            # Move all molecules
            for waldo in self.waldos:
                if waldo.molecule is not None and not waldo.is_stalled:
                    waldo.molecule.move(waldo.direction)

            # Perform collision checks against the moved molecules
            for waldo in self.waldos:
                if waldo.molecule is not None and not waldo.is_stalled:
                    self.check_collisions(waldo.molecule)

        # Move waldos and mark them as no longer stalled. Note that is_rotated must be left alone to tell it not to
        # rotate twice
        for waldo in self.waldos:
            if not waldo.is_stalled:
                waldo.position += waldo.direction
            waldo.is_stalled = False

    def bond_plus(self):
        for position, neighbor_posn, direction in self.bonder_pairs:
            # Identify the molecule on each bonder (may be same, doesn't matter for now)
            molecule_A = self.get_molecule(position)
            if molecule_A is None:
                continue

            molecule_B = self.get_molecule(neighbor_posn)
            if molecule_B is None:
                continue

            atom_A = molecule_A[position]

            # If the bond being increased is already at the max bond size of 3, don't do
            # anything. However, due to weirdness of Spacechem's bonding algorithm, we still
            # mark the molecule as modified below
            if direction not in atom_A.bonds or atom_A.bonds[direction] != 3:
                atom_B = molecule_B[neighbor_posn]

                # Do nothing if either atom is at its bond limit (spacechem does not mark
                # any molecules as modified in this case unless the bond was size 3)
                if (sum(atom_A.bonds.values()) == atom_A.element.max_bonds
                        or sum(atom_B.bonds.values()) == atom_B.element.max_bonds):
                    continue

                direction_B = direction.opposite()

                if direction not in atom_A.bonds:
                    atom_A.bonds[direction] = 0
                atom_A.bonds[direction] += 1
                if direction_B not in atom_B.bonds:
                    atom_B.bonds[direction_B] = 0
                atom_B.bonds[direction_B] += 1

            if molecule_A is molecule_B:
                # Mark molecule as modified by popping it to the back of the reactor's queue
                del self.molecules[molecule_A]
                self.molecules[molecule_A] = None  # dummy value
            else:
                # Add the smaller molecule to the larger one (faster), then delete the smaller
                # and mark the larger as modified
                molecules = [molecule_A, molecule_B]
                molecules.sort(key=lambda x: len(x))
                molecules[1] += molecules[0]

                # Also make sure that any waldos holding the to-be-deleted molecule are updated
                # to point at the combined molecule
                for waldo in self.waldos:
                    if waldo.molecule is molecules[0]:
                        waldo.molecule = molecules[1]

                del self.molecules[molecules[0]]
                del self.molecules[molecules[1]]
                self.molecules[molecules[1]] = None  # dummy value

    def bond_minus(self):
        for position, neighbor_posn, direction in self.bonder_pairs:
            # Continue if there isn't a molecule with a bond over this pair
            molecule = self.get_molecule(position)
            if molecule is None or direction not in molecule[position].bonds:
                continue

            # Now that we know for sure the molecule will be mutated, debond the molecule
            # and check if this broke the molecule in two
            split_off_molecule = molecule.debond(position, direction)

            # Mark the molecule as modified
            del self.molecules[molecule]
            self.molecules[molecule] = None  # Dummy value

            # If a new molecule broke off, add it to the reactor list
            if split_off_molecule is not None:
                self.molecules[split_off_molecule] = None  # Dummy value

                # If the molecule got split apart, ensure any waldos holding it are now holding
                # the correct split piece of it
                for waldo in self.waldos:
                    if waldo.molecule is molecule and waldo.position in split_off_molecule:
                        waldo.molecule = split_off_molecule

    def hash_state_and_check(self):
        '''Hash the current reactor state and check if it matches a past state, fast-forwarding cycles when possible.'''
        # If we previously did a lookahead to the next unexplored branch, we can remember how many cycles
        # we are from an unknown state and skip calculating the hash while we're advancing the reactor to that state.
        # This also guarantees that every time the below code notices we just did a random input, we are adding a new
        # segment and not following an already-explored segment
        if self.cycles_to_new_state > 0:
            self.cycles_to_new_state -= 1
            return

        # If the current state included a call to a random input, add a new branch node if the segment we were in
        # doesn't already link to one (i.e. we didn't just fast-forward), and add a segment for the new branch
        if any(did_input and len(self.level.input_molecules[i]) >= 2
               for i, did_input in enumerate(self.did_input_this_cycle)):
            last_segment = self.state_tree_segments[self.cur_state_tree_segment_idx]

            if last_segment.next_node is None:  # next_node should only be present if we previously fast-forwarded
                self.state_tree_nodes.append({})
                last_segment.next_node = len(self.state_tree_nodes) - 1

            cur_node = self.state_tree_nodes[last_segment.next_node]

            # Add a new segment and link to it from the node
            self.state_tree_segments.append(StateTreeSegment())
            self.cur_state_tree_segment_idx = len(self.state_tree_segments) - 1

            input_branch_key = tuple(self.last_random_molecule_indices[i] if did_input else None
                                     for i, did_input in enumerate(self.did_input_this_cycle))
            assert input_branch_key not in cur_node  # Should be guaranteed by our cycles_to_new_state fast-forwarding
            cur_node[input_branch_key] = self.cur_state_tree_segment_idx

        # Update the current segment's cycle and output counts
        cur_segment = self.state_tree_segments[self.cur_state_tree_segment_idx]
        for output_idx, did_output in enumerate(self.did_output_this_cycle):
            if did_output:
                cur_segment.output_cycles[output_idx].append(cur_segment.cycles)  # Cycle relative to segment start
        cur_segment.cycles += 1

        cur_state = hash(self)
        if cur_state not in self.prior_states:
            # Store the current state and which segment it's in
            self.prior_states[cur_state] = self.cur_state_tree_segment_idx
            return

        other_segment_idx = self.prior_states[cur_state]

        # Count how many cycles into the matched segment we skipped. We need to know this whether or not it's the same
        # as our current segment
        skipped_cycles = 1  # We always skip at least the cycle leading to the hash we merged with
        for state, segment_idx in self.prior_states.items():
            if segment_idx != other_segment_idx:
                continue
            elif state == cur_state:
                break

            skipped_cycles += 1

        # If the matched hash is in the current segment of the hash tree, we found a deterministic loop and can
        # safely fast-forward or declare an infinite loop
        if other_segment_idx == self.cur_state_tree_segment_idx:
            # Figure out how many cycles it will take each output to complete. The max is our winning cycle
            remaining_cycles = 0
            for i in self.level.output_counts.keys():
                # Ignore the output if it's already complete. At least one must not be complete or we'd have
                # exited while executing this cycle
                remaining_outputs = self.level.output_counts[i] - self.completed_output_counts[i]
                if remaining_outputs <= 0:
                    continue

                # Identify the outputs that are contained in the part of the segment we're looping over
                loop_outputs = [c for c in cur_segment.output_cycles[i] if c >= skipped_cycles]
                if not loop_outputs:
                    # If this loop doesn't output any molecules into an incomplete output, we will never win
                    raise InfiniteLoopError()

                # Note that the last loop can't be a full loop, so we add a -1 to the remaining outputs
                full_loops, outputs_remainder = divmod(remaining_outputs - 1, len(loop_outputs))
                outputs_remainder += 1  # Per the -1 above, we've ensure this is at least 1
                cycles_remainder = 0
                if outputs_remainder != 0:
                    # +1 since it's 0-indexed
                    cycles_remainder = loop_outputs[outputs_remainder - 1] - skipped_cycles + 1
                remaining_cycles = max(remaining_cycles,
                                       full_loops * (cur_segment.cycles - skipped_cycles) + cycles_remainder)
            self.cycle += remaining_cycles

            raise RunSuccess()

        # Update the current segment based on the other segment's remaining cycles
        # We will also initialize some incoming_<measure> vars that will be used in the lookahead after this
        other_segment = self.state_tree_segments[other_segment_idx]

        incoming_cycles = other_segment.cycles - skipped_cycles
        incoming_outputs = [0, 0]
        for i in range(2):
            other_outputs = [cur_segment.cycles + c - skipped_cycles
                             for c in other_segment.output_cycles[i]
                             if c >= skipped_cycles]
            incoming_outputs[i] += len(other_outputs)
            cur_segment.output_cycles[i].extend(other_outputs)
        cur_segment.cycles += incoming_cycles
        cur_segment.next_node = other_segment.next_node
        incoming_cycles += 1  # I think the above had an off-by-1 error?

        # At this point we must have finished exploring a branch and have at least one loopback; look for another
        # unexplored branch. We can't actually change the current reactor state to the hash before a new branch,
        # but we can fast-forward the cycle/output counts past any loops we will hit, and skip hash checks on the
        # way to that new branch
        # Starting from the matched state, search forward in the state tree, drawing random inputs as needed,
        # and tallying up the future cycle and output counts for each jump.
        # If we find...
        # * that our accumulation of output counts is about to make us win: stop and step through each state
        #   in the offending segment to figure out exactly what cycle we won on.
        # * An unexplored branch: stop searching and exit
        # * A loop to a node we've visited: Add the loop's cycles/outputs to the reactor's real counts,
        #   then reset the tally of future cycles/outputs to those from before the loop, and continue searching.
        segment_idx = self.cur_state_tree_segment_idx
        segment = cur_segment
        visit_path = {}
        while True:
            # Check if the level will be complete
            if all(self.completed_output_counts[output_idx] + incoming_outputs[output_idx] >= required_count
                   for output_idx, required_count in self.level.output_counts.items()):
                if self.debug:
                    print(f'Fast-forwarding to end from cycle {self.cycle}')

                # Figure out exactly which cycle we won on from the segment we just fast-forwarded through
                # We could check before fast-forwarding but this is more convenient since our first
                # lookahead above may have been only partway through the segment we matched
                for i in range(len(self.completed_output_counts)):
                    self.completed_output_counts[i] += incoming_outputs[i]

                winning_cycle = 0
                for i in self.level.output_counts.keys():
                    output_diff = self.completed_output_counts[i] - self.level.output_counts[i]
                    assert output_diff >= 0  # Sanity check that we actually won
                    if output_diff < len(segment.output_cycles[i]):
                        winning_cycle = max(winning_cycle, segment.output_cycles[i][-output_diff - 1])
                self.cycle += incoming_cycles - (segment.cycles - winning_cycle)

                raise RunSuccess()

            node_idx = segment.next_node
            node = self.state_tree_nodes[node_idx]
            # If this node is already in our visit path, we found a loop; pre-emptively increase our cycle/outputs
            # and remove the loop from our visit history (i.e. its random inputs will not be re-added to a queue
            # to be used up first)
            if node_idx in visit_path:
                # Add the loop's cycles/outputs pre-emptively; this is safe since we've already checked that we
                # won't win before or during the discovered loop
                if self.debug:
                    print(f'Fast-forwarding a loop of {incoming_cycles - visit_path[node_idx][1]} cycles')

                self.cycle += incoming_cycles - visit_path[node_idx][1]  # Add loop cycles to our total
                for i in range(2):
                    self.completed_output_counts[i] += incoming_outputs[i] - visit_path[node_idx][2][i]

                # Reset accumulated cycles/outputs
                incoming_cycles = visit_path[node_idx][1]
                incoming_outputs = list(visit_path[node_idx][2])
                # Reset visit path to before the loop (note that the random inputs we drew get implicitly removed)
                new_visit_path = {}
                for k, v in visit_path.items():
                    new_visit_path[k] = v
                    if k == node_idx:
                        break
                visit_path = new_visit_path

            # Examine one branch of this node (there must be at least 1) to know which random input(s) we need,
            # then draw new inputs of these
            sample_input_key = next(iter(node))
            new_input_key = tuple(self.level.get_input_molecule_idx(i) if sample_mol_idx is not None else None
                                  for i, sample_mol_idx in enumerate(sample_input_key))
            # In addition to the random input indices predicted, store the cycles and output counts accumulated
            # after each node so we can quickly measure loops and reset our counts after finding one
            visit_path[node_idx] = [new_input_key, incoming_cycles, tuple(incoming_outputs)]

            # Check if this branch is already explored or not
            if new_input_key in node:
                # Fast-forward through the explored segment we found, updating our accumulation of cycles/outputs
                segment_idx = node[new_input_key]
                segment = self.state_tree_segments[segment_idx]
                incoming_cycles += segment.cycles
                for output_idx, output_cycles in enumerate(segment.output_cycles):
                    incoming_outputs[output_idx] += len(output_cycles)
            else:
                # Add the random inputs we forecasted to queues to be drawn from before the PRNG
                for random_input_indices, _, _ in visit_path.values():
                    for input_idx, input_mol_idx in enumerate(random_input_indices):
                        if input_mol_idx is not None:
                            self.incoming_inputs[input_idx].append(input_mol_idx)

                # Indicate how many cycles we should skip state checking for, to avoid accidentally re-triggering
                # a search (as all the states we'll pass through on the way to the new branch are known)
                # The last forecast cycle is the one that will put us in a new branch and needs to be checked, so
                # we need a -1 here.
                # TODO: I think this is correct but the random input cycle - technically the start of a
                #       segment - may have been 'two-wrongs-make-a-right'ed
                self.cycles_to_new_state = incoming_cycles - 1
                # Also update the 'current segment' to what it will be when we start state-checking again
                self.cur_state_tree_segment_idx = segment_idx

                break

    def debug_print(self, duration=0.5):
        '''Print the current reactor state then clear it from the terminal.
        Args:
            duration: Seconds before clearing the printout from the screen. Default 0.5.
        '''
        # Print the current state
        output = str(self)
        print(output)  # Could use end='' but that makes keyboard interrupt output ugly

        time.sleep(duration)

        # Use the ANSI escape code for moving to the start of the previous line to reset the terminal cursor
        cursor_reset = (output.count('\n') + 1) * "\033[F"  # +1 for the implicit newline print() appends
        print(cursor_reset, end='')

        # In order to play nice with any other print statements that may occur between debug prints, instead of just
        # moving the terminal cursor back, overwrite the existing output with whitespace then move the cursor back again
        # Note: This probably cries if str(self) contains characters like '\r', but uh it doesn't
        # TODO: This is pretty ugly, may not be worth the trouble of not crapping on debug print statements
        #print('\n'.join(len(s) * ' ' for s in output.split('\n')))
        #print(cursor_reset, end='')  # Move terminal cursor back again

    def do_cycle(self):
        '''Raises RunSuccess (before the end of the cycle) if the solution completed this cycle.'''
        self.cycle += 1

        for waldo in self.waldos:
            self.exec_instrs(waldo)

        if self.debug and self.cycle >= self.debug:
            self.debug_print(duration=0.0625)

        # Move waldos/molecules
        self.move_contents()

        if self.debug and self.cycle >= self.debug:
            self.debug_print(duration=0.0625)

        self.hash_state_and_check()

        self.end_cycle()

    def end_cycle(self):
        for i in range(2):
            self.did_input_this_cycle[i] = False
            self.did_output_this_cycle[i] = False

    def run(self):
        try:
            while True:
                self.do_cycle()
        except RunSuccess:
            return self.cycle - 1, 1, self.solution.symbols
        finally:
            # Persist the last debug printout
            if self.debug:
                print(str(self))


def score_solution(solution, debug=None):
    return Reactor(solution, debug=debug).run()


def validate_solution(level_code, soln_code, debug=None, verbose=False):
    level = ResearchLevel(level_code)
    solution = Solution(level, soln_code)
    score = Reactor(solution, debug=debug).run()
    assert score == solution.expected_score, (f"Expected score {'-'.join(str(x) for x in solution.expected_score)}"
                                              f" but got {'-'.join(str(x) for x in score)}")
    if verbose:
        print(f"Validated {level.get_name()} {'-'.join(str(x) for x in score)} by {solution.author}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', nargs='?', const=1, type=int,
                        help="Print an updating view of the reactor while it runs. If a value is provided, it is taken"
                             + " as the cycle to start debugging on.")

    args = parser.parse_args()

    level_code = tuple(test_data.valid.keys())[7]
    solution_code = tuple(test_data.valid[level_code])[0]
    validate_solution(level_code, solution_code, debug=args.debug, verbose=True)


if __name__ == '__main__':
    main()
