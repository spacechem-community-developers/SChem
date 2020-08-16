#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import time

from spacechem.exceptions import *
from spacechem.grid import Direction
from spacechem.level import ResearchLevel
from spacechem.molecule import ATOM_RADIUS
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
        return hash((self.position, self.direction, self.molecule is None, self.is_rotating,
                     tuple(self.flipflop_states.values())))

    def __repr__(self):
        return f'Waldo({self.idx}, pos={self.position}, dir={self.direction}, is_stalled={self.is_stalled})'

    def cur_cmd(self):
        '''Return a waldo's current 'command', i.e. non-arrow instruction, or None.'''
        return None if self.position not in self.instr_map else self.instr_map[self.position][1]


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

    __slots__ = ('level', 'solution', 'cycle', 'prior_states', 'waldos', 'molecules', 'did_input_this_cycle',
                 'did_output_this_cycle', 'completed_output_counts', 'bonder_pairs', 'debug')

    def __init__(self, solution, debug=None):
        self.debug = debug  # For convenience

        self.level = solution.level
        self.solution = solution
        self.cycle = 0

        # Optionally track the hashes of previous states of this run, paired with their cycle and
        # completed output counts
        self.prior_states = {}

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

        new_molecule = self.level.get_input_molecule(input_idx)
        self.molecules[new_molecule] = None  # Dummy value
        self.check_molecule_collisions(new_molecule)

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
        # Hash the current reactor state and check if it matches a past reactor state
        # TODO: Maybe only hash on cycles when either output count incremented?
        #       But then we can't detect infinite loops, which is pretty important to have for
        #       various applications (self-play, auto-solution-verifier). And hashing every Nth cycle will just make the
        #       solver slower if N happens to be coprime with the solution's loop size. Hashing 2/3 cycles + outputs
        #       might work but probably isn't worth the headache
        # TODO: To handle random input levels, in addition to cycle/output counts, store whether any
        #       inputs were called this cycle, and whenever we encounter a matching hash, only skip
        #       ahead to one cycle before another input will be called. Eventually, we'll have all possible states
        #       hashed and be skipping all cycles that don't call an input.
        # TODO: If we implement that, how do we know when we're in an infinite loop?
        #       Answer: TreeDict of past hashes and wait for the tree to be self-looping over all
        #               possible inputs (where branches split whenever a cycle does an input)
        cur_state_hash = hash(self)
        if cur_state_hash not in self.prior_states:
            # If this is a new state, store it
            # TODO: Implement memory limit
            self.prior_states[cur_state_hash] = (self.cycle,
                                                 # Store tuples rather than dict copies to reduce memory bloat
                                                 tuple(self.completed_output_counts[
                                                           i] if i in self.completed_output_counts
                                                       else None
                                                       for i in range(2)))
        else:
            # We're in a loop!
            # TODO: Random levels will throw a wrench in this. We can basically only progress
            # to the next input command state in them.
            prior_cycle, prior_completed_output_counts = self.prior_states[cur_state_hash]

            if any(self.completed_output_counts[i] <= prior_completed_output_counts[i]  # == if I weren't paranoid
                   for i in self.level.output_counts.keys()):
                # We're in the same state as before but at least one required output hasn't been
                # increased, therefore (TODO: assuming not random level) this solution is stuck
                # in an infinite loop that will never win
                raise InfiniteLoopError()

            if self.debug:
                print(f"Found a loop between cycles {prior_cycle} and {self.cycle}, fast-forwarding")

            # Figure out how many times we could repeat this state loop to get as close to
            # winning as possible without looping past the winning cycle
            future_loops = 0
            for i in self.level.output_counts.keys():
                loop_output_diff = self.completed_output_counts[i] - prior_completed_output_counts[i]
                remaining_outputs = self.level.output_counts[i] - self.completed_output_counts[i]
                # Calculate with remaining_outputs - 1 to ensure we don't loop past the
                # winning state
                future_loops = max((future_loops, (remaining_outputs - 1) // loop_output_diff))

            # Calculate the future cycle if we progress to the start of the last loop
            loop_size = self.cycle - prior_cycle
            loop_completed_output_counts = {i: self.completed_output_counts[i] - prior_completed_output_counts[i]
                                            for i in self.level.output_counts.keys()}

            future_cycle = self.cycle + future_loops * loop_size
            future_completed_output_counts = {
                i: self.completed_output_counts[i] + future_loops * loop_completed_output_counts[i]
                for i in self.level.output_counts.keys()}

            # We now know that one more loop will win us the game: walk through the prior states
            # until we identify the winning cycle
            in_loop = False
            for state_hash, (cycle, completed_output_counts) in self.prior_states.items():
                # Skip ahead to the loop we found...
                in_loop |= state_hash == cur_state_hash
                if in_loop and all((future_completed_output_counts[i]
                                    + (completed_output_counts[i] - prior_completed_output_counts[i]))
                                   >= self.level.output_counts[i]
                                   for i in self.level.output_counts.keys()):
                    # Skip ahead our run state cycle to the winning cycle then alert the caller
                    self.cycle = future_cycle + (cycle - prior_cycle)
                    raise RunSuccess()
            # If we haven't encountered the winning cycle in the last loop over the prior states,
            # the current run state must be the winning one (not in the priors loop)
            self.cycle = future_cycle + (self.cycle - prior_cycle)
            raise RunSuccess()

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
        # TODO: This makes the output kind of flashy which isn't as nice to look at... not really any way around that
        #       if I don't want it to crap on regular print() statements. Could maybe have a pretty-but-rude option
        print('\n'.join(len(s) * ' ' for s in output.split('\n')))
        print(cursor_reset, end='')  # Move terminal cursor back again

    def do_cycle(self):
        '''Raises RunSuccess (before the end of the cycle) if the solution completed this cycle.'''
        for waldo in self.waldos:
            self.exec_instrs(waldo)

        if self.debug and self.cycle >= self.debug:
            self.debug_print(duration=0.25)

        # Move waldos/molecules
        self.move_contents()

        if self.debug and self.cycle >= self.debug:
            self.debug_print(duration=0.25)

        self.hash_state_and_check()

        self.end_cycle()
        self.cycle += 1

    def end_cycle(self):
        for i in range(2):
            self.did_input_this_cycle[i] = False
            self.did_output_this_cycle[i] = False

    def run(self):
        try:
            while True:
                self.do_cycle()
        except RunSuccess:
            return self.cycle, 1, self.solution.symbols
        finally:
            # Persist the last debug printout
            if self.debug:
                print(str(self))


def score_solution(soln, debug=None):
    return Reactor(soln, debug=debug).run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', nargs='?', const=1, type=int,
                        help="Print an updating view of the reactor while it runs. If a value is provided, it is taken"
                             + " as the cycle to start debugging on.")

    args = parser.parse_args()

    level_code = tuple(test_data.valid_levels_and_solutions.keys())[4]
    solution_code = tuple(test_data.valid_levels_and_solutions[level_code])[2]
    print(score_solution(Solution(ResearchLevel(level_code), solution_code), debug=args.debug))


if __name__ == '__main__':
    main()
