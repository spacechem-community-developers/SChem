#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cProfile
from timeit import timeit

from spacechem.elements_data import Element, elements, elements_dict
from spacechem.grid import Position, Direction
from spacechem.level import ResearchLevel
from spacechem.molecule import Molecule
from spacechem.solution import Instruction, Solution

class Waldo:
    def __init__(self, idx, position, direction, instr_map):
        self.idx = idx
        self.position = position
        self.direction = direction
        self.is_stalled = False
        self.instr_map = instr_map # Position map of tuples containing arrow and 'command' (non-arrow) instructions
        self.molecule = None

    def __hash__(self):
        '''A waldo's run state is uniquely identified by its position, direction,
        whether it's stalled (to distinguish start vs end cycle of a rotate), and whether it's
        holding a molecule.
        '''
        # TODO: Maybe (self.is_stalled and self.instr_map[self.position] in (rotates))) instead, to
        #       avoid worrying about sync states? (since whether we just reached or were already on
        #       Sync doesn't affect future run state, unlike with Rotate)
        return hash((self.position, self.direction, self.is_stalled, self.molecule is None))

    def __repr__(self):
        return f'Waldo({self.idx}, pos={self.position}, dir={self.direction}, is_stalled={self.is_stalled})'

    def cur_cmd(self):
        '''Return a waldo's current 'command', i.e. non-arrow instruction, or None.'''
        return self.instr_map[self.position][1]

class RunSuccess(Exception):
    pass

class InfiniteLoopException(Exception):
    pass

class Reactor:
    '''Represents the current state of a running solution.
    To track state properly, requires that exec_cmd(waldo) be called with each of its waldos on each
    cycle for which that waldo has a cmd, followed by move(waldo) for each waldo each cycle.
    Might be able to skip ahead to cycles when either waldo has a command... but on the other hand
    if this is coded right it won't make a huge difference.
    '''
    def __init__(self, level, solution):
        self.level = level
        self.solution = solution
        self.cycle = 0

        # Optionally track the hashes of previous states of this run, paired with their cycle and
        # completed output counts
        self.prior_states = {}

        # Track whether each input zone has already done its max of 1 input per cycle
        self.did_input_this_cycle = [False, False]

        # Global output vars
        self.completed_output_counts = [None if output_count is None else 0
                                        for output_count in level.output_counts]
        # Track whether each input zone has already done its max of 1 output per cycle
        self.did_output_this_cycle = [False, False]

        # Store molecules as dict keys to be ordered (preserving Spacechem's hidden
        # 'least recently modified' rule) and to have O(1) add/delete.
        # Values are ignored.
        self.molecules = {}

        # Pre-process the soln into red and blue instr maps that allow for skipping along rows/columns
        # to the next arrow/instruction. Ignores Start commands.
        self.symbols = 0
        self.waldos = [Waldo(i,
                             solution.waldo_starts[i][0],
                             solution.waldo_starts[i][1],
                             solution.waldo_instr_maps[i])
                       for i in range(len(solution.waldo_starts))]
        self.flipflop_states = []

    def __hash__(self):
        '''Hash of the current run state. Ignores cycle/output counts.'''
        return hash((tuple(molecule.hashable_repr() for molecule in self.molecules.keys()),
                     tuple(self.waldos),
                     tuple(self.flipflop_states)))

    def do_cycle(self):
        '''Raises RunSuccess (before the end of the cycle) if the solution completed this cycle.'''
        for waldo in self.waldos:
            self.exec_instrs(waldo)

        # Move the waldos and their molecules (we know our start tiles only have Start commands
        # and don't need to be executed).
        for waldo in self.waldos:
            self.move(waldo)

        self.hash_state_and_check()

        self.end_cycle()
        self.cycle += 1

    def check_collisions(self, molecule):
        '''Check that the given molecule doesn't collide with any existing molecules in the grid
        (that aren't itself).
        Raise an exception if it does.
        '''
        for other_molecule in self.molecules.keys():
            molecule.check_collisions(other_molecule) # Implicitly ignores self

    def exec_instrs(self, waldo):
        if waldo.position not in waldo.instr_map:
            return

        arrow_direction, cmd = waldo.instr_map[waldo.position]

        # Update the waldo's direction based on any arrow in this cell
        if arrow_direction is not None:
            waldo.direction = arrow_direction

        # Execute the non-arrow instruction
        if cmd == Instruction.INPUT_ALPHA:
            self.input(waldo, 0)
        elif cmd == Instruction.INPUT_BETA:
            self.input(waldo, 1)
        elif cmd == Instruction.OUTPUT_PSI:
            self.output(waldo, 0)
        elif cmd == Instruction.OUTPUT_OMEGA:
            self.output(waldo, 1)
        elif cmd == Instruction.GRAB:
            self.grab(waldo)
        elif cmd == Instruction.DROP:
            self.drop(waldo)
        elif cmd == Instruction.GRAB_DROP:
            if waldo.molecule is None:
                self.grab(waldo)
            else:
                self.drop(waldo)
        elif cmd in (Instruction.ROTATE_CWISE, Instruction.ROTATE_CCWISE):
            # If we are holding a molecule and didn't just finish a stall, stall the waldo
            # In all other cases, unstall the waldo
            waldo.is_stalled = waldo.molecule is not None and not waldo.is_stalled
        elif cmd == Instruction.BOND_PLUS:
            self.bond_plus()
        elif cmd == Instruction.BOND_MINUS:
            self.bond_minus()
        elif cmd == Instruction.SYNC:
            # Mark this waldo as stalled if both waldos aren't on a Sync
            other_waldo = self.waldos[1 - waldo.idx]
            waldo.is_stalled = other_waldo.cur_cmd() != Instruction.SYNC
        elif cmd == Instruction.FUSE:
            pass
        elif cmd == Instruction.SPLIT:
            pass
        elif cmd == Instruction.FLIP_FLOP:
            pass
        elif cmd == Instruction.SWAP:
            pass

    def input(self, waldo, input_idx):
        # If an input has already been called this cycle (i.e. blue inputs after red), stall this
        # waldo and do nothing
        if self.did_input_this_cycle[input_idx]:
            waldo.is_stalled = True
            return

        new_molecule = self.level.get_input_molecule(input_idx)
        self.molecules[new_molecule] = None # Dummy value
        self.check_collisions(new_molecule)

    def output(self, waldo, output_idx):
        # If the output is disabled, do nothing
        if self.level.output_counts[output_idx] is None:
            return

        if not self.did_output_this_cycle[output_idx]:
            # Output the first molecule in the given output zone
            for molecule in self.molecules.keys():
                # TODO: It'd be nice to only have to calculate this for molecules that have been
                #       debonded or dropped, etc. However, the cost of pre-computing it every time
                #       we do such an action is probably not worth the cost of just doing it once
                #       over all molecules whenever output is called.
                # TODO 2: On the other hand, solutions with a waldo wall-stalling on output just
                #         got fucked
                if molecule.output_zone_idx() == output_idx:
                    if not molecule.isomorphic(self.level.get_output_molecule(output_idx)):
                        raise Exception("Invalid output molecule.")

                    self.completed_output_counts[output_idx] += 1

                    # Check if we've won
                    if all(self.level.output_counts[i] is None
                           or self.completed_output_counts[i] >= self.level.output_counts[i]
                           for i in range(len(self.level.output_counts))):
                        raise RunSuccess()

                    # Remove the molecule from the reactor
                    del self.molecules[molecule]
                    self.did_output_this_cycle[output_idx] = True
                    break

        # If there are still outputs remaining in this zone, stall this waldo
        if any(molecule.output_zone_idx() == output_idx for molecule in self.molecules.keys()):
            waldo.is_stalled = True

    def grab(self, waldo):
        if waldo.molecule is None:
            waldo.molecule = self.get_molecule(waldo.position)

    def get_molecule(self, position):
        '''Select the molecule at the given grid position, or None if no such molecule.
        Used by Grab, Bond+/-, Fuse, etc.
        '''
        for molecule in self.molecules.keys():
            if position in molecule:
                return molecule

    def drop(self, waldo):
        waldo.molecule = None # Remove the reference to the molecule

    def move(self, waldo):
        # Check that we're not pulling our molecule apart (note that for some reason spacechem
        # disallows rotating a doubly-held molecule even if both waldos are in the same position)
        # TODO: Not sure this code accounts for wall-stalled waldos since they won't have
        #       'is_stalled' marked but shouldn't rip apart if the other waldo is stalled.
        if waldo.molecule is not None:
            other_waldo = self.waldos[1 - waldo.idx]
            if (waldo.molecule is other_waldo.molecule
                and (waldo.is_stalled != other_waldo.is_stalled
                     or waldo.direction != other_waldo.direction
                     or waldo.cur_cmd() in (Instruction.ROTATE_CWISE, Instruction.ROTATE_CCWISE))):
                raise Exception("Molecule pulled apart")

        if waldo.is_stalled:
            # Rotate the molecule if it's stalled on a rotate cmd (hasn't rotated yet)
            if waldo.molecule is not None and waldo.cur_cmd() == Instruction.ROTATE_CWISE:
                waldo.molecule.rotate(waldo.position, Direction.CLOCKWISE)
                self.check_collisions(waldo.molecule)
            elif waldo.molecule is not None and waldo.cur_cmd() == Instruction.ROTATE_CCWISE:
                waldo.molecule.rotate(waldo.position, Direction.COUNTER_CLOCKWISE)
                self.check_collisions(waldo.molecule)
            else:
                # Un-stall the waldo (unless it just rotated, in which case we leave it marked as
                # stalled so we know not to rotate it again next cycle). It is indeed true that
                # waldos stalled against a wall on a rotate command only rotate every second cycle.
                waldo.is_stalled = False # TODO: avoid re-execing syncs for performance?
        elif not ((waldo.direction == Direction.UP and waldo.position.row == 0)
                  or (waldo.direction == Direction.DOWN and waldo.position.row == 8)
                  or (waldo.direction == Direction.LEFT and waldo.position.col == 0)
                  or (waldo.direction == Direction.RIGHT and waldo.position.col == 10)):
            # If the waldo is not wall-stalled, move it and its molecule
            waldo.position += waldo.direction
            if waldo.molecule is not None:
                waldo.molecule.move(waldo.direction)
                self.check_collisions(waldo.molecule)

    def bond_plus(self):
        for position in self.solution.bonders:
            bond_directions = [Direction.RIGHT, Direction.DOWN]
            # Bond to higher priority (lower index) bonders first
            bond_directions.sort(key=lambda d: 0 if (neighbor_posn := position + d) not in self.solution.bonders
                                                 else self.solution.bonders[neighbor_posn])
            for direction in bond_directions:
                neighbor_posn = position + direction
                if neighbor_posn not in self.solution.bonders:
                    continue

                # Identify the molecule on each bonder (may be same, doesn't matter for now)
                molecule_A = molecule_B = None
                for molecule in self.molecules:
                    if position in molecule:
                        molecule_A = molecule
                    if position + direction in molecule:
                        molecule_B = molecule

                if molecule_A is None or molecule_B is None:
                    continue

                # TODO: Make Molecule's external API's hide the internal positioning system
                #       Update this section to be agnostic of whether we use an internal positioning
                #       system so that we can try switching back to raw external coordinates later
                #       and see if there's a performance diff one way or the other

                atom_A = molecule_A[position]

                # If the bond being increased is already at the max bond size of 3, don't do
                # anything. However, due to weirdness of Spacechem's bonding algorithm, we still
                # mark the molecule as modified below
                internal_direction_A = direction + molecule_A.relative_orientation
                if (internal_direction_A not in atom_A.bonds
                    or atom_A.bonds[internal_direction_A]) != 3:
                    atom_B = molecule_B[neighbor_posn]

                    # Do nothing if either atom is at its bond limit (spacechem does not mark
                    # any molecules as modified in this case unless the bond was size 3)
                    if (sum(atom_A.bonds.values()) == atom_A.element.max_bonds
                        or sum(atom_B.bonds.values()) == atom_B.element.max_bonds):
                        continue

                    # TODO: the below expression calls the posn ctor twice but could be once
                    internal_direction_B = (direction + molecule_B.relative_orientation).opposite()

                    if internal_direction_A not in atom_A.bonds:
                        atom_A.bonds[internal_direction_A] = 0
                    atom_A.bonds[internal_direction_A] += 1
                    if internal_direction_B not in atom_B.bonds:
                        atom_B.bonds[internal_direction_B] = 0
                    atom_B.bonds[internal_direction_B] += 1

                if molecule_A is molecule_B:
                    # Mark molecule as modified by popping it to the back of the reactor's queue
                    del self.molecules[molecule_A]
                    self.molecules[molecule_A] = None # dummy value
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
                    self.molecules[molecules[1]] = None # dummy value

    def bond_minus(self):
        for position in self.solution.bonders:
            debond_directions = [Direction.RIGHT, Direction.DOWN]
            # Debond with higher priority (lower index) bonders first
            debond_directions.sort(key=lambda d: 0 if (neighbor_posn := position + d) not in self.solution.bonders
                                                   else self.solution.bonders[neighbor_posn])
            for direction in debond_directions:
                neighbor_posn = position + direction
                if neighbor_posn not in self.solution.bonders:
                    continue

                # Continue if there isn't a molecule with a bond over this pair
                if ((molecule := self.get_molecule(position)) is None
                    or direction not in molecule[position].bonds):
                    continue

                # Now that we know for sure the molecule will be mutated, debond the molecule
                # and check if this broke the molecule in two
                split_off_molecule = molecule.debond(position, direction)

                # Mark the molecule as modified
                del self.molecules[molecule]
                self.molecules[molecule] = None # Dummy value

                # If a new molecule broke off, add it to the reactor list
                if split_off_molecule is not None:
                    self.molecules[split_off_molecule] = None # Dummy value

                    # If the molecule got split apart, ensure any waldos holding it are now holding
                    # the correct split piece of it
                    for waldo in self.waldos:
                        if waldo.molecule is molecule and waldo.position in split_off_molecule:
                            waldo.molecule = split_off_molecule

    def hash_state_and_check(self):
        # Hash the current run state and check if it matches a past run state
        # TODO: Maybe only hash on cycles when either output count incremented?
        #       But then we can't detect infinite loops, which is pretty important to have for
        #       various applications (self-play, auto-solution-verifier).
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
            self.prior_states[cur_state_hash] = (self.cycle, tuple(self.completed_output_counts))
        else:
            # We're in a loop!
            # TODO: Random levels will throw a wrench in this. We can basically only progress
            # to the next input command state in them.
            prior_cycle, prior_completed_output_counts = self.prior_states[cur_state_hash]

            if any(self.level.output_counts[i] is not None
                   # == should be sufficient but I'm paranoid
                   and self.completed_output_counts[i] <= prior_completed_output_counts[i]
                   for i in range(len(self.level.output_counts))):
                # We're in the same state as before but at least one required output hasn't been
                # increased, therefore (TODO: assuming not random level) this solution is stuck
                # in an infinite loop that will never win
                raise InfiniteLoopException()

            # Figure out how many times we could repeat this state loop to get as close to
            # winning as possible without looping past the winning cycle
            future_loops = 0
            for i in range(len(self.level.output_counts)):
                if self.level.output_counts[i] is not None:
                    loop_output_diff = self.completed_output_counts[i] - prior_completed_output_counts[i]
                    remaining_outputs = self.level.output_counts[i] - self.completed_output_counts[i]
                    # Calculate with remaining_outputs - 1 to ensure we don't loop past the
                    # winning state
                    future_loops = max((future_loops, (remaining_outputs - 1) // loop_output_diff))

            # Calculate the future cycle if we progress to the start of the last loop
            loop_size = self.cycle - prior_cycle
            loop_completed_output_counts = [None if self.level.output_counts[i] is None
                                            else self.completed_output_counts[i] - prior_completed_output_counts[i]
                                            for i in range(len(self.level.output_counts))]

            future_cycle = self.cycle + future_loops * loop_size
            future_completed_output_counts = [None if self.level.output_counts[i] is None
                                              else self.completed_output_counts[i] + future_loops * loop_completed_output_counts[i]
                                              for i in range(len(self.level.output_counts))]

            # We now know that one more loop will win us the game: walk through the prior states
            # until we identify the winning cycle
            in_loop = False
            for state_hash, (cycle, completed_output_counts) in self.prior_states.items():
                # Skip ahead to the loop we found...
                in_loop |= state_hash == cur_state_hash
                if in_loop and all(self.level.output_counts[i] is None
                                   or (future_completed_output_counts[i]
                                       + (completed_output_counts[i] - prior_completed_output_counts[i])) >= self.level.output_counts[i]
                                   for i in range(len(self.level.output_counts))):
                    # Skip ahead our run state cycle to the winning cycle then alert the caller
                    self.cycle = future_cycle + (cycle - prior_cycle)
                    raise RunSuccess()
            # If we haven't encountered the winning cycle in the last loop over the prior states,
            # the current run state must be the winning one (not in the priors loop)
            self.cycle = future_cycle + (self.cycle - prior_cycle)
            raise RunSuccess()

    def end_cycle(self):
        for i in range(2):
            self.did_input_this_cycle[i] = False
            self.did_output_this_cycle[i] = False


def score_soln(level, soln):
    # Initialize a hashable object representing the current state of the solution being run
    reactor = Reactor(level, soln)

    try:
        while True:
            reactor.do_cycle()
            print(reactor.molecules)
    except RunSuccess:
        return (reactor.cycle, soln.symbols)


if __name__ == '__main__':
    level_code = '''H4sIAMa7aV4A/3WPQWvDMAyF/0rQaYME7FIYc07rIdDzbh07eImSGFwr2HKhC/nvs5sxtoZdBP
p47+lpBuOmyNUnOQygZhB53Fha32Y4k8U2WgQFr8Ze0NcvQy3l/kkIKKGl6BiU3C3vSwnyf29j
I3njsG6S+XnjTWaKvC2yuV48HB+LNazDVKW5dZGi3t2lipw56lBZ7Qes1nRQvbYBS/gg16Gvvs
X7VRnQBfI/moz6GPAvCZM1zHeQ0eJE/jfm65RbewyofTumZk6fMzk69tTFlg25gqk4pCrGDUmg
I4/5PpxMXjvT9yY9z1dQYvkCUjns8KkBAAA=
'''
    solution_code = '''
SOLUTION:An Introduction to Bonding,Zig,74-1-40,Cycles!!!
COMPONENT:'tutorial-research-reactor-2',2,0,''
MEMBER:'instr-start',-90,0,128,1,7,0,0
MEMBER:'instr-start',0,0,32,0,0,0,0
MEMBER:'feature-bonder',-1,0,1,1,1,0,0
MEMBER:'feature-bonder',-1,0,1,1,2,0,0
MEMBER:'feature-bonder',-1,0,1,1,4,0,0
MEMBER:'feature-bonder',-1,0,1,1,5,0,0
MEMBER:'instr-grab',-1,1,32,1,1,0,0
MEMBER:'instr-arrow',0,0,16,1,5,0,0
MEMBER:'instr-arrow',90,0,64,6,3,0,0
MEMBER:'instr-grab',-1,2,128,6,3,0,0
MEMBER:'instr-grab',-1,1,32,1,4,0,0
MEMBER:'instr-arrow',90,0,16,1,4,0,0
MEMBER:'instr-arrow',-90,0,16,3,5,0,0
MEMBER:'instr-arrow',180,0,16,3,4,0,0
MEMBER:'instr-arrow',-90,0,64,4,4,0,0
MEMBER:'instr-grab',-1,1,128,4,4,0,0
MEMBER:'instr-rotate',-1,0,128,5,3,0,0
MEMBER:'instr-grab',-1,2,32,3,4,0,0
MEMBER:'instr-rotate',-1,1,32,3,5,0,0
MEMBER:'instr-arrow',90,0,16,2,3,0,0
MEMBER:'instr-arrow',180,0,16,2,4,0,0
MEMBER:'instr-bond',-1,1,32,1,5,0,0
MEMBER:'instr-arrow',0,0,64,4,3,0,0
MEMBER:'instr-arrow',180,0,64,6,4,0,0
MEMBER:'instr-arrow',90,0,64,6,2,0,0
MEMBER:'instr-input',-1,1,32,2,3,0,0
MEMBER:'instr-input',-1,0,32,1,0,0,0
MEMBER:'instr-grab',-1,1,128,1,5,0,0
MEMBER:'instr-bond',-1,0,128,1,2,0,0
MEMBER:'instr-input',-1,1,128,5,4,0,0
MEMBER:'instr-arrow',0,0,64,1,2,0,0
MEMBER:'instr-bond',-1,0,128,5,2,0,0
MEMBER:'instr-input',-1,1,128,1,6,0,0
MEMBER:'instr-arrow',90,0,16,1,0,0,0
MEMBER:'instr-bond',-1,0,32,1,2,0,0
MEMBER:'instr-bond',-1,0,32,1,3,0,0
MEMBER:'instr-input',-1,0,128,1,4,0,0
MEMBER:'instr-input',-1,0,128,1,3,0,0
MEMBER:'instr-arrow',180,0,16,1,2,0,0
MEMBER:'instr-arrow',90,0,16,0,2,0,0
MEMBER:'instr-arrow',0,0,16,0,3,0,0
MEMBER:'instr-input',-1,0,128,2,2,0,0
MEMBER:'instr-bond',-1,0,32,2,4,0,0
MEMBER:'instr-input',-1,0,128,4,3,0,0
MEMBER:'instr-input',-1,0,128,4,2,0,0
MEMBER:'instr-output',-1,0,32,2,5,0,0
PIPE:0,4,1
PIPE:1,4,2
'''

    cProfile.run('score_soln(ResearchLevel(level_code), Solution(solution_code))', sort='cumtime')
