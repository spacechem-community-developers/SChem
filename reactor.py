#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time

from spacechem.elements_data import elements_dict
from spacechem.exceptions import *
from spacechem.grid import Position, Direction
from spacechem.molecule import ATOM_RADIUS
from spacechem.components import Pipe, Component
from spacechem.waldo import Waldo, Instruction, InstructionType

NUM_WALDOS = 2
NUM_MOVE_CHECKS = 10  # Number of times to check for collisions during molecule movement

# Production level codes don't specify available reactor properties like research levels; encode them here
REACTOR_TYPES = {
    'drag-starter-reactor': {
        'has-large-output': False,
        'bonder-count': 4,
        'has-sensor': False,
        'has-fuser': False,
        'has-splitter': False,
        'has-teleporter': False},
    'drag-advanced-reactor': {
        'has-large-output': False,
        'bonder-count': 4,
        'has-sensor': True,
        'has-fuser': False,
        'has-splitter': False,
        'has-teleporter': False}}


class Reactor(Component):
    # For convenience during float-precision rotation co-ordinates, we consider the center of the
    # top-left cell to be at (0,0), and hence the top-left reactor corner is (-0.5, -0.5).
    # Further, treat the walls as being one atom radius closer, so that we can efficiently check if an atom will collide
    # with them given only the atom's center co-ordinates
    walls = {Direction.UP: -0.5 + ATOM_RADIUS, Direction.DOWN: 7.5 - ATOM_RADIUS,
             Direction.LEFT: -0.5 + ATOM_RADIUS, Direction.RIGHT: 9.5 - ATOM_RADIUS}

    __slots__ = ('in_pipes', 'out_pipes',
                 'waldos', 'molecules',
                 'large_output', 'bonders', 'bonder_pairs', 'sensors', 'fusers', 'splitters', 'swappers',
                 'debug')

    def __init__(self, out_pipes, waldos, large_output=False, bonders=None, sensors=None,
                 component_type='', posn=(2, 0),
                 debug=None):
        super().__init__(component_type, posn)
        self.in_pipes = [None, None]
        self.out_pipes = out_pipes
        self.waldos = waldos
        self.large_output = large_output
        self.bonders = bonders if bonders is not None else []
        self.sensors = sensors if sensors is not None else []
        self.debug = debug  # For convenience

        # Store molecules as dict keys to be ordered (preserving Spacechem's hidden
        # 'least recently modified' rule) and to have O(1) add/delete.
        # Values are ignored.
        self.molecules = {}

        # For convenience/performance, pre-compute a list of (bonder_A, bonder_B, direction) triplets, sorted in the
        # order that bonds or debonds should occur in
        self.bonder_pairs = tuple((posn, neighbor_posn, direction)
                                  for posn in self.bonders
                                  for neighbor_posn, direction in
                                  sorted([(posn + direction, direction)
                                          for direction in (Direction.RIGHT, Direction.DOWN)
                                          if posn + direction in self.bonders],
                                         key=lambda x: self.bonders[x[0]]))

    @classmethod
    def from_export_str(cls, export_str, features_dict=None):
        export_str = export_str.strip()  # Sanitize

        output_pipes = [None, None]
        # Level Features
        # TODO: Now that we pre-compute bond firing orders in Reactor.__init__, just using a list here might be fine
        #       performance-wise and saves on size/index-handling code
        bonders = {}  # posn:idx dict for quick lookups
        sensors = []  # posns
        # Store waldo Starts as (posn, dirn) pairs for quick access when initializing reactor waldos
        waldo_starts = NUM_WALDOS * [None]
        # One map for each waldo, of positions to pairs of arrows (directions) and/or non-arrow instructions
        # TODO: usage might be cleaner if separate arrow_maps and instr_maps... but probably more space
        waldo_instr_maps = [{} for _ in range(NUM_WALDOS)]  # Can't use * or else dict gets multi-referenced

        feature_posns = set()  # for verifying features were not placed illegally
        bonder_count = 0  # Track bonder priorities

        # First line must be the COMPONENT line
        component_line, export_str = export_str.split('\n', maxsplit=1)
        assert component_line.startswith('COMPONENT:'), "Reactor.from_export_str expects COMPONENT line included"
        fields = component_line.split(',')
        assert len(fields) == 4, f"Unrecognized component line format:\n{component_line}"

        component_type = fields[0][len('COMPONENT:'):].strip("'")
        if features_dict is None:
            features_dict = REACTOR_TYPES[component_type]
        component_posn = Position(int(fields[1]), int(fields[2]))

        # TODO: Still don't know what the 4th field does...

        # All members should appear before all pipes
        members_str, pipes_str = export_str.strip().split('PIPE:', maxsplit=1)
        pipes_str = 'PIPE:' + pipes_str  # Awkward

        # TODO: Check where Annotations can legally appear. Probably have to be after the pipes?
        annotations_str = ''  # Oof awkward boilerplate
        if 'ANNOTATION:' in pipes_str:
            pipes_str, annotations_str = pipes_str.split('ANNOTATION:', maxsplit=1)
            pipes_str = pipes_str.strip()  # Oooooof
            annotations_str = 'ANNOTATION:' + annotations_str

        # Parse members
        for line in members_str.strip().split('\n'):
            assert line.startswith('MEMBER:'), f"Unexpected line in reactor component string:\n{line}"
            fields = line.split(',')

            if len(fields) != 8:
                raise Exception(f"Unrecognized member line format:\n{line}")

            member_name = fields[0][len('MEMBER:'):].strip("'")

            # Game stores directions in degrees, with right = 0, up = -90 (reversed so sin math works on
            # the reversed vertical axis)
            direction = None if int(fields[1]) == -1 else Direction(1 + int(fields[1]) // 90)

            # Red has a field which is 64 for arrows, 128 for instructions
            # The same field in Blue is 16 for arrows, 32 for instructions
            waldo_idx = 0 if int(fields[3]) >= 64 else 1

            position = Position(int(fields[4]), int(fields[5]))

            if member_name.startswith('feature-'):
                if position in feature_posns:
                    raise Exception(f"Solution contains overlapping features at {position}")
                feature_posns.add(position)

                if member_name == 'feature-bonder':
                    bonder_count += 1
                    bonders[position] = bonder_count
                elif member_name == 'feature-sensor':
                    sensors.append(position)

                continue

            # Since this member is an instr and not a feature, prep a slot in the instr map
            if position not in waldo_instr_maps[waldo_idx]:
                waldo_instr_maps[waldo_idx][position] = [None, None]

            if member_name == 'instr-start':
                waldo_starts[waldo_idx] = (position, direction)
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.START, direction=direction)
                continue

            # Note: Some similar instructions have the same name but are sub-typed by the
            #       second integer field
            instr_sub_type = int(fields[2])
            if member_name == 'instr-arrow':
                waldo_instr_maps[waldo_idx][position][0] = direction
            elif member_name == 'instr-input':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.INPUT, target_idx=instr_sub_type)
            elif member_name == 'instr-output':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.OUTPUT,
                                                                       target_idx=instr_sub_type)
            elif member_name == 'instr-grab':
                if instr_sub_type == 0:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.GRAB_DROP)
                elif instr_sub_type == 1:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.GRAB)
                else:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.DROP)
            elif member_name == 'instr-rotate':
                if instr_sub_type == 0:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.ROTATE,
                                                                           direction=Direction.CLOCKWISE)
                else:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.ROTATE,
                                                                           direction=Direction.COUNTER_CLOCKWISE)
            elif member_name == 'instr-sync':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SYNC)
            elif member_name == 'instr-bond':
                if instr_sub_type == 0:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.BOND_PLUS)
                else:
                    waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.BOND_MINUS)
            elif member_name == 'instr-sensor':
                # The last CSV field is used by the sensor for the target atomic number
                atomic_num = int(fields[7])
                if atomic_num not in elements_dict:
                    raise Exception(f"Invalid atomic number {atomic_num} on sensor command.")
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.SENSE,
                                                                       direction=direction,
                                                                       target_idx=atomic_num)
            elif member_name == 'instr-toggle':
                waldo_instr_maps[waldo_idx][position][1] = Instruction(InstructionType.FLIP_FLOP, direction=direction)
            else:
                raise Exception(f"Unrecognized member name {member_name}")

        waldos = [Waldo(idx=i,
                        position=waldo_starts[i][0],
                        direction=waldo_starts[i][1],
                        instr_map=waldo_instr_maps[i])
                  for i in range(len(waldo_starts))]

        # Parse pipes
        # Pipe lines may be interleaved; sort them apart and construct the pipes
        pipe_lines = pipes_str.split('\n')
        assert all(s.startswith('PIPE:0,') or s.startswith('PIPE:1,') for s in pipe_lines), \
               f"Unexpected line among component pipes: {pipe_lines}"
        num_pipes = 1 if component_type == 'drag-assembly-reactor' else 2
        # Note that Pipe.from_export_str implicitly checks that the constructed pipe is not empty
        output_pipes = [Pipe.from_export_str('\n'.join(line for line in pipes_str.split('\n')
                                                       if line.startswith(f'PIPE:{i},')))
                        for i in range(num_pipes)]
        # TODO: Verify that each pipe starts from the right relative posn

        # TODO: Parse annotations

        # Verify all features are legal and placed
        assert len(bonders) == features_dict['bonder-count']

        # TODO: ('fuser', self.fusers, 1), ('splitter', self.splitters, 1), ('teleporter', self.swappers, 2)
        for feature, container, default_count in (('sensor', sensors, 1),):
            assert not container or features_dict[f'has-{feature}'], f"Illegal reactor feature {feature}"

            # Regular vs Community Edition sanity checks
            if f'{feature}-count' in features_dict:
                assert len(container) == features_dict[f'{feature}-count']
            elif features_dict[f'has-{feature}']:
                assert len(container) == default_count

        return Reactor(component_type=component_type, posn=component_posn,
                       out_pipes=output_pipes, waldos=waldos,
                       large_output=features_dict['has-large-output'], bonders=bonders, sensors=sensors)

    def export_str(self):
        '''Represent this reactor in solution export string format.'''
        export_str = f"COMPONENT:'{self.type}',{self.posn[0]},{self.posn[1]},''"

        # Features
        for posn in self.bonders:
            export_str += f"\nMEMBER:'feature-bonder',-1,0,1,{posn.col},{posn.row},0,0"
        for posn in self.sensors:
            export_str += f"\nMEMBER:'feature-sensor',-1,0,1,{posn.col},{posn.row},0,0"
        # TODO: fusers, splitters, swappers, the special assembly/disassembly bonders

        # Instructions
        export_str += ''.join('\n' + waldo.export_str() for waldo in self.waldos)

        # Pipes
        export_str += ''.join('\n' + pipe.export_str(pipe_idx=i) for i, pipe in enumerate(self.out_pipes))

        return export_str

    def __hash__(self):
        '''Hash of the current reactor state. Ignores cycle/output counts.'''
        return hash((tuple(molecule.hashable_repr() for molecule in self.molecules),
                     tuple(self.waldos)))

    def __str__(self):
        '''Return a pretty-print string representing this reactor.'''
        num_cols = 10
        num_rows = 8

        # 2 characters per atom + 1 space between atoms/walls (we'll use that space to show waldos)
        grid = [['   ' for _ in range(num_cols)] + [' '] for _ in range(num_rows)]

        # Map out the molecules in the reactor
        for molecule in self.molecules:
            for (c, r), atom in molecule.atom_map.items():
                # Round co-ordinates in case we are mid-rotate
                c, r = round(c), round(r)
                if grid[r][c] != '   ':
                    grid[r][c] = ' XX'  # Colliding atoms
                else:
                    grid[r][c] = f' {atom.element.symbol.rjust(2)}'

        # Represent waldos as |  | when not grabbing and (  ) when grabbing. Distinguishing red/blue isn't too important
        for waldo in self.waldos:
            c, r = waldo.position
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

        return result

    def do_instant_actions(self, cycle):
        for waldo in self.waldos:
            self.exec_instrs(waldo)

    def move_contents(self):
        '''Move all waldos in this reactor and any molecules they are holding.'''
        for pipe in self.out_pipes:
            if pipe is not None:
                pipe.move_contents()

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
            for posn in self.sensors:
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
        # If there is no such pipe or it has no molecule available, stall the waldo
        if (self.in_pipes[input_idx] is None
                or self.in_pipes[input_idx][-1] is None):
            waldo.is_stalled = True
            return

        # Grab the molecule from the appropriate pipe or stall if no such molecule (or no pipe)
        new_molecule = self.in_pipes[input_idx][-1]
        self.in_pipes[input_idx][-1] = None

        # Update the molecule's co-ordinates to those of the correct zone if it came from an opposite output zone
        sample_posn = next(iter(new_molecule.atom_map))
        if input_idx == 0 and sample_posn.row >= 4:
            new_molecule.move_fine(Direction.UP, 4)
        elif input_idx == 1 and sample_posn.row < 4:
            new_molecule.move_fine(Direction.DOWN, 4)

        self.molecules[new_molecule] = None  # Dummy value

        self.check_molecule_collisions(new_molecule)

    def output(self, waldo, output_idx):
        # If the there is no such output pipe (e.g. assembly reactor, large output research), do nothing
        if self.out_pipes[output_idx] is None:
            return

        # TODO: It'd be nice to only have to calculate this for molecules that have been
        #       debonded or dropped, etc. However, the cost of pre-computing it every time
        #       we do such an action is probably not worth the cost of just doing it once
        #       over all molecules whenever output is called.
        # TODO 2: On the other hand, solutions with a waldo wall-stalling on output just
        #         got fucked
        # This manual iter is a little awkward but helps ensure we don't iterate more than once into this dict while
        # we're deleting from it
        molecules_in_zone = iter(molecule for molecule in self.molecules
                                 # Ignore grabbed molecules
                                 if not any(waldo.molecule is molecule for waldo in self.waldos)
                                 and molecule.output_zone_idx(large_output=self.large_output) == output_idx)
        molecule = next(molecules_in_zone, None)

        # Try to output the first molecule in the zone if an output hasn't already been done this cycle
        if molecule is not None:
            if self.out_pipes[output_idx][0] is None:
                # Put the molecule in the pipe
                self.out_pipes[output_idx][0] = molecule

                # Look for any other outputable molecule
                molecule = next(molecules_in_zone, None)  # Look for another outputable molecule

                # Remove the outputted molecule from the reactor (make sure not to use the iterator again now!)
                del self.molecules[self.out_pipes[output_idx][0]]

        # If there is any output(s) remaining in this zone (regardless of whether we outputted), stall this waldo
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

    def debug_print(self, cycle, duration=0.5):
        '''Print the current reactor state then clear it from the terminal.
        Args:
            duration: Seconds before clearing the printout from the screen. Default 0.5.
        '''
        # Print the current state
        output = str(self)
        output += f'\nCycle: {cycle}'
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
