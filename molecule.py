#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

from spacechem.grid import Position, Direction
from spacechem.elements_data import elements_dict


# TODO: I'm seriously reconsidering the value of keeping all bonds on each atom given that it'll
#       cut atom sizes significantly and parsing the game's level data (which only stores right and
#       down) will be less kludgy.
#       Investigate how hard it would be to handle the asymmetry at runtime.
class Atom:
    """Represent an Atom, including its element and attached bonds."""
    __slots__ = 'element', 'bonds'

    def __init__(self, element, bonds=None):
        self.element = element
        self.bonds = bonds if bonds is not None else {}

    def __str__(self):
        return self.element.symbol

    def __repr__(self):
        return f'Atom({self.element}, {self.bonds})'

    def get_json_str(self):
        '''Return a string representing this atom in the level json's format.'''
        return (f'{self.element.atomic_num}'
                f'{0 if Direction.RIGHT not in self.bonds else self.bonds[Direction.RIGHT]}'
                f'{0 if Direction.DOWN not in self.bonds else self.bonds[Direction.DOWN]}')

    def hashable_repr(self, relative_orientation):
        '''Return a hashable object representing this molecule, for use in comparing run states.
        Requires the parent molecule's relative orientation to the main grid to ensure bonds will
        be hashed correctly.
        '''
        return (self.element.atomic_num,
                frozenset((dir + relative_orientation, bond_count)
                          for dir, bond_count in self.bonds.items()))


# Performance requirements:
# Molecule:
# * moving/rotating molecule is fast
# * lookup if pos in molecule is fast
# * lookup if molecule in an output zone is fast
# * (Nice-to-have) Isomorphism algorithm is easy to implement

# Molecule internal struct candidates:
# * direct grid pos-atom dict: O(1) lookup by pos
# * internal positions, with one map from an internal atom's position/direction to a grid
#   position/direction for reference
# * Only need to update one position and direction on move/rotate

# Reactor.molecules:
# * Molecules are ordered by last creation/modification
#   Used By: compliance with spacechem hidden priorities (output order)
# * Molecules container has fast add/delete (dict? If so, ensure __hash__ is left
#   as the default implementation which just checks object identity)
#   Used By: input, output, bond+, bond-
# * molecules can be quickly looked up by a single grid position
#   Used By: grab, bond+, bond-, fuse, etc.

# Molecules container candidates:
# * dict +: ordered, O(1) add/delete, O(N) lookup by pos (assuming O(1) pos lookup on Molecule)
# * list -: ordered, O(1) add, O(N) delete, O(N) lookup by pos
class Molecule:
    '''Class used for representing a molecule in a level's input/output zones and for evaluating
    solutions during runtime.
    All externally-exposed methods should accept external positions. However, to increase
    performance when moving/rotating large molecules, internal positions will be relative.
    '''
    __slots__ = 'name', 'origin', 'origin_offset', 'relative_orientation', 'atom_map'

    def __init__(self, name='', atom_map=None,
                 origin=Position(0, 0), origin_offset=Position(0, 0),
                 relative_orientation=Direction.UP):
        self.name = name
        self.origin = origin
        self.origin_offset = origin_offset
        self.relative_orientation = relative_orientation
        self.atom_map = atom_map if atom_map is not None else {}

    @classmethod
    def from_json_string(cls, json_string):
        parts = json_string.split(';')
        name = parts[0]
        atom_map = {}
        # The second field is the atomic formula which we can ignore (TODO: do something with it)
        for atom_str in parts[2:]:
            # formatted as: {col}{row}{atomic_num}{right_bonds}{down_bonds}
            # Note that atomic_num is from 1-3 characters long so we reference the values after it
            # via negative indices
            position = Position(int(atom_str[1]), int(atom_str[0]))
            atom = Atom(elements_dict[int(atom_str[2:-2])])
            right_bonds = int(atom_str[-2])
            down_bonds = int(atom_str[-1])
            if right_bonds != 0:
                atom.bonds[Direction.RIGHT] = right_bonds
            if down_bonds != 0:
                atom.bonds[Direction.DOWN] = down_bonds

            # Update other existing atoms
            for dir in Direction.RIGHT, Direction.DOWN:
                # Check if any existing atoms above and to our left have right/down bonds for us
                neighbor_posn = position + dir.opposite()
                if neighbor_posn in atom_map and dir in atom_map[neighbor_posn].bonds:
                    atom.bonds[dir.opposite()] = atom_map[neighbor_posn].bonds[dir]

                # Check if any existing atoms need our right/down bonds
                # This doubles information but makes working with atoms less complex/asymmetrical
                neighbor_posn = position + dir
                if neighbor_posn in atom_map and dir in atom.bonds:
                    atom_map[neighbor_posn].bonds[dir.opposite()] = atom.bonds[dir]

            atom_map[position] = atom

        return cls(name=name, atom_map=atom_map)

    def __repr__(self):
        return f'Molecule({self.origin}, {self.origin_offset}, {self.relative_orientation}, {self.atom_map})'

    def __str__(self):
        return f'Molecule({list(self.items())})'

    def get_external_posn(self, internal_posn):
        if self.relative_orientation == Direction.UP:
            return Position(self.origin_offset.row + internal_posn.row,
                            self.origin_offset.col + internal_posn.col)
        elif self.relative_orientation == Direction.RIGHT:
            return Position(self.origin_offset.row + self.origin.row
                            + (internal_posn.col - self.origin.col),
                            self.origin_offset.col + self.origin.col
                            - (internal_posn.row - self.origin.row))
        elif self.relative_orientation == Direction.DOWN:
            return Position(self.origin_offset.row + 2 * self.origin.row - internal_posn.row,
                            self.origin_offset.col + 2 * self.origin.col - internal_posn.col)
        elif self.relative_orientation == Direction.LEFT:
            return Position(self.origin_offset.row + self.origin.row
                            - (internal_posn.col - self.origin.col),
                            self.origin_offset.col + self.origin.col
                            + (internal_posn.row - self.origin.row))

    def get_internal_posn(self, external_posn):
        if self.relative_orientation == Direction.UP:
            return Position(external_posn.row - self.origin_offset.row,
                            external_posn.col - self.origin_offset.col)
        elif self.relative_orientation == Direction.RIGHT:
            return Position(self.origin_offset.col + self.origin.col
                            - (external_posn.col - self.origin.row),
                            external_posn.row + self.origin.col
                            - (self.origin_offset.row + self.origin.row))
        elif self.relative_orientation == Direction.DOWN:
            return Position(self.origin_offset.row + 2 * self.origin.row - external_posn.row,
                            self.origin_offset.col + 2 * self.origin.col - external_posn.col)
        elif self.relative_orientation == Direction.LEFT:
            return Position(external_posn.col + self.origin.row
                            - (self.origin_offset.col + self.origin.col),
                            self.origin_offset.row + self.origin.row
                            - (external_posn.row - self.origin.col))

    def __len__(self):
        '''Return the # of atoms in this molecule.'''
        return len(self.atom_map)

    def __contains__(self, external_posn):
        return self.get_internal_posn(external_posn) in self.atom_map

    def __getitem__(self, external_posn):
        '''Return the atom at the specified position or None.'''
        return self.atom_map[self.get_internal_posn(external_posn)]

    def __setitem__(self, external_posn, atom):
        '''Set the atom at the specified position.'''
        self.atom_map[self.get_internal_posn(external_posn)] = \
                Atom(atom.element, {dir - self.relative_orientation: c for dir, c in atom.bonds.items()})

    def items(self):
        return ((self.get_external_posn(posn),
                 Atom(atom.element, {dir + self.relative_orientation: c for dir, c in atom.bonds.items()}))
                for posn, atom in self.atom_map.items())

    def __iadd__(self, other):
        '''Since we'll never need to preserve the old molecule when bonding, only implement
        mutating add for performance reasons.
        It is expected that each half of the bond between the molecules has already been
        created on them before calling this method.
        '''
        for external_posn, atom in other.items():
            self[external_posn] = atom
        return self

    def get_json_str(self):
        '''Return a string representing this molecule in the level json's format.'''
        # TODO: formula
        result = f'{self.name};{self.formula.get_json_str()}'
        for pos, atom in self.atom_map.items():
            result += ';' + f'{pos.col}{pos.row}' + atom.get_json_str()
        return result

    def move(self, direction):
        self.origin_offset += direction

    def rotate(self, pivot_pos, direction):
        # set the molecule's new 'origin' to the point we're pivoting around, calculate the new
        # origin's offset from its external grid position, then update the internal orientation
        self.origin = self.get_internal_posn(pivot_pos)
        self.origin_offset = Position(pivot_pos.row - self.origin.row,
                                      pivot_pos.col - self.origin.col)
        self.relative_orientation += direction

    def check_collisions(self, other):
        '''Check for collisions with another molecule. Do nothing if checked against itself.'''
        if other is not self:
            for posn in self.atom_map:
                if self.get_external_posn(posn) in other:
                    raise Exception("Collision between molecules.")

    def debond(self, posn, direction):
        '''Decrement the specified bond in this molecule. If doing so disconnects this molecule,
        mutate this molecule to its new size and return the extra molecule that was split off.
        return it (else return None).
        '''
        posn_A = self.get_internal_posn(posn)
        atom_A = self.atom_map[posn_A]
        internal_direction_A = direction - self.relative_orientation

        if internal_direction_A not in atom_A.bonds:
            return

        cur_bond_count = atom_A.bonds[internal_direction_A]

        posn_B = posn_A + internal_direction_A
        atom_B = self.atom_map[posn_B]
        internal_direction_B = internal_direction_A.opposite()

        # Decrement and/or remove the bond on each atom
        atom_A.bonds[internal_direction_A] -= 1
        if atom_A.bonds[internal_direction_A] == 0:
            del atom_A.bonds[internal_direction_A]

        atom_B.bonds[internal_direction_B] -= 1
        if atom_B.bonds[internal_direction_B] == 0:
            del atom_B.bonds[internal_direction_B]

        # Search from atom B, stopping if we find atom A. If not, remove all atoms we discovered
        # connected via B from this molecule and add them to a new molecule
        visited_posns = set()
        visit_queue = [posn_B]
        while visit_queue:
            cur_posn = visit_queue.pop()
            visited_posns.add(cur_posn)

            for dir, count in self.atom_map[cur_posn].bonds.items():
                neighbor_posn = cur_posn + dir
                if neighbor_posn not in visited_posns:
                    visit_queue.append(neighbor_posn)

        if len(visited_posns) == len(self):
            # Molecule was not split, nothing left to do
            return None

        # If the molecule was split, create a new molecule from the positions that were accessible
        # from the disconnected neighbor (and remove those positions from this molecule)
        new_atom_map = {}
        for posn in visited_posns:
            new_atom_map[posn] = self.atom_map[posn]
            del self.atom_map[posn]
        return Molecule(atom_map=new_atom_map, origin=self.origin, origin_offset=self.origin_offset,
                        relative_orientation=self.relative_orientation)


    def output_zone_idx(self):
        if not self:
            return None

        if any(self.get_external_posn(posn).col < 6 for posn in self.atom_map.keys()):
            return None

        # Check if any atom doesn't match the output zone of the first atom
        # Note that this is a generator so we avoid re-checking the first position
        posns = (self.get_external_posn(posn) for posn in self.atom_map.keys())
        is_psi = next(posns).row <= 4
        if any((posn.row <= 4) != is_psi for posn in posns):
            return None

        # If we haven't returned yet, all atoms were in the same output zone
        return 0 if is_psi else 1

    def get_neighbor_bonds(self, internal_posn):
        '''Helper to return tuples of (bond_count, neighbor_posn) for a given internal position.
        An atom must exist at the given position. Used by isomorphism algorithm.
        '''
        return ((bond_count, internal_posn + direction)
                for direction, bond_count in self.atom_map[internal_posn].bonds.items())

    # Note that we must be careful not to implement __eq__ so that we don't interfere with e.g.
    # removing a molecule from the reactor list.
    def isomorphic(self, other):
        '''Check if this molecule is topologically equivalent to the given molecule.'''

        # Fail early if inequal length
        if len(self) != len(other):
            return False

        def get_atom_struct_dicts(molecule):
            '''Helper to return dicts mapping an atom structure (element + bonds, unordered)
            to the frequency of atoms of that structure in the molecule, or mapping a position
            in the given molecule to its atom structure.
            '''
            atom_struct_to_freq = {}
            posn_to_atom_struct = {}
            for posn, atom in molecule.atom_map.items():
                atom_struct = (atom.element.atomic_num,
                               frozenset(Counter(atom.bonds.values()).items()))

                if atom_struct not in atom_struct_to_freq:
                    atom_struct_to_freq[atom_struct] = 0
                atom_struct_to_freq[atom_struct] += 1

                posn_to_atom_struct[posn] = atom_struct

            return (atom_struct_to_freq, posn_to_atom_struct)

        # Identify the positions of the rarest unique element/bonds combination (agnostic of
        # bond order).
        this_atom_struct_to_freq, this_posn_to_atom_struct = get_atom_struct_dicts(self)
        other_atom_struct_to_freq, other_posn_to_atom_struct = get_atom_struct_dicts(other)

        # Also take this opportunity to fail early if the two molecules don't exactly match in
        # distributions of atom structures
        if this_atom_struct_to_freq != other_atom_struct_to_freq:
            return False

        # Now that we know that on the surface the two molecules match, check their topology in-depth.
        # Take the rarest combination of element and bond counts, and try to map the graphs onto each
        # other starting from any such atom in this molecule, and comparing against all matching atoms
        # in the other molecule. We'll trickle the 'tree' down from each possible starting atom until we find a tree that matches.
        rarest_atom_struct = min(this_atom_struct_to_freq.items(),
                                 key=lambda x: x[1])[0]
        our_root_posn = next(posn for posn, atom_struct in this_posn_to_atom_struct.items()
                             if atom_struct == rarest_atom_struct)

        for their_root_posn in (posn for posn, atom_struct in other_posn_to_atom_struct.items()
                                if atom_struct == rarest_atom_struct):
            # Track which positions from each molecule we've visited during this algorithm
            our_visited_posns = set()
            their_visited_posns = set()

            def molecules_match_recursive(our_posn, their_posn):
                '''Attempt to recursively map any unvisited atoms of the molecules onto each other
                starting from the given position in each.
                '''
                if this_posn_to_atom_struct[our_posn] != other_posn_to_atom_struct[their_posn]:
                    return False

                our_visited_posns.add(our_posn)
                their_visited_posns.add(their_posn)

                # Attempt to recursively match each of our bonded neighbors in turn (comparing
                # against all possible bonds of their
                for our_bond_count, our_neighbor in self.get_neighbor_bonds(our_posn):
                    if our_neighbor not in our_visited_posns:
                        if not any(molecules_match_recursive(our_neighbor, their_neighbor)
                                   for their_bond_count, their_neighbor in other.get_neighbor_bonds(their_posn)
                                   # Only check bonds of theirs
                                   if (their_neighbor not in their_visited_posns
                                        and our_bond_count == their_bond_count)):
                            our_visited_posns.remove(our_posn)
                            their_visited_posns.remove(their_posn)
                            return False
                return True

            if molecules_match_recursive(our_root_posn, their_root_posn):
                return True
        return False

    # We don't overload __hash__ (which by default checks instance matching) because we want to
    # preserve the ability to store molecules in a dictionary (ordered and provides O(1) add,
    # delete). Otherwise if we move a molecule its hash will change and it will no longer be
    # accessible in the dict.
    def hashable_repr(self):
        '''Return a hashable object representing this molecule, for use in comparing run states.'''
        return frozenset((self.get_external_posn(posn),
                          atom.hashable_repr(self.relative_orientation))
                         for posn, atom in self.atom_map.items())
