#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

from .exceptions import ReactionError
from .grid import *
from .elements import elements_dict

# Diameter of an atom relative to a grid cell, per https://www.reddit.com/r/spacechem/wiki/gamemechanics#wiki_collisions
# Lower bound: 0.7465 (hard bound) based on a rotating 7x3 molecule colliding with an atom (5, 5) away from the waldo
# Upper bound: 0.7538 (or possibly a little higher due to limited collision checks) based on 9x3 missing (9, 0)
ATOM_DIAMETER = 0.75
ATOM_DIAMETER_SQUARED = ATOM_DIAMETER**2
ATOM_RADIUS = ATOM_DIAMETER / 2  # Convenience


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
        return (f'{self.element.atomic_num if self.element.symbol != "Av" else 204}'  # Necessary hack I think
                f'{0 if RIGHT not in self.bonds else self.bonds[RIGHT]}'
                f'{0 if DOWN not in self.bonds else self.bonds[DOWN]}')

    def hashable_repr(self):
        '''Return a hashable object representing this molecule, for use in comparing run states.'''
        return self.element.atomic_num, frozenset(self.bonds.items())

    def rotate(self, rotate_dirn):
        self.bonds = {dirn + rotate_dirn: bond_count for dirn, bond_count in self.bonds.items()}
        return self


# Performance requirements:
# Molecule:
# * moving/rotating molecule is fast
# * lookup if pos in molecule is fast
# * lookup if molecule in an output zone is fast
# * (Nice-to-have) Isomorphism algorithm is easy to implement

# Molecule internal struct candidates:
# * direct grid pos-atom dict: O(1) lookup by pos
# * internal positions, with one map from an internal atom's position/direction to a grid
#   position/direction for reference: O(1) lookup, but a lot of overhead. Seemed to be
#   slower in practice.
#   * Only need to update one position and direction on move/rotate

# Reactor.molecules:
# * Molecules are ordered by last creation/modification
#   Used By: compliance with spacechem hidden priorities (output order)
# * Molecules container has fast add/delete (dict? If so, ensure __hash__ is left
#   as the default implementation which just checks object identity)
#   Used By: input, output, bond+, bond-
# * molecules can be quickly looked up by a single grid position
#   Used By: grab, bond+, bond-, fuse, etc.

# Reactor molecules container candidates:
# * dict + a posn to molecule dict: adds more memory than single dict, but allows O(1) lookup by pos
#   * moving a molecule now has overhead equal to it size... but if it speeds up collision checks it's
#     probably worth it, not to mention commands like bond+, fuse.
# * dict +: ordered, O(1) add/delete, O(N) lookup by pos (assuming O(1) pos lookup on Molecule)
# * list -: ordered, saves memory, O(1) add, O(N) delete, O(N) lookup by pos
class Molecule:
    '''Class used for representing a molecule in a level's input/output zones and for evaluating
    solutions during runtime.
    '''
    __slots__ = 'name', 'atom_map'

    def __init__(self, name='', atom_map=None):
        self.name = name
        self.atom_map = atom_map if atom_map is not None else {}

    @classmethod
    def from_json_string(cls, json_string):
        name, _, *atom_strs = json_string.split(';') # The second field is the atomic formula which we can ignore
        atom_map = {}
        for atom_str in atom_strs:
            assert len(atom_str) >= 5, "Invalid atom string in level json"
            # formatted as: {col}{row}{atomic_num}{right_bonds}{down_bonds}
            # Note that atomic_num is variable length (1-3 chars) so we use negative indices for values after it
            position = Position(col=int(atom_str[0]), row=int(atom_str[1]))
            atom = Atom(elements_dict[int(atom_str[2:-2])])
            right_bonds = int(atom_str[-2])
            down_bonds = int(atom_str[-1])
            if right_bonds != 0:
                atom.bonds[RIGHT] = right_bonds
            if down_bonds != 0:
                atom.bonds[DOWN] = down_bonds

            # Add up/left bonds; this doubles information but makes working with atoms less complex/asymmetrical
            for dir in RIGHT, DOWN:
                # Add up/left bonds to this atom based on the down/right bonds of its neighbors
                neighbor_posn = position + dir.opposite()
                if neighbor_posn in atom_map and dir in atom_map[neighbor_posn].bonds:
                    atom.bonds[dir.opposite()] = atom_map[neighbor_posn].bonds[dir]

                # Add up/left bonds to atoms down/right of this atom
                neighbor_posn = position + dir
                if neighbor_posn in atom_map and dir in atom.bonds:
                    atom_map[neighbor_posn].bonds[dir.opposite()] = atom.bonds[dir]

            atom_map[position] = atom

        return cls(name=name, atom_map=atom_map)

    def __repr__(self):
        return f'Molecule({self.atom_map})'

    def __str__(self):
        """Represent this molecule in a grid."""
        s = ''
        hor_bond_sym = {0: ' ', 1: '-', 2: '=', 3: '≡'}
        ver_bond_sym = {0: '   ', 1: ' | ', 2: '|| ', 3: '|‖ '}

        # Identify the bounds of the grid we have to draw
        min_col = min(posn.col for posn in self.atom_map)
        max_col = max(posn.col for posn in self.atom_map)
        min_row = min(posn.row for posn in self.atom_map)
        max_row = max(posn.row for posn in self.atom_map)

        for row in range(min_row, max_row + 1):
            # Add this row's atoms and horizontal bonds
            for col in range(min_col, max_col + 1):
                posn = Position(col=col, row=row)
                if posn in self:
                    atom = self[posn]
                    s += atom.element.symbol.rjust(2)
                    s += hor_bond_sym[atom.bonds[RIGHT]] if RIGHT in atom.bonds else ' '
                else:
                    s += '   '

            s += '\n'

            # Add lower bonds
            for col in range(min_col, max_col + 1):
                posn = Position(col=col, row=row)
                if posn in self and DOWN in self[posn].bonds:
                    s += ver_bond_sym[self[posn].bonds[DOWN]]
                else:
                    s += '   '
            s += '\n'

        return s.rstrip()

    def __len__(self):
        '''Return the # of atoms in this molecule.'''
        return len(self.atom_map)

    def __contains__(self, posn):
        return posn in self.atom_map

    def __getitem__(self, posn):
        '''Return the atom at the specified position or None.'''
        return self.atom_map[posn]

    def __setitem__(self, posn, atom):
        '''Set the atom at the specified position.'''
        self.atom_map[posn] = atom

    def items(self):
        return self.atom_map.items()

    def __iadd__(self, other):
        '''Since we'll never need to preserve the old molecule when bonding, only implement
        mutating add for performance reasons.
        It is expected that each half of the bond between the molecules has already been
        created on them before calling this method.
        '''
        self.atom_map.update(other.atom_map)
        return self

    def get_json_str(self):
        '''Return a string representing this molecule in the level json's format.'''
        # TODO: formula stuff. Actually the game seems to use slashes in formulas of unknown molecules
        result = f'{self.name};{self.formula.get_json_str()}'
        for pos, atom in self.atom_map.items():
            result += ';' + f'{pos.col}{pos.row}' + atom.get_json_str()
        return result

    def move(self, direction, distance=1):
        self.atom_map = {posn.move(direction, distance=distance): atom for posn, atom in self.atom_map.items()}
        return self

    def rotate_fine(self, pivot_pos, direction, radians):
        '''Rotate the positions in the molecule a certain number of radians, but don't change bonds yet.
        Used for step-wise rotation collision checks.
        '''
        self.atom_map = {posn.rotate_fine(pivot_pos, direction, radians=radians): atom
                         for posn, atom in self.atom_map.items()}
        return self

    def rotate_bonds(self, direction):
        '''Used for completing the rotation of a molecule after all the position collision checks are done.
        Rotate the bonds on each atom in this molecule but leave co-ordinates untouched
        '''
        for atom in self.atom_map.values():
            atom.rotate(direction)
        return self

    def round_posns(self):
        '''Used after performing float-precision movement collision checks. Round all posns in this molecule back to
        integers.
        '''
        self.atom_map = {posn.round(): atom for posn, atom in self.atom_map.items()}
        return self

    def check_collisions_lazy(self, other):
        '''Check for collisions with another molecule, assuming integer co-ordinates in both molecules.
        Do nothing if checked against itself.
        '''
        if other is not self:
            for posn in self.atom_map:
                if posn in other:
                    raise ReactionError("Collision between molecules.")

    def check_collisions(self, other):
        '''Check for collisions with another molecule while accounting for potential decimal positions.
        Do nothing if checked against itself.
        '''
        if other is not self:
            for posn in self.atom_map:
                for other_posn in other.atom_map:
                    if ((posn.col - other_posn.col)**2 + (posn.row - other_posn.row)**2
                            < ATOM_DIAMETER_SQUARED):
                        raise ReactionError("Collision between molecules.")

    def debond(self, position, direction):
        '''Decrement the specified bond. If doing so disconnects this molecule, mutate this molecule to its new size and
        return the extra molecule that was split off (else return None).
        '''
        posn_A = position
        atom_A = self.atom_map[posn_A]
        if direction not in atom_A.bonds:
            return None

        posn_B = posn_A + direction
        atom_B = self.atom_map[posn_B]
        direction_B = direction.opposite()

        # Decrement and/or remove the bond on each atom
        atom_A.bonds[direction] -= 1
        if atom_A.bonds[direction] == 0:
            del atom_A.bonds[direction]

        atom_B.bonds[direction_B] -= 1
        if atom_B.bonds[direction_B] == 0:
            del atom_B.bonds[direction_B]

        # Search from atom B to determine if we disconnected the molecule. If not, remove all atoms we discovered
        # connected via B from this molecule and add them to a new molecule
        visited_posns = set()
        visit_queue = [posn_B]
        while visit_queue:
            cur_posn = visit_queue.pop()
            visited_posns.add(cur_posn)

            for dirn in self.atom_map[cur_posn].bonds:
                neighbor_posn = cur_posn + dirn
                if neighbor_posn not in visited_posns:
                    visit_queue.append(neighbor_posn)

        if len(visited_posns) == len(self):
            # Molecule was not split, nothing left to do
            return None

        # Given that the molecule was split, create a new molecule from the positions that were accessible
        # from the disconnected neighbor (and remove those positions from this molecule)
        new_atom_map = {}
        for posn in visited_posns:
            new_atom_map[posn] = self.atom_map[posn]
            del self.atom_map[posn]

        return Molecule(atom_map=new_atom_map)

    def defrag(self, modified_posn):
        '''Given an atom position at which existing bonds have been broken by nuclear operations or a swap,
        check if they broke this molecule into two or more pieces. Return a list of the new molecules, with an ordering
        matching that empirically observed in SpaceChem. If the molecule has not broken apart the list will contain only
        a copy of itself.
        Args:
            modified_posn: The position at which an atom had its bonds reduced
        '''
        # Force the return order based on empirical data of how SpaceChem changes molecule priorities
        # Big hack since SpaceChem no doubt produces its order as a side effect of its graph algorithms,
        # which I'm not sure how to exactly mimic since it seems somewhat asymmetrical (East in particular)
        # Exerpt from https://www.reddit.com/r/spacechem/wiki/gamemechanics#wiki_molecule_selection_priority:
        # ```
        # Molecules are referred to by cardinal position, based on the bond(s) that attached them to the atom being
        # debonded from. That atom's molecule is referred to as the 'middle' molecule. Note that a molecule may be
        # bonded to multiple sides of the central atom. Molecules that remain bonded to the central atom after
        # bond removal is complete are not considered "debonded."
        # 1. A South-debonded molecule always has lowest priority (e.g. gets outputted last).
        #    This rule applies whether or not the molecule is also debonded from another direction.
        # 2. Subject to rule 1, a West-debonded molecule has higher priority than a North-debonded molecule, which has
        #    higher priority than the middle molecule. This rule applies whether or not the molecule is also debonded
        #    from East.
        # 3. If an East-debonded molecule is not debonded from West, North or South, it has exactly 2nd-highest priority
        #    out of all the molecules.
        # ```

        # This dict is sorted in the order molecules should be returned (with the exception of RIGHT, due to rule 3),
        # but populated in an order that respects the fact that anything attached to the modified atom counts as
        # 'middle', and that otherwise South's rule dominates the other directions' default orderings.
        # Note: the (0, 0) is a quick hack since it behaves like a Direction.NONE when added to a Position.
        new_molecules = {LEFT: None,
                         UP: None,
                         (0, 0): None,
                         RIGHT: None,
                         DOWN: None}

        # Search in an order that ensures other directions get put in the 'middle' or 'down' molecules first if possible
        # (see `not considered "debonded"` and rule 1 above for why middle and down are prioritized first)
        for search_dirn in ((0, 0), DOWN, LEFT, UP, RIGHT):
            start_posn = modified_posn + search_dirn
            # If we already found this posn attached to one of the other posns and removed it,
            # or it did't exist in the first places, skip it
            if start_posn not in self.atom_map:
                continue

            # Search from the neighbor posn to determine if we disconnected the molecule
            visited_posns = set()
            visit_queue = [start_posn]
            while visit_queue:
                cur_posn = visit_queue.pop()
                visited_posns.add(cur_posn)

                for dirn in self.atom_map[cur_posn].bonds:
                    neighbor_posn = cur_posn + dirn
                    if neighbor_posn not in visited_posns:
                        visit_queue.append(neighbor_posn)

            # Create a new molecule from the positions that were accessible from start_posn, and remove those positions
            # from this molecule
            new_atom_map = {}
            for posn in visited_posns:
                new_atom_map[posn] = self.atom_map[posn]
                del self.atom_map[posn]  # Ensures we don't make duplicate molecules

            new_molecules[search_dirn] = Molecule(atom_map=new_atom_map)

        # Per priority rule 3 above, if the right neighbor was not found in any of the other searches, force its
        # molecule to be second in the returned list
        mol_iter = (m for k, m in new_molecules.items() if m is not None and k != RIGHT)
        out = [next(mol_iter)]
        if new_molecules[RIGHT] is not None:
            out.append(new_molecules[RIGHT])
        out += mol_iter

        return out

    def output_zone_idx(self, large_output=False):
        if not self:
            return None

        if any(posn.col < 6 for posn in self.atom_map):
            return None
        elif large_output:
            return 0

        # Check if any atom doesn't match the output zone of the first atom
        posns = iter(self.atom_map)  # Keep a ref to the generator so we avoid re-checking the first position
        is_psi = next(posns).row < 4
        if any((posn.row < 4) != is_psi for posn in posns):
            return None

        # If we haven't returned yet, all atoms were in the same output zone
        return 0 if is_psi else 1

    def neighbor_bonds(self, posn):
        '''Helper to return tuples of (bond_count, neighbor_posn) for a given atom position.
        An atom must exist at the given position. Used by isomorphism algorithm.
        '''
        return ((bond_count, posn + direction)
                for direction, bond_count in self.atom_map[posn].bonds.items())

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
        # in the other molecule. We'll trickle the 'tree' down from each possible starting atom until we find a tree
        # that matches.
        rarest_atom_struct = min(this_atom_struct_to_freq.items(),
                                 key=lambda x: x[1])[0]
        our_root_posn = next(posn for posn, atom_struct in this_posn_to_atom_struct.items()
                             if atom_struct == rarest_atom_struct)

        def molecules_match_recursive(our_visited_posns: dict, our_posn: Position,
                                      their_visited_posns: dict, their_posn: Position):
            '''Attempt to recursively map any unvisited atoms of the molecules onto each other
            starting from the given position in each.

            If unsuccessful, revert any changes made to the given visit dicts, and return 0.
            If successful, return the number of items added to each dict.
            '''
            # Make sure the two atoms have the same element / bond counts
            if this_posn_to_atom_struct[our_posn] != other_posn_to_atom_struct[their_posn]:
                return 0

            # Make sure the two atoms have the same number of unvisited neighbors. This avoids issues if all of A's
            # neighbors get matched but unmatched neighbors of B remain, which shouldn't count as a successful match
            # even if they have the same atom structures
            if (sum(1 for _, our_neighbor in self.neighbor_bonds(our_posn)
                    if our_neighbor not in our_visited_posns)
                    != sum(1 for _, their_neighbor in other.neighbor_bonds(their_posn)
                           if their_neighbor not in their_visited_posns)):
                return 0

            # Mark the current nodes as visited (storing a dummy None; we'd use set() but we need ordering and popitem())
            total_visits = 1
            our_visited_posns[our_posn] = None
            their_visited_posns[their_posn] = None

            # Attempt to recursively match each of our bonded neighbors in turn (comparing against all possible
            # neighbors of theirs each time)
            for our_bond_count, our_neighbor in self.neighbor_bonds(our_posn):
                if our_neighbor not in our_visited_posns:  # Ignore already-matched positions
                    # Attempt to match any neighbor of the other molecule's atom to this neighbor
                    success = False
                    for their_bond_count, their_neighbor in other.neighbor_bonds(their_posn):
                        if their_neighbor not in their_visited_posns and our_bond_count == their_bond_count:
                            num_visits = molecules_match_recursive(our_visited_posns, our_neighbor,
                                                                   their_visited_posns, their_neighbor)
                            if num_visits:
                                # Success
                                success = True
                                total_visits += num_visits
                                break

                    if not success:
                        # If we couldn't find any matches for this neighbor, revert any changes to the visits we made
                        # while matching prior neighbors
                        for _ in range(total_visits):
                            our_visited_posns.popitem()
                            their_visited_posns.popitem()

                        return 0

            # If all unvisited neighbors were successfully matched, we succeeded; return the total visits we added
            return total_visits

        for their_root_posn in (posn for posn, atom_struct in other_posn_to_atom_struct.items()
                                if atom_struct == rarest_atom_struct):
            if molecules_match_recursive({}, our_root_posn, {}, their_root_posn):
                return True

        return False

    # We don't overload __hash__ (which by default checks instance matching) because we want to
    # preserve the ability to store molecules in a dictionary (ordered and provides O(1) add,
    # delete). Otherwise if we move a molecule its hash will change and it will no longer be
    # accessible in the dict.
    def hashable_repr(self):
        '''Return a hashable object representing this molecule, for use in comparing run states.'''
        return frozenset((posn, atom.hashable_repr()) for posn, atom in self.atom_map.items())
