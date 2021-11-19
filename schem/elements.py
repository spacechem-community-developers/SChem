#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple


class Element(namedtuple("Element", ('atomic_num', 'symbol', 'max_bonds'))):
    """Immutable class representing a SpaceChemical element."""
    __slots__ = ()

    def __str__(self):
        return self.symbol
    __repr__ = __str__


elements = [Element(0, '?', 12),
            Element(1, 'H', 1),
            Element(2, 'He', 0),
            Element(3, 'Li', 1),
            Element(4, 'Be', 2),
            Element(5, 'B', 3),
            Element(6, 'C', 4),
            Element(7, 'N', 5),
            Element(8, 'O', 2),
            Element(9, 'F', 1),
            Element(10, 'Ne', 0),
            Element(11, 'Na', 1),
            Element(12, 'Mg', 2),
            Element(13, 'Al', 4),
            Element(14, 'Si', 4),
            Element(15, 'P', 5),
            Element(16, 'S', 6),
            Element(17, 'Cl', 7),
            Element(18, 'Ar', 0),
            Element(19, 'K', 1),
            Element(20, 'Ca', 2),
            Element(21, 'Sc', 3),
            Element(22, 'Ti', 4),
            Element(23, 'V', 5),
            Element(24, 'Cr', 6),
            Element(25, 'Mn', 7),
            Element(26, 'Fe', 6),
            Element(27, 'Co', 5),
            Element(28, 'Ni', 4),
            Element(29, 'Cu', 4),
            Element(30, 'Zn', 2),
            Element(31, 'Ga', 3),
            Element(32, 'Ge', 4),
            Element(33, 'As', 5),
            Element(34, 'Se', 6),
            Element(35, 'Br', 7),
            Element(36, 'Kr', 0),
            Element(37, 'Rb', 1),
            Element(38, 'Sr', 2),
            Element(39, 'Y', 3),
            Element(40, 'Zr', 4),
            Element(41, 'Nb', 5),
            Element(42, 'Mo', 6),
            Element(43, 'Tc', 7),
            Element(44, 'Ru', 8),
            Element(45, 'Rh', 6),
            Element(46, 'Pd', 4),
            Element(47, 'Ag', 3),
            Element(48, 'Cd', 2),
            Element(49, 'In', 3),
            Element(50, 'Sn', 4),
            Element(51, 'Sb', 5),
            Element(52, 'Te', 6),
            Element(53, 'I', 7),
            Element(54, 'Xe', 0),
            Element(55, 'Cs', 1),
            Element(56, 'Ba', 2),
            Element(57, 'La', 3),
            Element(58, 'Ce', 4),
            Element(59, 'Pr', 4),
            Element(60, 'Nd', 3),
            Element(61, 'Pm', 3),
            Element(62, 'Sm', 3),
            Element(63, 'Eu', 3),
            Element(64, 'Gd', 3),
            Element(65, 'Tb', 4),
            Element(66, 'Dy', 3),
            Element(67, 'Ho', 3),
            Element(68, 'Er', 3),
            Element(69, 'Tm', 3),
            Element(70, 'Yb', 3),
            Element(71, 'Lu', 3),
            Element(72, 'Hf', 4),
            Element(73, 'Ta', 5),
            Element(74, 'W', 6),
            Element(75, 'Re', 7),
            Element(76, 'Os', 8),
            Element(77, 'Ir', 6),
            Element(78, 'Pt', 6),
            Element(79, 'Au', 5),
            Element(80, 'Hg', 4),
            Element(81, 'Tl', 3),
            Element(82, 'Pb', 4),
            Element(83, 'Bi', 5),
            Element(84, 'Po', 6),
            Element(85, 'At', 7),
            Element(86, 'Rn', 0),
            Element(87, 'Fr', 1),
            Element(88, 'Ra', 2),
            Element(89, 'Ac', 3),
            Element(90, 'Th', 4),
            Element(91, 'Pa', 5),
            Element(92, 'U', 6),
            Element(93, 'Np', 7),
            Element(94, 'Pu', 7),
            Element(95, 'Am', 6),
            Element(96, 'Cm', 4),
            Element(97, 'Bk', 4),
            Element(98, 'Cf', 4),
            Element(99, 'Es', 3),
            Element(100, 'Fm', 3),
            Element(101, 'Md', 3),
            Element(102, 'No', 3),
            Element(103, 'Lr', 3),
            Element(104, 'Rf', 4),
            Element(105, 'Db', 5),
            Element(106, 'Sg', 6),
            Element(107, 'Bh', 7),
            Element(108, 'Hs', 8),
            Element(109, 'Mt', 6),

            # Greek Elements
            Element(200, 'Θ', 12),
            Element(201, 'Ω', 12),
            Element(202, 'Σ', 12),
            Element(203, 'Δ', 12),

            # Australium
            Element(204, 'Av', 5)]  # 🦘


class ElementDict(dict):
    """Class for looking up elements by atomic number, augmented internally to allow lookup by atomic symbol too."""
    __slots__ = 'symbol_dict',

    def __init__(self, elements):
        super().__init__()
        self.symbol_dict = {}

        for element in elements:
            self[element.atomic_num] = self.symbol_dict[element.symbol] = element

    def __contains__(self, key):
        return super().__contains__(key) or key in self.symbol_dict

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            return self.symbol_dict[key]
        else:
            raise TypeError(f"Elements must be looked up by atomic # or symbol; received {key}")


elements_dict = ElementDict(elements)
