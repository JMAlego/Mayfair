#!/usr/bin/env python3
"""Mayfair constraint solver.

Basic solver for CONS module.
"""

from itertools import product
from string import ascii_uppercase as alphabet
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

VariableReference = int


class Variable:
    """A variable."""

    _INDEX = 0

    def __init__(self, pretty_name: Optional[str] = None) -> None:
        """Initialise variable."""
        self._values: Set[int] = set()
        if pretty_name is None:
            self._pretty_name = "Var{}".format(Variable._INDEX)
            Variable._INDEX += 1
        else:
            self._pretty_name = pretty_name

    @staticmethod
    def from_range(domain_end, domain_start=1, pretty_name: Optional[str] = None) -> "Variable":
        """Create a variable with a range based domain."""
        new_domain = Variable(pretty_name=pretty_name)
        new_domain._values = set(range(domain_start, domain_end + 1))
        return new_domain

    def copy(self) -> "Variable":
        """Create a copy of the variable."""
        new_domain = Variable(pretty_name=self._pretty_name)
        new_domain._values = set(self._values)
        return new_domain

    def prune(self, values: Set[int]):
        """Prune specified values."""
        self._values.difference_update(values)

    def assign(self, value: int):
        """Prune specified values."""
        self._values = set((value, ))

    @property
    def values(self) -> Set[int]:
        """Get the values which are in the variable's domain."""
        return self._values

    def __contains__(self, other) -> bool:
        """Implement for "in" operator."""
        return other in self._values

    def pretty(self) -> str:
        """Pretty print the variable."""
        return self._pretty_name


class VariableDomain:
    """Variable domain."""

    def __init__(self) -> None:
        """Initialise domain."""
        self.variables: List[Variable] = []

    @property
    def variable_references(self) -> Set[VariableReference]:
        """Get all variable references in this domain."""
        return set(range(len(self.variables)))

    def from_name(self, name: str) -> VariableReference:
        """Get a variable by name in the domain."""
        index = 0
        for item in self.variables:
            if item.pretty() == name:
                return index
            index += 1
        raise IndexError("No variable by that name")

    @staticmethod
    def from_range(domain_count, domain_end, domain_start=1, alpha_names=False) -> "VariableDomain":
        """Create a domain from a range and variable count."""
        new_domains_container = VariableDomain()
        for i in range(domain_count):
            pretty_name = None
            if alpha_names:
                pretty_name = alphabet[i % len(alphabet)] * (1 + i // len(alphabet))
            new_domains_container.variables.append(
                Variable.from_range(domain_end, domain_start, pretty_name=pretty_name))
        return new_domains_container

    def copy(self) -> "VariableDomain":
        """Create a copy of the domain."""
        new_domains_container = VariableDomain()
        new_domains_container.variables = [variable.copy() for variable in self.variables]
        return new_domains_container

    def get_variable(self, v: VariableReference) -> Variable:
        """Get a variable from the domain."""
        return self.variables[v]

    def __getitem__(self, key) -> Variable:
        """Implement indexing to get variable from the domain."""
        return self.get_variable(key)


class UnaryConstraint(ABC):
    """A constraint on one variable."""

    def __hash__(self) -> int:
        """Hash for a constraint."""
        return hash(self.__class__.__module__ + self.__class__.__name__ + "|" + str(self.x) + "|" +
                    str(self.v))

    def __init__(self, x: VariableReference, v: int) -> None:
        """Initialise constraints."""
        self.x = x
        self.v = v

    def __contains__(self, other) -> bool:
        """Check whether this constraint is affected by the provided variable(s)."""
        if isinstance(other, set):
            return self.x in other
        return other == self.x

    @abstractmethod
    def pretty(self, vd: Optional[VariableDomain] = None) -> str:
        """Pretty print the constraint."""

    @abstractmethod
    def revise(self, vd: VariableDomain) -> bool:
        """Return the revised domain if a revision was made."""


class BinaryConstraint(ABC):
    """A constraint on two variables."""

    def __hash__(self) -> int:
        """Hash for a constraint."""
        return hash(self.__class__.__module__ + self.__class__.__name__ + "|" + str(self.x) + "|" +
                    str(self.y))

    def __init__(self, x: VariableReference, y: VariableReference) -> None:
        """Initialise constraints."""
        self.x = x
        self.y = y

    def _revise_condition(self, vd, condition: Callable[[VariableReference, VariableReference],
                                                        bool]):
        x, y = self.pair
        x_values = vd[x].values
        y_values = vd[y].values
        unsupported_values: Set[int] = set()
        for x_value in x_values:
            supported = False
            for y_value in y_values:
                if condition(x_value, y_value):
                    supported = True
                    break
            if not supported:
                unsupported_values.add(x_value)
        vd[x].prune(unsupported_values)

        return bool(vd[x].values)

    @abstractmethod
    def revise(self, vd: VariableDomain) -> bool:
        """Return the revised domain if a revision was made."""

    def __contains__(self, other) -> bool:
        """Check whether this constraint is affected by the provided variable(s)."""
        if isinstance(other, set):
            return self.x in other or self.y in other
        return other in (self.x, self.y)

    @property
    def pair(self) -> Tuple[VariableReference, VariableReference]:
        """Get x and y as a tuple."""
        return (self.x, self.y)

    @abstractmethod
    def pretty(self, vd: Optional[VariableDomain] = None) -> str:
        """Pretty print the constraint."""


class AdjacencyConstraint(BinaryConstraint):
    """Constraint that two variables are not adjacent."""

    def revise(self, vd: VariableDomain) -> bool:
        """Revise, returning false if x is empty."""
        if len(vd[self.y].values) > 4:
            return True

        return self._revise_condition(vd, lambda x, y: abs(x - y) > 1)

    def pretty(self, vd: Optional[VariableDomain] = None) -> str:
        """Pretty print the constraint."""
        if vd is not None:
            return "|{} - {}| > 1".format(vd[self.x].pretty(), vd[self.y].pretty())
        return "|x - y| > 1"


class NotEqualConstraint(BinaryConstraint):
    """Constraint that two variables are not equal."""

    def revise(self, vd: VariableDomain) -> bool:
        """Return the revised domain if a revision was made."""
        # We could use the generic constraint:
        #   return self._revise_condition(vd, lambda x, y: x != y)
        # but we can do better...

        x, y = self.pair

        if not vd[x].values or not vd[y].values:
            return False

        if not len(vd[y].values) > 1:
            vd[x].values.difference_update(vd[y].values)

        return bool(vd[x].values)

    def pretty(self, vd: Optional[VariableDomain] = None) -> str:
        """Pretty print the constraint."""
        if vd is not None:
            return "{} != {}".format(vd[self.x].pretty(), vd[self.y].pretty())
        return "x != y"


class GenericUnaryConstraint(UnaryConstraint):
    """A constraint on one variable."""

    def __init__(self,
                 x: VariableReference,
                 v: int,
                 constraint_condition: Callable[[VariableReference, int], bool],
                 operator_format: str = "{} ⊙ {}") -> None:
        """Initialise constraints."""
        super().__init__(x, v)
        self._condition = constraint_condition
        self._operator_format = operator_format

    def revise(self, vd: VariableDomain) -> bool:
        """Return the revised domain if a revision was made."""
        values_to_prune: Set[int] = set()
        for value in vd[self.x].values:
            if not self._condition(value, self.v):
                values_to_prune.add(value)

        vd[self.x].prune(values_to_prune)

        return bool(vd[self.x].values)

    def pretty(self, vd: Optional[VariableDomain] = None) -> str:
        """Pretty print the constraint."""
        if vd is not None:
            return self._operator_format.format(vd[self.x].pretty(), self.v)
        return self._operator_format.format("x", self.v)


class GenericBinaryConstraint(BinaryConstraint):
    """Constraint that two variables are not equal."""

    def __init__(self,
                 x: VariableReference,
                 y: VariableReference,
                 constraint_condition: Callable[[VariableReference, VariableReference], bool],
                 operator_format: str = "{} ⊙ {}") -> None:
        """Initialise constraints."""
        super().__init__(x, y)
        self._condition = constraint_condition
        self._operator_format = operator_format

    def revise(self, vd: VariableDomain) -> bool:
        """Return the revised domain if a revision was made."""
        return self._revise_condition(vd, self._condition)

    def pretty(self, vd: Optional[VariableDomain] = None) -> str:
        """Pretty print the constraint."""
        if vd is not None:
            return self._operator_format.format(vd[self.x].pretty(), vd[self.y].pretty())
        return self._operator_format.format("x", "y")


def bidirectional(constraint: BinaryConstraint) -> Tuple[BinaryConstraint, BinaryConstraint]:
    """Take a single constraint and return the constraint and it's inverse."""
    return constraint, constraint.__class__(constraint.y, constraint.x)


Constraint = Union[BinaryConstraint, UnaryConstraint]


class Constraints:
    """Constraints container."""

    def __init__(self) -> None:
        """Initialise constraints container."""
        self.constraints: Set[Constraint] = set()
        self._dict_constraints: Dict[Tuple[VariableReference, ...], Set[BinaryConstraint]] = dict()

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint."""
        if isinstance(constraint, BinaryConstraint):
            if constraint.pair not in self._dict_constraints:
                self._dict_constraints[constraint.pair] = set()
            self._dict_constraints[constraint.pair].add(constraint)
        self.constraints.add(constraint)

    def add_constraints(self, constraints: Iterable[Constraint]) -> None:
        """Add a constraint."""
        for constraint in constraints:
            self.add_constraint(constraint)

    def relevant_constraints(self, variables: Set[int]) -> Set[Constraint]:
        """Get constraints which mention specified variables."""
        result: Set[Constraint] = set()
        for constraint in self.constraints:
            if variables in constraint:
                result.add(constraint)
        return result

    def unary_constraints(self) -> Set[UnaryConstraint]:
        """Get all unary constraints."""
        constraints: Set[UnaryConstraint] = set()
        for constraint in self.constraints:
            if isinstance(constraint, UnaryConstraint):
                constraints.add(constraint)
        return constraints

    def arc_constraints(self, x, y) -> Set[BinaryConstraint]:
        """Get all constraints on an arc."""
        if (x, y) in self._dict_constraints:
            return self._dict_constraints[(x, y)]
        return set()


def AllDifferent(*variables) -> Set[BinaryConstraint]:
    """Enforce that all the specified variables are different."""
    constraints: Set[BinaryConstraint] = set()
    for x, y in product(variables, variables):
        if x == y:
            continue
        constraints.add(NotEqualConstraint(x, y))
    return constraints


class ForwardChecker:
    """Forward checker."""

    def __init__(self, constraints: Constraints, debug=False) -> None:
        """Initialise forward checker."""
        self.constraints = constraints
        self.debug = debug

    def forward_check(self,
                      vd: VariableDomain,
                      variable: VariableReference = 0) -> Optional[VariableDomain]:
        """Run forward checker on a domain starting at a variable."""
        if self.debug:
            print("    " * variable, "-> Forward checking at depth {}".format(variable))

        for unary_constraint in self.constraints.unary_constraints():
            unary_constraint.revise(vd)

        max_depth = len(vd.variables)
        for value in vd[variable].values:
            if self.debug:
                print("    " * variable, "  = Trying variable assignment", vd[variable].pretty(),
                      "=", value)

            working_vd = vd.copy()
            working_vd[variable].assign(value)

            consistent = True
            for other_variable in range(variable + 1, max_depth):
                if not consistent:
                    break
                for constraint in self.constraints.arc_constraints(other_variable, variable):
                    consistent = consistent and constraint.revise(working_vd)
                    if self.debug:
                        print("    " * variable, "    Constraint", constraint.pretty(working_vd),
                              "application with", working_vd[other_variable].pretty(), "=",
                              working_vd[other_variable].values, "and",
                              working_vd[variable].pretty(), "=", working_vd[variable].values,
                              "gives consistent =", consistent)
                    if not consistent:
                        break
            if consistent:
                if variable + 1 == max_depth:
                    if self.debug:
                        print("    " * variable, "<- Solution found")
                    return working_vd
                else:
                    result = self.forward_check(working_vd, variable + 1)
                    if result:
                        return result
        if self.debug:
            print("    " * variable, "<- No assignments found")
        return None


def _test():
    """Run an example using the Crystal Maze problem."""
    constraints = Constraints()
    domain = VariableDomain.from_range(domain_count=8,
                                       domain_start=1,
                                       domain_end=8,
                                       alpha_names=True)

    connected = {
        (domain.from_name("A"), domain.from_name("B")),
        (domain.from_name("A"), domain.from_name("C")),
        (domain.from_name("A"), domain.from_name("D")),
        (domain.from_name("B"), domain.from_name("C")),
        (domain.from_name("D"), domain.from_name("C")),
        (domain.from_name("H"), domain.from_name("E")),
        (domain.from_name("H"), domain.from_name("F")),
        (domain.from_name("H"), domain.from_name("G")),
        (domain.from_name("E"), domain.from_name("F")),
        (domain.from_name("G"), domain.from_name("F")),
        (domain.from_name("B"), domain.from_name("E")),
        (domain.from_name("C"), domain.from_name("F")),
        (domain.from_name("D"), domain.from_name("G")),
        (domain.from_name("B"), domain.from_name("F")),
        (domain.from_name("E"), domain.from_name("C")),
        (domain.from_name("C"), domain.from_name("G")),
        (domain.from_name("D"), domain.from_name("F")),
    }

    for connected_a, connected_b in connected:
        constraints.add_constraints(bidirectional(AdjacencyConstraint(connected_a, connected_b)))

    constraints.add_constraints(AllDifferent(*domain.variable_references))

    solution = ForwardChecker(constraints).forward_check(domain)
    if solution:
        for variable in solution.variables:
            print("{} = {}".format(variable.pretty(), next(variable.values.__iter__())))


if __name__ == "__main__":
    _test()
