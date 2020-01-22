#!/bin/env python
# 2020/01/22
# Stanislaw Grams <sjg@fmdx.pl>
# 09-3sat_project/src/literals.py
### NOTE: in boolean logic, a formula is in CNF (conjunctive normal form) if
###       it is a conjunction of one or more clauses, where a clause is a disjunction
###       of literals (it is an AND of ORs)
class Literals (object):
    def __init__ (self):
        self.literals  = set ()
        self.assigned  = []
        self.conflicts = []
        self.resolved  = []

    def empty (self):
        return True if not self.stack else False

    def push (self, literal):
        if literal not in self.literals:
            self.literals.add (literal)

    def pop (self):
        popped = self.stack.pop ()
        self.resolved (popped)
        return popped

    def assign (self, literal):
        assign = False
        if ((literal or -literal) not in self.assigned) and not self.unresolved (literal):
            self.assigned.append (literal)
            assign = True
        return assign

    def unassign (self, literal):
        if literal in self.stack:
            self.stack.remove (literal)
        if literal in self.conflicts:
            self.conflicts.remove (literal)
        if literal not in self.resolved:
            self.resolved.append (literal)

    def conflict (self, literal):
        self.conflicts.append (literal)

    def remove (self, literal):
        if literal in self.literals:
            self.literals.remove (literal)

    def resolved (self, literal):
        return ((literal or -literal) in self.resolved)

    def unresolved (self, literal):
        return ((literal or -literal) in self.conflicts)
