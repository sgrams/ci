#!/bin/env python
# 2020/01/21
# Stanislaw Grams <sjg@fmdx.pl>
# 09-3sat_project/src/dimacs.py
import os, regex
from literals import Literals

### NOTE: class Dimacs parses text file in DIMACS CNF input format,
### where comments start with 'c' character AND
### where descriptions start wiht 'p' character
### More on DIMACS: https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html
DIMACS_DESC_CHAR        = 'p'
DIMACS_VARS_INDEX       = 2
DIMACS_CLAUSES_INDEX    = 3
DIMACS_NUM_VARS_IN_LINE = 3

class Dimacs (object):
    def __init__ (self, filepath):
        self.filepath = filepath
        self.clauses  = []
        self.literals = Literals ()

    def parse (self):
        literals_list = []
        if os.path.isfile (str (self.filepath)):
            with open (self.filepath) as data:
                lines = (line.rstrip ('\r\n') for line in data)
                for line in lines:
                    is_desc   = regex.match ("^" + DIMACS_DESC_CHAR, line) ## descriptions begin with p
                    is_clause = regex.match ("^[0-9]|^\-", line)
                    line_splitted = line.rsplit ()

                    if is_desc:
                        print ("num_vars=%i, num_clauses=%i" % (int(line_splitted[DIMACS_VARS_INDEX]),
                            int(line_splitted[DIMACS_CLAUSES_INDEX])))
                    if is_clause:
                        clause = [int (line_splitted[i]) for i in range (3)]
                        print ("clause = " + str (clause))
                        self.clauses.append (tuple(clause))
                        literals_list.extend (tuple(clause))
            [self.literals.push (x) for x in literals_list]
