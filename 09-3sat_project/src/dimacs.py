"""
2020/01/21
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/dimacs.py
"""
import os
import regex
from basic_types import Equation

### NOTE: class Dimacs parses text file in DIMACS CNF input format,
### where comments start with 'c' character AND
### where descriptions start wiht 'p' character
### More on DIMACS: https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html
DIMACS_DESC_CHAR = 'p'
DIMACS_VARS_INDEX = 2
DIMACS_CLAUSES_INDEX = 3
DIMACS_NUM_VARS_IN_LINE = 3

class Dimacs():
    """ represents ands fetches input from DIMACS format """
    def __init__(self, filepath: str):
        self._filepath = filepath

    @property
    def filepath(self) -> str:
        """ returns filepath assigned to Dimacs class """
        return self._filepath

    @property
    def equation(self) -> Equation:
        """ parses DIMACS CNF file into Equation object """
        literals_list = []
        variables = 0
        clauses = 0
        if os.path.isfile(self.filepath):
            with open(self.filepath) as data:
                lines = (line.rstrip('\r\n') for line in data)
                for line in lines:
                    is_desc = regex.match(r"^" + DIMACS_DESC_CHAR, line)
                    is_clause = regex.match(r"^[0-9]|^\-", line)
                    line_splitted = line.rsplit()
                    clauses_read = 0

                    if is_desc:
                        variables = int(line_splitted[DIMACS_VARS_INDEX])
                        clauses = int(line_splitted[DIMACS_CLAUSES_INDEX])
                    if is_clause and clauses_read <= clauses:
                        clauses_read += 1
                        clause = [int(line_splitted[i]) for i in range(3)]
                        literals_list.append(tuple(clause))
            return Equation(variables, clauses, literals_list)
        return None
