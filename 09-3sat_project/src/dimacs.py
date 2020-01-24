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
        self._loaded = False
        self._clauses = 0
        self._loaded_clauses = 0
        self._variables = 0

    @property
    def filepath(self) -> str:
        """ returns filepath assigned to Dimacs class """
        return self._filepath

    def validate(self) -> bool:
        """ validates Dimacs object
            (checks numbers of loaded clauses with given number of clauses)
        """
        if self._loaded is False:
            return False
        return self._clauses == self._loaded_clauses

    def parse(self) -> Equation:
        """ parses DIMACS CNF file into Equation object """
        literals_list = list()
        if os.path.isfile(self.filepath):
            with open(self.filepath) as data:
                lines = (line.rstrip('\r\n') for line in data)
                for line in lines:
                    is_desc = regex.match(r"^" + DIMACS_DESC_CHAR, line)
                    is_clause = regex.match(r"^[0-9]|^\-", line)
                    line_splitted = line.strip().rsplit()

                    if is_desc:
                        self._variables = int(line_splitted[DIMACS_VARS_INDEX])
                        self._clauses = int(line_splitted[DIMACS_CLAUSES_INDEX])
                    if is_clause:
                        clause = [int(line_splitted[i]) for i in range(3)]
                        literals_list.append(tuple(clause))

            self._loaded_clauses = len(literals_list)
            self._loaded = True
            return Equation(self._variables, self._clauses, literals_list)
        return None
