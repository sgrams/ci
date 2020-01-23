"""
2020/01/22
Stanislaw Grams <sjg@fmdx.pl>
09-3sat_project/src/basic_types.py
"""
class Equation():
    """
    describes a CNF equation

    NOTE: in boolean logic, a formula is in CNF(conjunctive normal form) if
    it is a conjunction of one or more clauses, where a clause is a disjunction
    of literals(it is an AND of ORs)
    """
    def __init__(self, variables: int, clauses: int, literals: list):
        self._variables = variables
        self._clauses = clauses
        self._literals = literals
        self._equation = self.__build_equation()

    def __repr__(self):
        return self._equation

    def __build_equation(self):
        equation = []
        for i, triplet in enumerate(self._literals):
            equation.append('(')
            for j, literal in enumerate(triplet):
                if literal < 0:
                    equation.append('~' + str(abs(literal)))
                else:
                    equation.append((str(literal)))
                if j != len(triplet) - 1:
                    equation.append('v')
            equation.append(')')

            if i != len(self._literals) - 1:
                equation.append('^')
        return ''.join(equation)

    @property
    def clauses(self) -> int:
        """ returns number of clauses in equation """
        return self._clauses

    @property
    def variables(self) -> int:
        """ returns number of variables in equation"""
        return self._variables

    @property
    def literals(self) -> list:
        """ returns list of literals in equation """
        return self._literals

class Chromosome():
    """ represents a single chromosome - solution to 3-SAT equation """
    def __init__(self, equation: Equation, genes: list):
        self._equation = equation
        self._genes = genes
        self._valid = None
        self._fitness = None

    def __getitem__(self, index: int):
        return self._genes[index]

    def __setitem__(self, key: int, value: int):
        self._genes[key] = value
        self._fitness = None
        self._valid = None

    def __len__(self) -> int:
        return len(self._genes)

    def __repr__(self) -> str:
        return ''.join(map(str, self._genes))

    @property
    def equation(self) -> Equation:
        """ returns assigned equation """
        return self._equation

    @property
    def validate(self):
        """ validates chromosome against 3-SAT formula """
        passed_clauses = 0
        for literals in self._equation.literals:
            for literal in literals:
                if literal < 0:
                    if not self._genes[abs(literal) - 1]:
                        passed_clauses += 1
                        break
                else:
                    if self._genes[literal - 1]:
                        passed_clauses += 1
                        break
        return passed_clauses, passed_clauses == self.equation.clauses

    @property
    def fitness(self):
        """ fitness function """
        if self._fitness is None:
            clauses, passed = self.validate
            self._fitness = clauses / self.equation.clauses
            self._valid = passed
        return self._fitness

    @property
    def genes(self):
        """ returns genes of the chromosome """
        return self._genes

    @genes.setter
    def genes(self, value: str):
        self._genes = value
        self._fitness = None
        self._valid = None

    @property
    def valid(self) -> bool:
        """ returns true if chromosome is valid """
        return self.validate[1]

    def copy(self):
        """ copies itself into new object """
        return Chromosome(self._equation, self._genes[:])
