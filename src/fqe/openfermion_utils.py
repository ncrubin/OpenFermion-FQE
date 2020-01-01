#   Copyright 2019 Quantum Simulation Technologies Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Utilities which specifically require import from OpernFermion
"""

from typing import Any, Dict, KeysView, List, Optional, Tuple, TYPE_CHECKING

from openfermion import FermionOperator, up_index, down_index, QubitOperator
from openfermion.transforms import jordan_wigner, reverse_jordan_wigner

import numpy

from fqe.bitstring import gbit_index, integer_index, count_bits
from fqe.bitstring import lexicographic_bitstring_generator
from fqe.util import alpha_beta_electrons, bubblesort
from fqe.util import init_bitstring_groundstate, paritysort_int

if TYPE_CHECKING:
    from fqe.wavefunction import Wavefunction


def ascending_index_order(ops: 'FermionOperator',
                          coeff: complex,
                          order: str = '') -> 'FermionOperator':
    """Permute a product of creation FermionOperators so that the index goes
    from the lowest to the highest inside of each spin sector and multiply by
    the correct parity.

    Args:
        ops (FermionOperator) - A product of FermionOperators
        coeff (complex64) - The coefficient in front of the operator strings
        order (string) - specify if the operators should be returned in
            numerical order or not

    Returns:
        (FermionOperator) - A product of FermionOperators with the correct
            sign
    """

    ilist = []


    if order == 'inorder':
        for term in ops:
            ilist.append(term[0])
        nperm = bubblesort(ilist)
        ops = FermionOperator('', 1.0)
        for i in ilist:
            ops *= FermionOperator(str(i) + '^', 1.0)
    else:
        for term in ops:
            ilist.append(term[0])
        nperm, _ = paritysort_int(ilist)
        alpl = []
        betl = []
        for i in ilist:
            if i % 2 == 0:
                alpl.append(i)
            else:
                betl.append(i)
        nperm += bubblesort(alpl)
        nperm += bubblesort(betl)
        ops = FermionOperator('', 1.0)
        for i in alpl:
            ops *= FermionOperator(str(i) + '^', 1.0)
        for i in betl:
            ops *= FermionOperator(str(i) + '^', 1.0)
    return coeff*ops*(-1.0 + 0.0j)**nperm


def bit_to_fermion_creation(string: int,
                            spin: Optional[str] = None) -> 'FermionOperator':
    """Convert an occupation bitstring representation for a single spin case
    into openfermion operators

    Args:
        string (bitstring) - a string occupation representation
        spin (string) - a flag to indicate if the indexes of the occupation
            should be converted to alpha, beta or spinless

    Returns:
        ops (Openfermion operators) - a FermionOperator string
    """
    ops = None
    if string != 0:
        def _index_parser(spin):
            if spin is None:

                def _spinless_index(i):
                    return i
                return _spinless_index

            if spin.lower() == 'a' or spin.lower() == 'up':
                return up_index
            if spin.lower() == 'b' or spin.lower() == 'down':
                return down_index
            raise ValueError("Unidentified spin option passed to"
                             "bit_to_fermion_creation")

        spin_index = _index_parser(spin)
        gbit = gbit_index(string)
        ops = FermionOperator(str(spin_index(next(gbit))) + '^', 1. + .0j)
        for i in gbit:
            ops *= FermionOperator(str(spin_index(i)) + '^', 1. + .0j)
    return ops


def classify_symmetries(operator: 'FermionOperator') -> Tuple[bool]:
    """Given FermionOperators, classify it's effect on the fqe space.

    Args:
        ops (FermionOperator) - a FermionOperator

    Returns:
        (bool, bool) - return a pair of boolean values from the routine
            indicating if the particle number and spin will change
    """
    pn_change = 0
    spin_change = 0
    for cluster in operator.terms:
        for ops in cluster:
            pn_change += (-1)**(ops[1] + 1)
            spin_change += (-1)**(ops[0])

    return pn_change, spin_change


def fqe_to_fermion_operator(wfn: 'Wavefunction') -> 'FermionOperator':
    """Given an FQE wavefunction, convert it into strings of openfermion
    operators with appropriate coefficients for weights.

    Args:
        wfn (fqe.wavefunction)

    Returns:
        ops (FermionOperator)
    """
    ops = FermionOperator('', 1.0)
    genstate = wfn.generator()
    for sector in genstate:
        for alpha, beta, coeffcient in sector.generator():
            ops += coeffcient*determinant_to_ops(alpha, beta)
    return ops - FermionOperator('', 1.0)


def convert_qubit_wfn_to_fqe_syntax(ops: 'QubitOperator') -> 'FermionOperator':
    """This takes a qubit wavefunction in the form of a string of qubit
    operators and returns a string of FermionOperators with the proper
    formatting for easy digestion by FQE

    Args:
        ops (QubitOperator) - a string of qubit operators

    Returns:
        out (FermionOperator) - a string of fermion operators
    """
    ferm_str = reverse_jordan_wigner(ops)
    out = FermionOperator('', 1.0)
    for term in ferm_str.terms:
        out += ascending_index_order(term, ferm_str.terms[term])
    return out - FermionOperator('', 1.0)


def determinant_to_ops(a_str: int, b_str: int, inorder: bool = False) -> 'FermionOperator':
    """Given the alpha and beta occupation strings return, a fermion operator
    which would produce that state when acting on the vacuum.

    Args:
        a_str (bitstring) - spatial orbital occupation by the alpha electrons
        b_str (bitstring) - spatial orbital occupation by the beta electrons
        inorder (bool) - flag to control creation and annihilation order

    Returns:
        (FermionOperator)
    """
    if a_str + b_str == 0:
        out = FermionOperator('', 1.0)
    else:
        if inorder:
            occ = 0
            # Get occupation list from beta. Convert to spin down.
            for i in integer_index(b_str):
                occ += 2**down_index(i)
            for i in integer_index(a_str):
                occ += 2**up_index(i)
            out = bit_to_fermion_creation(occ)
        else:
            if b_str == 0:
                ops = 1.0
            else:
                ops = bit_to_fermion_creation(b_str, 'down')
            if a_str == 0:
                out = ops
            else:
                out = bit_to_fermion_creation(a_str, 'up')*ops
    return out


def fci_fermion_operator_representation(norb: int,
                                        nele: int,
                                        m_s: int) -> 'FermionOperator':
    """Generate the Full Configuration interaction wavefunction in the
    openfermion FermionOperator representation with coefficients of 1.0.

    Args:
        norb (int) - number of spatial orbitals
        nele (int) - number of electrons
        m_s (int) - s_z spin quantum number

    Returns:
        FermionOperator
    """
    nalpha, nbeta = alpha_beta_electrons(nele, m_s)
    gsstr = init_bitstring_groundstate(nalpha)
    alphadets = lexicographic_bitstring_generator(gsstr, norb)
    gsstr = init_bitstring_groundstate(nbeta)
    betadets = lexicographic_bitstring_generator(gsstr, norb)
    ops = FermionOperator('', 1.0)
    for bstr in betadets:
        for astr in alphadets:
            ops += determinant_to_ops(astr, bstr)
    return ops - FermionOperator('', 1.0)


def fci_qubit_representation(norb: int,
                             nele: int,
                             m_s: int) -> 'QubitOperator':
    """Create the qubit representation of Full CI according to the parameters
    passed

    Args:
        norb (int) - number of spatial orbitals
        nele (int) - number of electrons
        m_s (int) - spin projection onto sz

    Returns:
        ops (QubitOperator)
    """
    return jordan_wigner(fci_fermion_operator_representation(norb, nele, m_s))


def fermion_operator_to_bitstring(term: 'FermionOperator') -> Tuple[int, int]:
    """Convert an openfermion FermionOperator object into a bitstring
    representation.

    Args:
        cluster (FermionOperator) - A product of FermionOperators with a coefficient

    Returns:
        (bitstring, bitstring) - a pair of bitstrings representing the
            alpha and beta occupation in the orbitals.
    """
    upstring = 0
    downstring = 0
    for ops in term:
        if ops[0] % 2 == 0:
            upstring += 2**(ops[0]//2)
        else:
            downstring += 2**((ops[0]-1)//2)
    return upstring, downstring


def fermion_opstring_to_bitstring(ops) -> List[List[Any]]:
    """Convert an openfermion FermionOperator object into the corresponding
    bitstring representation with the appropriate coefficient.

    Args:
        ops (FermionOperator) - A product of FermionOperators with a coefficient

    Returns:
        list(list(int, int, coeff)) - a pair of integers representing a bitstring of
            alpha and beta occupation in the orbitals.
    """
    raw_data = []
    for term in ops.terms:
        rval = fermion_operator_to_bitstring(term)
        raw_data.append([rval[0], rval[1], ops.terms[term]])
    return raw_data


def generate_one_particle_matrix(ops: 'FermionOperator'
                                ) -> Tuple[numpy.ndarray]:
    """Convert a string of FermionOperators into a matrix.  If the dimension
    is not passed we will search the string to find the largest value.

    Args:
        ops (FermionOperator) - a string of FermionOperators

    Returns:
        Tuple(numpy.array(dtype=numpy.complex64),
              numpy.array(dtype=numpy.complex64),
              numpy.array(dtype=numpy.complex64))
    """
    ablk, bblk = largest_operator_index(ops)
    dim = max(2*ablk, 2*bblk)
    ablk = dim //2
    h1a = numpy.zeros((dim, dim), dtype=numpy.complex64)
    h1b = numpy.zeros((dim, dim), dtype=numpy.complex64)
    h1b_conj = numpy.zeros((dim, dim), dtype=numpy.complex64)
    for term in ops.terms:

        left, right = term[0][0], term[1][0]

        if left % 2:
            ind = (left - 1)//2 + ablk
        else:
            ind = left // 2

        if right % 2:
            jnd = (right - 1)//2 + ablk
        else:
            jnd = right // 2

        if term[0][1] and term[1][1]:
            h1b[ind, jnd] += ops.terms[term]
        elif term[0][1] and not term[1][1]:
            h1a[ind, jnd] += ops.terms[term]
        else:
            h1b_conj[ind, jnd] += ops.terms[term]

    return h1a, 2.0*h1b, 2.0*h1b_conj


def generate_two_particle_matrix(ops: 'FermionOperator') -> numpy.ndarray:
    """Convert a string of FermionOperators into a matrix.  If the operators
    in the term do not fit the action form of (1, 1, 0, 0) then permute them
    to git the form

    Args:
        ops (FermionOperator) - a string of FermionOperators

    Returns:
        numpy.array(dtype=numpy.complex64)
    """
    ablk, bblk = largest_operator_index(ops)
    dim = max(2*ablk, 2*bblk)
    ablk = dim // 2
    g2e = numpy.zeros((dim, dim, dim, dim), dtype=numpy.complex64)
    for term in ops.terms:

        iact, jact, kact, lact = term[0][1], term[1][1], term[2][1], term[3][1]

        if iact + jact + kact + lact != 2:
            raise ValueError('Unsupported four index tensor')

        first, second, third, fourth = term[0][0], term[1][0], term[2][0], term[3][0] 

        nper = 0
        if not iact:
            if not jact:
                jact, kact = kact, jact
                second, third = third, second
                nper += 1
            iact, jact = jact, iact
            first, second = second, first
            nper += 1

        if not jact:
            if not kact:
                kact, lact = lact, kact
                third, fourth = fourth, third
                nper += 1
            jact, kact = kact, jact
            second, third = third, second
            nper += 1

        if first % 2:
            ind = (first - 1)//2 + ablk
        else:
            ind = first // 2

        if second % 2:
            jnd = (second - 1)//2 + ablk
        else:
            jnd = second // 2

        if third % 2:
            knd = (third - 1)//2 + ablk
        else:
            knd = third // 2

        if fourth % 2:
            lnd = (fourth - 1)//2 + ablk
        else:
            lnd = fourth // 2

        g2e[ind, jnd, knd, lnd] = ((-1.0)**nper)*ops.terms[term]

    return g2e


def largest_operator_index(ops: 'FermionOperator') -> Tuple[int, int]:
    """Search through a FermionOperator string and return the largest even
    value and the largest odd value to find the total dimension of the matrix
    and the size of the spin blocks

    Args:
        ops (OpenFermion operators) - a string of operators comprising a
            Hamiltonian

    Returns:
        int, int - the largest even integer and the largest odd integer of the
            indexes passed to the subroutine.
    """
    maxodd = -1
    maxeven = -1
    for term in ops.terms:
        for ele in term:
            if ele[0] % 2:
                maxodd = max(ele[0], maxodd)
            else:
                maxeven = max(ele[0], maxeven)

    return maxeven, maxodd


def ladder_op(ind: int, step: int):
    """Local function to generate ladder operators in Qubit form given an index
    and action.

    Args:
        ind (int) - index of the orbital occupation to create
        step (int) - an integer indicating whether you should create or
        annhilate the orbital
    """
    xstr = 'X' + str(ind)
    ystr = 'Y' + str(ind)
    op_ = QubitOperator(xstr, 1. +.0j) + \
            ((-1)**step)*QubitOperator(ystr, 0. + 1.j)
    zstr = QubitOperator('', 1. +.0j)
    for i in range(ind):
        zstr *= QubitOperator('Z' + str(i), 1. +.0j)
    return zstr*op_/2.0


def mutate_config(stra: int,
                  strb: int,
                  term: Tuple[Tuple[int, int]]) -> Tuple[int, int, int]:
    """Apply creation and annihilation operators to a configuration in the
    bitstring representation and return the new configuration and the parity.

    Args:
        stra (bitstring) - the alpha string to be manipulated
        stra (bitstring) - the beta string to be manipulated
        term (Operators) - the operations to apply

    Returns:
        tuple(bitstring, bitstring, int) - the mutated alpha and beta
            configuration and the parity to go with it.
    """
    newa = stra
    newb = strb
    parity = 1

    for ops in reversed(term):
        if ops[0] % 2:
            abits = count_bits(newa)
            indx = (ops[0] - 1)//2
            if ops[1] == 1:
                if newb & 2**indx:
#                    return -1, -1, 0
                    return stra, strb, 0

                newb += 2**indx
            else:
                if not newb & 2**indx:
#                    return -1, -1, 0
                    return stra, strb, 0

                newb -= 2**indx

            lowbit = (1<<(indx)) - 1
            parity *= (-1)**(count_bits(lowbit & newb) + abits)

        else:
            indx = ops[0]//2
            if ops[1] == 1:
                if newa & 2**indx:
#                    return -1, -1, 0
                    return stra, strb, 0

                newa += 2**indx
            else:
                if not newa & 2**indx:
#                    return -1, -1, 0
                    return stra, strb, 0

                newa -= 2**indx

            lowbit = (1<<(indx)) - 1
            parity *= (-1)**count_bits(lowbit & newa)

    return newa, newb, parity


def new_wfn_from_ops(ops: 'FermionOperator', configs: KeysView[Tuple[int, int]],
                     norb: int) -> Tuple[List[List[int]], bool, bool]:
    """Look at each configuration in a wavefunction and compare it to the
    operations that will be applied.  Determine if there will be at least one
    non-zero configuration after this process and if so, add that configuration
    to the new wavefunction.

    Args:
        ops (FermionOperator) - a FermionOperator object
        configs (list[(int, int)]) - a list of keys signifying the current
            non-zero configurations in the wavefunction

    Returns:
        This returns three things.  First is a new set of parameters to build
            a new wavefunction from.  It also returns two boolean values to
            determine whether or not the particle number and spin is changing.

        list[list[nele, m_s, norb]], bool, bool
    """
    particlechange = False
    spinchange = False
    new_wavefunction_params: List[List[int]] = []
    difflist = particle_change(ops, norb)
    for diff in difflist:
        for val in configs:
            new = [diff[0] + val[0], diff[1] + val[1], norb]
            if new not in new_wavefunction_params:
                new_wavefunction_params.append(new)

    for diff in difflist:
        if diff[0] != 0:
            particlechange = True
            break

    for diff in difflist:
        if diff[1] != 0:
            spinchange = True
            break

    return new_wavefunction_params, particlechange, spinchange


def sector_change(ops: 'FermionOperator', norb: int) -> List[List[int]]:
#def particle_change(ops: 'FermionOperator', norb: int) -> List[List[int]]:
    """Given a FermionOperator, return the change in particle number and spin
    that will occur upon the application to a state.  Also check for operator
    motifs which will make the entire term zero or for operators outside the
    current space

    Args:
        ops (FermionOperator) - a set of creation and annihilation operators

    Returns:
        list[list[int, int]] - a list of list containing the change in particle
            number and change in spin
    """
    allowed = [-1, 0, 1]
    changes = []

    for term in ops.terms:
        spin = 0
        particles = 0
        err: Dict[int, int] = {}
        operr = False

        for operator in reversed(term):
            if operator[0] >= 2*norb:
                raise ValueError('{} is outside the orbital space'
                                 .format(operator[0]))

            if operator[0] not in err:
                err[operator[0]] = -(-1)**operator[1]
            else:
                err[operator[0]] -= (-1)**operator[1]

            if err[operator[0]] not in allowed:
                operr = True
                break

            particles -= (-1) ** operator[1]
            spin -= (-1) ** (operator[0] + operator[1])

        if not operr:
            changes.append([particles, spin])

    return changes


def split_openfermion_tensor(ops: 'FermionOperator'
                            ) -> Dict[int, 'FermionOperator']:
    """Given a string of openfermion operators, split them according to their
    rank.

    Args:
        ops (FermionOperator) - a string of Fermion Operators

    Returns:
        split dict[int] = FermionOperator - a list of Fermion Operators sorted
            according to their degree
    """
    split: Dict[int, 'FermionOperator'] = {}
    for term in ops:
        degree = term.many_body_order()
        if degree not in split:
            split[degree] = term
        else:
            split[degree] += term

    return split


def update_operator_coeff(operators: 'FermionOperator',
                          coeff: numpy.ndarray) -> None:
    """Given a string of Symbolic operators, set each prefactor equal to the
    value in coeff.  Note that the order in which SymbolicOperators in
    OpenFermion are printed out is not necessarily the order in which they are
    iterated over.

    Args:
        operators (FermionOperator) - A linear combination of operators.
        coeff (numpy.array(ndim=1,dtype=numpy.complex64)

    Returns:
        nothing - mutates coeff in place
    """
    for ops, val in zip(operators.terms, coeff):
        operators.terms[ops] = 1.0*val
