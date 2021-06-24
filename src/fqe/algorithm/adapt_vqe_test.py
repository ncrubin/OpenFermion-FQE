#   Copyright 2020 Google LLC

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
"""Unit tests for Adapt-VQE."""

import numpy as np
import pytest

import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

import fqe
from fqe.algorithm.adapt_vqe import OperatorPool, ADAPT
from fqe.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata,)

def get_molecule():
    from openfermionpyscf import run_pyscf

    dim = 1.5
    geometry = [['Li', [0, 0, 0]], ['H', [0, dim, 0]]]
    basis = 'sto-3g'
    charge = 0
    multiplicity = 1
    molecule = of.MolecularData(geometry=geometry,
                                charge=charge,
                                multiplicity=multiplicity,
                                basis=basis)
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    return molecule


def test_op_pool():
    op = OperatorPool(2, [0], [1])
    op.singlet_t2()
    true_generator = of.FermionOperator(((1, 1), (0, 1), (3, 0), (2, 0))) - \
                     of.FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)))

    assert np.isclose(
        of.normal_ordered(op.op_pool[0] - true_generator).induced_norm(), 0)

    op = OperatorPool(2, [0], [1])
    op.two_body_sz_adapted()
    assert len(op.op_pool) == 14

    true_generator0 = of.FermionOperator('1^ 0^ 2 1') - \
                      of.FermionOperator('2^ 1^ 1 0')
    assert np.isclose(
        of.normal_ordered(op.op_pool[0] - true_generator0).induced_norm(), 0)

    true_generator_end = of.FermionOperator('3^ 0^ 3 2') - \
                         of.FermionOperator('3^ 2^ 3 0')
    assert np.isclose(
        of.normal_ordered(op.op_pool[-1] - true_generator_end).induced_norm(),
        0)

    op = OperatorPool(2, [0], [1])
    op.one_body_sz_adapted()
    for gen in op.op_pool:
        for ladder_idx, _ in gen.terms.items():
            # check if one body terms are generated
            assert len(ladder_idx) == 2


def test_adapt():
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    oei, tei = molecule.get_integrals()
    norbs = molecule.n_orbitals
    nalpha = molecule.n_electrons // 2
    nbeta = nalpha
    sz = nalpha - nbeta
    occ = list(range(nalpha))
    virt = list(range(nalpha, norbs))
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(molecular_hamiltonian,
                                                   molecule.n_electrons)

    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')

    sop = OperatorPool(norbs, occ, virt)
    sop.two_body_sz_adapted()  # initialize pool
    adapt = ADAPT(oei, tei, sop, nalpha, nbeta, iter_max=1, verbose=False)
    assert np.isclose(
        np.linalg.norm(adapt.reduced_ham.two_body_tensor -
                       reduced_ham.two_body_tensor), 0)

    adapt.adapt_vqe(fqe_wf)
    assert np.isclose(adapt.energies[0], -8.957417182801091)
    assert np.isclose(adapt.energies[-1], -8.970532463968661)

    sop = OperatorPool(norbs, occ, virt)
    sop.one_body_sz_adapted()
    adapt = ADAPT(oei,
                  tei,
                  sop,
                  nalpha,
                  nbeta,
                  iter_max=10,
                  stopping_epsilon=10,
                  verbose=True)
    adapt.adapt_vqe(fqe_wf)
    assert np.isclose(adapt.energies[-1], -8.957417182801091)
    assert np.isclose(adapt.energies[0], -8.95741717733075)


def cost_func(params, pool, initial_wf, sdim, k2_fop, s2_fqe, s2_penalty, ss, shift):
    import copy
    from fqe.fqe_ops.fqe_ops import S2Operator
    from fqe.hamiltonians.hamiltonian import Hamiltonian as ABCHamiltonian
    opt_method='BFGS'

    assert len(params) == len(pool)
    # compute wf for function call
    wf = copy.deepcopy(initial_wf)
    for op, coeff in zip(pool, params):
        if np.isclose(coeff, 0):
            continue
        if isinstance(op, ABCHamiltonian):
            fqe_op = op
        else:
            print("Found a OF Hamiltonian")
            fqe_op = build_hamiltonian(1j * op,
                                       sdim,
                                       conserve_number=True)
        if isinstance(fqe_op, ABCHamiltonian):
            wf = wf.time_evolve(coeff, fqe_op)
        else:
            raise ValueError("Can't evolve operator type {}".format(
                type(fqe_op)))

    return wf.expectationValue(k2_fop).real

def cost_func_grad(params, pool, initial_wf, sdim, k2_fop):
    import copy
    from fqe.fqe_ops.fqe_ops import S2Operator
    from fqe.hamiltonians.hamiltonian import Hamiltonian as ABCHamiltonian
    opt_method='BFGS'

    assert len(params) == len(pool)
    # compute wf for function call
    wf = copy.deepcopy(initial_wf)
    for op, coeff in zip(pool, params):
        if np.isclose(coeff, 0):
            continue
        if isinstance(op, ABCHamiltonian):
            fqe_op = op
        else:
            print("Found a OF Hamiltonian")
            fqe_op = build_hamiltonian(1j * op,
                                       sdim,
                                       conserve_number=True)
        if isinstance(fqe_op, ABCHamiltonian):
            wf = wf.time_evolve(coeff, fqe_op)
        else:
            raise ValueError("Can't evolve operator type {}".format(
                type(fqe_op)))

    # compute gradients
    grad_vec = np.zeros(len(params), dtype=np.complex128)
    # avoid extra gradient computation if we can
    for pidx, _ in enumerate(params):
        # evolve e^{iG_{n-1}g_{n-1}}e^{iG_{n-2}g_{n-2}}x
        # G_{n-3}e^{-G_{n-3}g_{n-3}...|0>
        grad_wf = copy.deepcopy(initial_wf)
        for gidx, (op, coeff) in enumerate(zip(pool, params)):
            if isinstance(op, ABCHamiltonian):
                fqe_op = op
            else:
                fqe_op = build_hamiltonian(1j * op,
                                           sdim,
                                           conserve_number=True)
            if not np.isclose(coeff, 0):
                grad_wf = grad_wf.time_evolve(coeff, fqe_op)
                # if looking at the pth parameter then apply the
                # operator to the state
            if gidx == pidx:
                grad_wf = grad_wf.apply(fqe_op)

        grad_val = grad_wf.expectationValue(k2_fop, brawfn=wf)
        print(grad_val)
        grad_vec[pidx] = -1j * grad_val + 1j * grad_val.conj()
        assert np.isclose(grad_vec[pidx].imag, 0)

    return grad_vec


def pyscf_to_fqe_wf(pyscf_cimat, pyscf_mf=None, norbs=None, nelec=None):
    from pyscf.fci.cistring import make_strings
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    norb_list = tuple(list(range(norbs)))
    n_alpha_strings = [x for x in make_strings(norb_list, nelec[0])]
    n_beta_strings = [x for x in make_strings(norb_list, nelec[1])]

    fqe_wf_ci = fqe.Wavefunction([[sum(nelec), nelec[0] - nelec[1], norbs]])
    fqe_data_ci = fqe_wf_ci.sector((sum(nelec), nelec[0] - nelec[1]))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()))
    for paidx, pyscf_alpha_idx in enumerate(n_alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(n_beta_strings):
            fqe_orderd_coeff[fqe_graph_ci.index_alpha(
                pyscf_alpha_idx), fqe_graph_ci.index_beta(pyscf_beta_idx)] = \
                pyscf_cimat[paidx, pbidx]

    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci

def test_adapt_s2():
    """Penalty parameter for adapt s^2"""
    import copy
    from fqe.openfermion_utils import integrals_to_fqe_restricted
    molecule = get_molecule()# build_lih_moleculardata()

    mf = molecule._pyscf_data['scf']
    mol = molecule._pyscf_data['mol']

    from pyscf import fci
    nelec = mol.nelec
    mci = fci.FCI(mol, mf.mo_coeff)
    mci = fci.addons.fix_spin(mci, shift=1, ss=0)
    mci.nroots = 8
    # Use keyword argument nelec to explicitly control the spin. Otherwise
    # mol.spin is applied.
    e, civec = mci.kernel(nelec=nelec)
    pyscf_fci_e = e
    print(' E = %.12f S^2 = %.7f  2S+1 = %.7f' %
          (e[0], mci.spin_square(civec[0], mf.mo_coeff.shape[1], nelec)[0],
           mci.spin_square(civec[0], mf.mo_coeff.shape[1], nelec)[1]
           ))
    print("E nonuc", e - molecule.nuclear_repulsion)

    fqe_fci_wf = pyscf_to_fqe_wf(civec[0], mf)


    print(molecule.fci_energy - molecule.nuclear_repulsion)
    n_electrons = molecule.n_electrons
    oei, tei = molecule.get_integrals()
    elec_ham = integrals_to_fqe_restricted(oei, tei)
    norbs = molecule.n_orbitals
    nalpha = molecule.n_electrons // 2
    nbeta = nalpha
    sz = nalpha - nbeta
    occ = list(range(nalpha))
    virt = list(range(nalpha, norbs))
    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    molecular_hamiltonian = of.InteractionOperator(0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(molecular_hamiltonian,
                                                   molecule.n_electrons)

    np.random.seed(10)
    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    # fqe_wf.set_wfn(strategy='random')
    fqe_wf.set_wfn(strategy='hartree-fock')
    fqe_wf.sector((n_electrons, sz)).coeff = fqe_wf.sector((n_electrons, sz)).coeff.real
    fqe_wf.normalize()

    sop = OperatorPool(norbs, occ, virt)
    sop.two_body_sz_adapted()  # initialize pool
    # sop.one_body_sz_adapted()
    # sop.generalized_two_body()
    adapt = ADAPT(oei, tei, sop, nalpha, nbeta, iter_max=50, verbose=True,
                  stopping_epsilon=1.0E-4, delta_e_eps=-1)
    print(fqe_fci_wf.expectationValue(adapt.k2_fop).real)
    print(fqe_fci_wf.expectationValue(elec_ham).real)
    fqe_fci_wf.print_wfn()
    # adapt.adapt_vqe(fqe_fci_wf, opt_method='BFGS', v_reconstruct=False)


    adapt.adapt_vqe(fqe_wf, opt_method='BFGS', v_reconstruct=False,
                    shift=10, ss=0)





if __name__ == "__main__":
    test_adapt_s2()