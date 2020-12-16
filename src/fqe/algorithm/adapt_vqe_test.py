import pytest
import fqe
from fqe.algorithm.adapt_vqe import OperatorPool, ADAPT
from fqe.unittest_data.generate_openfermion_molecule import (
    build_lih_moleculardata,
)
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial
import numpy as np


def test_op_pool():
    op = OperatorPool(2, [0], [1])
    op.singlet_t2()
    true_generator = of.FermionOperator(((1, 1), (0, 1), (3, 0), (2, 0))) - \
                     of.FermionOperator(((3, 1), (2, 1), (1, 0), (0, 0)))

    assert np.isclose(
        of.normal_ordered(op.op_pool[0] - true_generator).induced_norm(), 0)

    op = OperatorPool(2, [0], [1])
    op.two_body_sz_adapted()
    assert len(op.op_pool) == 20

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
        for ladder_idx, coeff in gen.terms.items():
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
    molecular_hamiltonian = of.InteractionOperator(
        0, soei, 0.25 * astei)
    reduced_ham = of.chem.make_reduced_hamiltonian(
        molecular_hamiltonian, molecule.n_electrons
    )

    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')

    sop = OperatorPool(norbs, occ, virt)
    sop.two_body_sz_adapted()  # initialize pool
    adapt = ADAPT(oei, tei, sop, nalpha, nbeta, iter_max=1, verbose=False)
    assert np.isclose(np.linalg.norm(
        adapt.reduced_ham.two_body_tensor - reduced_ham.two_body_tensor), 0)

    adapt.adapt_vqe(fqe_wf)
    assert np.isclose(adapt.energies[0], -8.970532463968661)

    sop = OperatorPool(norbs, occ, virt)
    sop.one_body_sz_adapted()
    adapt = ADAPT(oei, tei, sop, nalpha, nbeta, iter_max=10,
                  stopping_epsilon=10, verbose=True)
    adapt.adapt_vqe(fqe_wf)
    assert np.isclose(adapt.energies[0], -8.95741717733075)


def test_vbc():
    molecule = build_lih_moleculardata()
    n_electrons = molecule.n_electrons
    oei, tei = molecule.get_integrals()
    norbs = molecule.n_orbitals
    nalpha = molecule.n_electrons // 2
    nbeta = nalpha
    sz = nalpha - nbeta
    occ = list(range(nalpha))
    virt = list(range(nalpha, norbs))

    fqe_wf = fqe.Wavefunction([[n_electrons, sz, molecule.n_orbitals]])
    fqe_wf.set_wfn(strategy='hartree-fock')

    sop = OperatorPool(norbs, occ, virt)
    sop.two_body_sz_adapted()  # initialize pool
    adapt = ADAPT(oei, tei, sop, nalpha, nbeta, iter_max=1, verbose=False)
    adapt.vbc(fqe_wf)
    assert np.isclose(adapt.energies[0], -8.97304439380826)

    with pytest.raises(ValueError):
        adapt.vbc(fqe_wf, 3)