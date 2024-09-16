#!/usr/bin/env python
import sys
import os
import numpy as np
from mpi4py import MPI
import synergia
import synergia.simulation as SIM
ET = synergia.lattice.element_type
MT = synergia.lattice.marker_type
PCONST = synergia.foundation.pconstants

#####################################

from iota_sc_options import opts

#######################################################

DEBUG=False

logger = synergia.utils.Logger(0)

#######################################################


def print_statistics(bunch, fout=sys.stdout):

    parts = bunch.get_particles_numpy()
    print(parts.shape,  ", ", parts.size , file=fout)
    print("shape: {0}, {1}".format(parts.shape[0], parts.shape[1]))

    mean = synergia.bunch.Core_diagnostics.calculate_mean(bunch)
    std = synergia.bunch.Core_diagnostics.calculate_std(bunch, mean)
    print("mean = {}".format(mean), file=fout)
    print("std = {}".format(std), file=fout)

#######################################################
#######################################################

def save_json_lattice(lattice, jlfile='iota_lattice.json'):
    # save the lattice
    if MPI.COMM_WORLD.rank == 0:
        f = open(jlfile, 'w')
        print(lattice.as_json(), file=f)
        f.close()

################################################################################

def get_lattice():
    # read the lattice in from a MadX sequence file
    lattice = synergia.lattice.MadX_reader().get_lattice("iota", "machine.seq")

    # The sequence doesn't have a reference particle so define it here
    KE = opts.kinetic_energy
    mass = PCONST.mp

    etot = KE + mass
    refpart = synergia.foundation.Reference_particle(1, mass, etot)

    lattice.set_reference_particle(refpart)
    # Change the tune of one plane to break the coupling resonance
    # for elem in lattice.get_elements():
    #     if elem.get_type() == ET.quadrupole:
    #         k1 = elem.get_double_attribute('k1')
    #         elem.set_double_attribute('k1', k1*0.99)
    #         break

    return lattice

################################################################################

# set a circular aperture on all elements

def set_apertures(lattice):

    radius = opts.aperture_radius
    for elem in lattice.get_elements():
        elem.set_string_attribute("aperture_type", "circular")
        elem.set_double_attribute("circular_aperture_radius", radius)

    return lattice

################################################################################


def get_iota_twiss(lattice):
    synergia.simulation.Lattice_simulator.CourantSnyderLatticeFunctions(lattice)
    synergia.simulation.Lattice_simulator.calc_dispersions(lattice)
    elements = lattice.get_elements()
    return elements[-1].lf

################################################################################

def create_bunch_simulator(refpart, num_particles, real_particles):
    commxx = synergia.utils.Commxx()
    sim = synergia.simulation.Bunch_simulator.create_single_bunch_simulator(
        #refpart, num_particles, real_particles, commxx)
        refpart, num_particles, real_particles, commxx, 101)

    return sim

################################################################################

# determine the bunch charge from beam current
def beam_current_to_numpart(current, length, beta, harmonic):
    rev_time = length/(beta*PCONST.c)
    total_charge = current*rev_time/PCONST.e
    # charge is divided into bunches by harmonoic number
    return total_charge/harmonic
    

################################################################################

# populate a matched distribution using emittances, dispersions and the lattice
def populate_matched_distribution(lattice, bunch, emitx, betax, Dx, emity, betay, stddpop):
    # construct standard deviations
    stdx = np.sqrt(emitx*betax + stddpop**2*Dx**2)
    stdy = np.sqrt(emity*betay)
    beta = lattice.get_reference_particle().get_beta()
    print('stdx: ', stdx, file=logger)
    print('stdy: ', stdy, file=logger)
    print('stdcdt: ', stddpop, file=logger)

    map = synergia.simulation.Lattice_simulator.get_linear_one_turn_map(lattice)
    corr_matrix =  synergia.bunch.get_correlation_matrix(map,
                                                              stdx,
                                                              stdy,
                                                              stddpop,
                                                              beta,
                                                              (0, 2, 5))

    dist = synergia.foundation.PCG_random_distribution(1234567, synergia.utils.Commxx.World.rank())

    means = np.zeros(6)
    tc = opts.transverse_cutoff
    lc = opts.dpop_cutoff
    limits = np.array([tc, tc, tc, tc, lc, lc])
    synergia.bunch.populate_6d_truncated(dist, bunch, means,
                                         corr_matrix, limits)
    # localnum = bunch.get_local_num()
    # lp = bunch.get_particles_numpy()
    # lp[0:localnum, 0:6] = 0.0
    # bunch.checkin_particles()

    # populate spectator particles
    #spart = bunch.get_particles_numpy(synergia.bunch.ParticleGroup.spectator)
    #for ixspect in range(51):
    #    spart[ixspect, 0:6] = 0.0
    #    spart[ixspect, 0] = ixspect * stdx/10
    #for iyspect in range(51,101):
    #    spart[iyspect, 0:6] = 0.0
    #    spart[iyspect, 2] = (iyspect-50) * stdy/10


################################################################################


def register_diagnostics(sim):
    # diagnostics
    diag_full2 = synergia.bunch.Diagnostics_full2("diag.h5")
    sim.reg_diag_per_turn(diag_full2)

    diag_bt = synergia.bunch.Diagnostics_bulk_track("tracks.h5", opts.tracks, 0)
    sim.reg_diag_per_turn(diag_bt)

    #diag_spect = synergia.bunch.Diagnostics_bulk_track('stracks.h5', 101, 0, synergia.bunch.ParticleGroup.spectator)

    if opts.save_particles and opts.particles_period > 0:
        diag_part = synergia.bunch.Diagnostics_particles("particles.h5")
        sim.reg_diag_per_turn(diag_part, opts.particles_period)

    if opts.step_tracks:
        diag_step_track = synergia.bunch.Diagnostics_bulk_track("step_tracks.h5", opts.step_tracks, 0)
        sim.reg_diag_per_step(diag_step_track)
    
    if opts.step_diag:
        diag_stp = synergia.bunch.Diagnostics_full2("diag_step.h5")
        sim.reg_diag_per_step(diag_stp)


################################################################################

def get_propagator(lattice):
    if  DEBUG:
        print('get_propagator operating on lattice: ', id(lattice), file=logger)

    steps = opts.steps

    if not opts.collective:
        sc_ops = synergia.collective.Dummy_CO_options()

    elif opts.collective == "2d-openhockney":
        sc_ops = synergia.collective.Space_charge_2d_open_hockney_options(opts.gridx, opts.gridy, opts.gridz)
        sc_opts.comm_group_size = opts.comm_group_size
        print(f"using 2d-openhockney collective operator with grid size {opts.gridx}, {opts.gridy}, {opts.gridz}", file=logger)

    else:
        # unknown collective operator not defined
        raise RuntimeError(f'unhandled collective operator specified: {opts.collective}')

    if opts.stepper == "independent":
        # independent stepper is incompatible with a collective operation
        if opts.collective:
            raise RuntimeError("error, may not specify collective operator with independent stepper")
        else:
            stepper = synergia.simulation.Independent_stepper_elements(opts.steps)
            print("using independent stepper elements with steps: ", opts.steps, file=logger)

    elif opts.stepper == "elements":
        if opts.collective:
            stepper = synergia.simulation.Split_operator_stepper_elements(sc_ops, steps)

        else:
            stepper = synergia.simulation.Independent_stepper_elements(1)

    elif opts.stepper == "splitoperator":
        stepper = synergia.simulation.Split_operator_stepper(sc_ops, steps)


    # if opts.impedance:
    #     imp_coll_op = get_impedance_op(lattice.get_length())
    #     stepper.append_collective_op(imp_coll_op)

    propagator = synergia.simulation.Propagator(lattice, stepper)

    if DEBUG:
        print('lattice from propagator: ', id(propagator.get_lattice()), file=logger)

    return propagator

################################################################################

# turn off RF cavities by setting their voltage to 0
def disable_rfcavities(lattice):
    for elem in lattice.get_elements():
        if elem.get_type() == ET.rfcavity:
            elem.set_double_attribute('volt', 0.0)


################################################################################

def main():

    lattice = get_lattice()
    print('Read lattice, length = {}, {} elements'.format(lattice.get_length(), len(lattice.get_elements())), file=logger)
    lattice_length = lattice.get_length()
    bucket_length = lattice_length/opts.harmonic_number

    print('RF cavity frequency should be: ', opts.harmonic_number*lattice.get_reference_particle().get_beta() * PCONST.c/lattice_length)

    # set the aperture early
    set_apertures(lattice)

    state = SIM.Lattice_simulator.tune_circular_lattice(lattice)
    print('state: ', state)
    print('length maybe: ', state[4]*lattice.get_reference_particle().get_beta())

    for elem in lattice.get_elements():
        if elem.get_type() == ET.rfcavity:
            print('RF cavity: ', elem, file=logger)

    refpart = lattice.get_reference_particle()

    energy = refpart.get_total_energy()
    momentum = refpart.get_momentum()
    gamma = refpart.get_gamma()
    beta = refpart.get_beta()

    print("energy: ", energy, file=logger)
    print("momentum: ", momentum, file=logger)
    print("gamma: ", gamma, file=logger)
    print("beta: ", beta, file=logger)

    save_json_lattice(lattice, 'iota_lattice.json')

    iota_twiss = get_iota_twiss(lattice)
    lf = iota_twiss
    print('IOTA lattice Twiss parameters:', file=logger)
    print(f'beta x: {lf.beta.hor}, alpha x: {lf.alpha.hor}, x tune: {lf.psi.hor/(2*np.pi)}', file=logger)
    print(f'disp x: {lf.dispersion.hor}, dprime x: {lf.dPrime.hor}', file=logger)
    print(f'beta y: {lf.beta.ver}, alpha y: {lf.alpha.ver}, y tune: {lf.psi.ver/(2*np.pi)}', file=logger)
    print(f'disp y: {lf.dispersion.ver}, dprime y: {lf.dPrime.ver}', file=logger)
    
    map = SIM.Lattice_simulator.get_linear_one_turn_map(lattice)
    l, v = np.linalg.eig(map)
    print("eigenvalues: ", file=logger)
    for z in l:
        print("|z|: ", abs(z), " z: ", z, " tune: ", np.log(z).imag/(2.0*np.pi), file=logger)

    longitudinal_map = map[4:6, 4:6]
    longitudinal_map[0,1] = -longitudinal_map[0,1]
    longitudinal_map[1,0] = -longitudinal_map[1,0]
    lm = longitudinal_map
    print('determinant: ', lm[0,0]*lm[1,1]-lm[0,1]*lm[1,0], file=logger)
    print('cos(mu): ', 0.5*(lm[0, 0]+lm[1, 1]), file=logger)
    print('longitudinal map:\n', np.array2string(longitudinal_map, separator=','), file=logger)
    a, b, mus = SIM.Lattice_simulator.map_to_twiss(longitudinal_map)
    print(f'longitudinal beta: {b}, Qs: {mus/(2*np.pi)}', file=logger)

    bunch_charge = beam_current_to_numpart(opts.current, lattice_length, beta, opts.harmonic_number)
    bunch_sim = create_bunch_simulator(refpart, opts.macroparticles, bunch_charge)

    # set longitudinal
    bdy = getattr(synergia.bunch.LongitudinalBoundary, opts.longitudinal_boundary)
    bunch_sim.get_bunch().set_longitudinal_boundary(bdy, bucket_length)
    lb, ll = bunch_sim.get_bunch(0,0).get_longitudinal_boundary()
    print(f'Setting longitudinal boundary conditions: {lb}, size: {ll}')
                                                    
    print('beam current: ', opts.current, ' mA', file=logger)
    print('bunch created with ', opts.macroparticles, ' macroparticles', file=logger)
    print('bunch charge: ', bunch_charge, file=logger)

    #### generate bunch
    populate_matched_distribution(lattice, bunch_sim.get_bunch(0,0),
                                  opts.emitx, lf.beta.hor, lf.dispersion.hor,
                                  opts.emity, lf.beta.ver,
                                  opts.std_dpop)

    print_statistics(bunch_sim.get_bunch(0, 0), logger)


    register_diagnostics(bunch_sim)


    ####  stepper and collective operators

    if opts.disable_rf:
        disable_rfcavities(lattice)

    propagator = get_propagator(lattice)

    # logger for simulation
    simlog = synergia.utils.parallel_utils.Logger(0, 
            synergia.utils.parallel_utils.LoggerV.INFO_TURN)
            #synergia.utils.parallel_utils.LoggerV.INFO)
            #synergia.utils.parallel_utils.LoggerV.INFO_STEP)

    # if opts.disable_rf:
    #     disable_rfcavities(propagator.get_lattice())

    propagator.propagate(bunch_sim, simlog, opts.turns)



if __name__ == "__main__":

    main()
