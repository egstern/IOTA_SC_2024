#!/usr/bin/env python

from math import pi
import synergia_workflow

opts = synergia_workflow.Options("iota_ac")

opts.add("seed", 12345791, "Pseudorandom number generator seed", int)

opts.add("kinetic_energy", 0.00250, "Beam kinetic energy [GeV]")
opts.add("aperture_radius", 0.050/2, "Radius of beam pipe aperture [m]")
opts.add("harmonic_number", 4, "Harmonic number of RF cavity")
opts.add("matching", "6dmoments", "matching procedure 6dmoments|uniform|zero")

opts.add("emitx", 4.3e-6, "unnormalized x RMS emittance [m-rad]")
opts.add("emity", 3.0e-6, "unnormalized y RMS emittance [m-rad]")
opts.add("std_dpop", 2.1e-3, "RMS dp/p spread")
opts.add("dpop_cutoff", 2.5, "Cutoff on dp/p in sigma")
opts.add("transverse_cutoff", 3.0, "Cutoff on transverse distributions in sigma")

opts.add("std_bunchlen", 1.23574, "RMS bunch length [m]")

opts.add("num_bunches", 1, "number of bunches in bunch train")
opts.add("macroparticles", 1048576, "number of macro particles")
opts.add("spectparticles", 96, "number of spectator particles")
opts.add("current", 0.1, 'beam current [mA]')
opts.add("periodic", True, "make bunch periodic boundary conditions")
opts.add("turns", 10, "number of turns")

opts.add("collective", "2d-openhockney", "space charge [off|2d-openhockney|2d-bassetti-erskine|3d-openhockney", str)
opts.add("gridx", 32, "x grid size")
opts.add("gridy", 32, "y grid size")
opts.add("gridz", 128, "z grid size")
opts.add("comm_group_size", 1, "Communication group size for space charge solvers (must be 1 on GPUs), probably 16 on CPU", int)

opts.add("stepper", "elements", "which stepper to use independent|elements|splitoperator")
opts.add("steps", 3, "# steps")

opts.add("disable_rf", False, "Turn off RF cavities before propagation")
opts.add("monochrome", False, "Set all dp/p to 0")

opts.add("test_particles", False, "place test particles in the bunch")

opts.add("step_basic", False, "Basic diagnostics each step")
opts.add("tracks", 100, "number of particles to track")
opts.add("save_particles", False, "if True, save particles")
opts.add("particles_period", 1, "n!=0, save particles every n turns")
opts.add("step_tracks", 0, "number of tracks to save/step")
opts.add("step_diag", False, "whether to do diagnostics/step")

opts.add("longitudinal_boundary", "open", '{open|periodic|aperture|bucket_barrier}')

opts.add('split_tunes', False, "split the tunes of the planes")
opts.add('split_tunes_factor', 0.005, 'factor to multiply quads by for tune splitting')

job_mgr = synergia_workflow.Job_manager("iota_sc.py", opts, ["machine.seq", "print_mapping.py"])
