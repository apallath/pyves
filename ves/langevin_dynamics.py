"""
Classes for performing langevin dynamics simulations with VES bias
using OpenMM.
"""
import multiprocessing
import pickle

import numpy as np
from openmm import unit
from openmm import openmm
from tqdm import tqdm

from ves.bias import Bias


class SingleParticleSimulation:
    """
    Performs langevin dynamics simulation of a particle on a 2D potential energy surface
    with an optional VES bias.
    """
    def __init__(self,
                 potential: openmm.CustomExternalForce,
                 mass: int = 1,
                 temp: float = 300,
                 friction: float = 100,
                 timestep: float = 10,
                 init_state: openmm.State = None,
                 init_coord: np.ndarray = np.array([0, 0, 0]).reshape((1, 3)),
                 gpu: bool = False,
                 seed=None):
        # Properties
        self.mass = mass * unit.dalton  # mass of particles
        self.temp = temp * unit.kelvin  # temperature
        self.friction = friction / unit.picosecond  # LD friction factor
        self.timestep = timestep * unit.femtosecond   # LD timestep

        self.init_state = init_state
        self.gpu = gpu

        # Init simulation objects
        self.system = openmm.System()
        self.potential = potential
        self.system.addParticle(self.mass)
        self.potential.addParticle(0, [])  # no parameters associated with each particle
        self.system.addForce(potential)

        self.integrator = openmm.LangevinIntegrator(self.temp,
                                                    self.friction,
                                                    self.timestep)
        if seed is not None:
            self.integrator.setRandomNumberSeed(seed)

        if self.gpu:
            platform = openmm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            print("Running simulation on GPU.")
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
            num_threads = str(multiprocessing.cpu_count())
            properties = {'Threads': num_threads}
            print("Running simulation on {} CPU threads.".format(num_threads))

        self.context = openmm.Context(self.system, self.integrator, platform, properties)

        # Init state
        if init_state is None:
            self.context.setPositions(init_coord)
            if seed is not None:
                self.context.setVelocitiesToTemperature(self.temp, randomSeed=seed)
            else:
                self.context.setVelocitiesToTemperature(self.temp)
        else:
            self.context.setState(init_state)

        # By default, the simulation is not biased
        self.biased = False

    def init_ves(self,
                 bias: Bias,
                 static: bool = False,
                 startafter: int = 2,
                 learnevery: int = 50):
        """
        Initialize simulation with VES bias.
        """
        self.biased = True

        if static:
            self.static = True
            self.startafter = None
            self.learnevery = None
        else:
            self.static = False
            if startafter < 2:
                raise ValueError("Cannot start VES updates before timestep 2.")
            self.startafter = startafter
            self.learnevery = learnevery

        self.bias = bias
        self.update_on = False

    def __call__(self,
                 nsteps: int = 1000,
                 chkevery: int = 500,
                 trajevery: int = 1,
                 energyevery: int = 1,
                 chkfile="./chk_state.pkl",
                 trajfile="./traj.dat",
                 energyfile="./energies.dat",
                 ves_update_params={}):
        # Data
        self.traj = None
        self.PE = []
        self.KE = []

        for i in tqdm(range(nsteps)):
            ####################################################################
            # Begin VES mod
            #
            # OpenMM-related notes
            # ** During update steps, previous force needs to be removed by
            #    index and a new one needs to be added
            # ** The simulation Context needs to be re-initialized after
            #    modifying the system (i.e. adding or removing forces)
            #
            ####################################################################
            if self.biased:
                if not self.static:
                    # Repeat updates
                    if i == self.startafter:
                        # I/O
                        print("[VES] {} bias: Initializing dynamic bias at timestep {}.".format(type(self.bias).__name__, i))
                        # Update bias
                        self.bias.update(self.traj, **ves_update_params)
                        # Add bias force, and store index
                        self.bias_force_idx = self.system.addForce(self.bias.force)
                        # Re-initialize context
                        self.context.reinitialize(preserveState=True)

                    if i > self.startafter and i % self.learnevery == 0:
                        # Do not repeat update @ startevery (=> elif)

                        # I/O
                        print("[VES] {} bias: Updating dynamic bias at timestep {}.".format(type(self.bias).__name__, i))
                        # Update bias
                        self.bias.update(self.traj, **ves_update_params)
                        # Remove existing bias force
                        self.system.removeForce(self.bias_force_idx)
                        # Add updated bias force, and store index
                        self.bias_force_idx = self.system.addForce(self.bias.force)
                        # Re-initialize context
                        self.context.reinitialize(preserveState=True)

                else:
                    # Do this only at the first timestep
                    if i == 0:
                        # I/O
                        print("[VES] {} bias: Initializing static bias.".format(type(self.bias).__name__))
                        # Add force, and store index
                        self.bias_force_idx = self.system.addForce(self.bias.force)
                        # Re-initialize context
                        self.context.reinitialize(preserveState=True)

            ####################################################################
            # End VES mod
            ####################################################################

            # Checkpoint
            if i % chkevery == 0:
                self.dump_state(chkfile)

            # Store positions
            if i % trajevery == 0:
                pos = self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                if i == 0:
                    self.traj = pos
                else:
                    self.traj = np.vstack((self.traj, pos))

            # Store energy
            if i % energyevery == 0:
                PE = self.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
                self.PE.append(PE)
                KE = self.context.getState(getEnergy=True).getKineticEnergy() / unit.kilojoule_per_mole
                self.KE.append(KE)

            # Step
            self.integrator.step(1)

        ####################################
        # Final state
        ####################################

        # Checkpoint
        self.dump_state(chkfile)

        # Store positions
        pos = self.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        self.traj = np.vstack((self.traj, pos))

        # Store energy
        PE = self.context.getState(getEnergy=True).getPotentialEnergy() / unit.kilojoule_per_mole
        self.PE.append(PE)
        KE = self.context.getState(getEnergy=True).getKineticEnergy() / unit.kilojoule_per_mole
        self.KE.append(KE)

        ####################################
        # Write traj and energies
        ####################################

        self.write_trajectory(trajfile)
        self.write_energies(energyfile)

    def write_trajectory(self, ofilename):
        np.savetxt(ofilename, self.traj, header='x [nm]    y [nm]    z [nm]')

    def write_energies(self, ofilename):
        energy_array = np.vstack((np.array(self.PE), np.array(self.KE))).T
        np.savetxt(ofilename, energy_array, header='PE [kJ/mol]    KE [kJ/mol]')

    def dump_state(self, ofilename):
        state = self.context.getState(getPositions=True, getVelocities=True)
        with open(ofilename, "wb") as fh:
            pickle.dump(state, fh)
