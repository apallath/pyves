"""
Convenience classes for performing langevin dynamics simulations using OpenMM.
"""
import multiprocessing
import pickle

import numpy as np
from simtk import unit
from simtk.openmm import openmm
from tqdm import tqdm


class SingleParticleSimulation:
    """
    Performs langevin dynamics simulation of a particle on a 2D potential energy surface.
    """
    def __init__(self,
                 potential: openmm.CustomExternalForce,
                 mass: int = 1,
                 temp: float = 300,
                 friction: float = 100,
                 timestep: float = 10,
                 init_state: openmm.State = None,
                 gpu: bool = False):
        # Properties
        self.mass = mass * unit.dalton  # mass of particles
        self.temp = temp * unit.kelvin  # temperature
        self.friction = friction / unit.picosecond  # LD friction factor
        self.timestep = timestep * unit.femtosecond   # LD timestep

        self.init_state = init_state
        self.gpu = gpu

        # Number of particles
        # Fixed
        n = 1

        # Init simulation objects
        self.system = openmm.System()
        self.potential = potential
        for i in range(n):
            self.system.addParticle(self.mass)
            self.potential.addParticle(i, [])  # no parameters associated with each particle
        self.system.addForce(potential)

        self.integrator = openmm.LangevinIntegrator(self.temp, self.friction, self.timestep)

        if self.gpu:
            platform = openmm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
        else:
            platform = openmm.Platform.getPlatformByName('CPU')
            num_threads = str(multiprocessing.cpu_count())
            properties = {'Threads': num_threads}

        self.context = openmm.Context(self.system, self.integrator, platform, properties)

        # Init state
        if init_state is None:
            init_coord = (np.random.rand(n, 3) * np.array([12, 12, 0]) + np.array([-6, -6, 0]))
            self.context.setPositions(init_coord)
            self.context.setVelocitiesToTemperature(self.temp)
        else:
            self.context.setState(init_state)

    def __call__(self,
                 nsteps: int = 1000,
                 chkevery: int = 500,
                 trajevery: int = 1,
                 energyevery: int = 1,
                 chkfile="./chk_state.pkl",
                 trajfile="./traj.dat",
                 energyfile="./energies.dat"):
        # Data
        self.traj = None
        self.PE = []
        self.KE = []

        for i in tqdm(range(nsteps)):
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
