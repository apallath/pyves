"""
Miscellanious utility classes and functions.
"""
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# Trajectory reader
################################################################################


class TrajectoryReader:
    """Utility class for reading large trajectories.

    Args:
        traj_file (str): Path to trajectory file.
        comment_char (str): Character marking the beginning of a comment line.
        format (str): Format of each line (options = 'txyz' or 'xyz'; default = 'txyz')
    """
    def __init__(self, traj_file, comment_char='#', format='txyz'):
        self.traj_file = traj_file
        self.comment_char = comment_char
        self.format = format

    def read_traj(self, skip=1):
        """
        Reads trajectory.

        Args:
            skip (int): Number of frames to skip between reads (default = 1).

        Returns:
            tuple(T, traj) if self.format == 'txyz'
            traj if self.format == 'xyz'
        """
        if self.format == 'txyz':
            return self._read_traj_txyz(skip)
        elif self.format == 'xyz':
            return self._read_traj_xyz(skip)
        else:
            raise ValueError('Invalid format {}'.format(format))

    def _read_traj_txyz(self, skip):
        times = []
        traj = []

        count = 0
        with open(self.traj_file, 'r') as trajf:
            for line in trajf:
                if line.strip()[0] != self.comment_char:
                    if count % skip == 0:
                        txyz = [float(c) for c in line.strip().split()]
                        times.append(txyz[0])
                        traj.append([txyz[1], txyz[2], txyz[3]])

        return np.array(times), np.array(traj)

    def _read_traj_xyz(self, skip):
        traj = []

        count = 0
        with open(self.traj_file, 'r') as trajf:
            for line in trajf:
                if line.strip()[0] != self.comment_char:
                    if count % skip == 0:
                        xyz = [float(c) for c in line.strip().split()]
                        traj.append([xyz[0], xyz[1], xyz[2]])

        return np.array(traj)
