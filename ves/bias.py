"""
Classes defining bias potentials and update mechanisms.
"""
from openmmtorch import TorchForce
import torch
import torch.optim as optim


torch.set_default_tensor_type(torch.DoubleTensor)

################################################################################
# Analytical biases
################################################################################


class Bias:
    """
    Abstract class defining biases that can be added to langevin dynamics simulations.
    """
    def __init__(self):
        pass

    def update(self, traj):
        pass

    @property
    def force(self):
        pass


class LegendreBias_SingleParticle_x(Bias):
    """
    Applies a legendre potential bias along the x coordinate of a single particle.
    """
    pass


################################################################################
# Neural network biases
################################################################################


class NNBias(Bias):
    """
    Abstract class defining neural network biases.
    """
    def __init__(self, model_loc="model.pt"):
        super().__init__()
        self.model_loc = model_loc

    @property
    def force(self):
        torch_force = TorchForce(self.model_loc)
        return torch_force


class HarmonicBias_SingleParticle_x(NNBias):
    """
    Applies a harmonic bias potential along the x coordinate of a single particle.
    """
    def __init__(self, k, x0, model_loc="model.pt"):
        super().__init__(model_loc)
        module = torch.jit.script(HarmonicBias_SingleParticle_x_ForceModule(k, x0))
        module.save(self.model_loc)


class HarmonicBias_SingleParticle_x_ForceModule(torch.nn.Module):
    """
    A harmonic potential k/2 (x-x0)^2 as a static compute graph.

    The potential is only applied to the x-coordinate of the particle.
    """
    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0
        super().__init__()

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Args:
            positions : torch.Tensor with shape (1, 3)
                positions[0, k] is the position (in nanometers) of spatial dimension k of particle 0

        Returns:
            potential : torch.Scalar
                The potential energy (in kJ/mol)
        """
        return self.k / 2 * torch.sum((positions[:, 0] - self.x0) ** 2)


class VESBias_SingleParticle_x(NNBias):
    """
    Applies a basis set expanded potential along the x coordinate of a single particle.

    The potential (V_module) can either be:
        a) an expansion over a basis set (Valsson and Parrinello 2014).
        b) a neural network (Valsson and Parrinello 2014).

    Note:
        Some basis sets (such as the Legendre basis set) can be slow to work with due to the large
        overhead associated with backpropagation for gradient computation.
    """
    def __init__(self, V_module, x_min, x_max, target, beta, optimizer_type, optimizer_params, model_loc):
        # Bias potential V(x)
        self.model = V_module

        # Scaling parameters
        self.x_min = x_min
        self.x_max = x_max

        # Target distribution p(x)
        self.target = target

        # Required for computing loss function
        self.beta = beta

        if optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_params)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), **optimizer_params)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError("Requested optimizer not yet supported.")

        super().__init__(model_loc)

    def update(self, traj):
        ########################################################################
        # Training loop with one optimizer step
        ########################################################################

        # Scale traj
        traj = traj / (self.x_max - self.x_min) + self.x_min

        # Zero gradients
        self.optimizer.zero_grad()

        # Evaluate biased part of loss
        # Accumulate loss over entire trajectory
        # TODO: Add support for different averaging methods
        eBV = torch.tensor([[0.0]], requires_grad=True)
        eBV_sum = eBV.clone()
        for t in range(traj.shape[0]):
            xyz = torch.tensor(traj[t]).reshape((1, 3))
            eBV_sum += torch.exp(self.beta * self.model(xyz))
        loss_V = 1 / self.beta * torch.log(eBV_sum)

        # Evaluate target part of loss
        # Accumulate loss over target x
        loss_p = torch.tensor([[0.0]], requires_grad=True)
        loss_p_sum = loss_p.clone()

        x_target = torch.from_numpy(self.target.x)
        p_target = torch.from_numpy(self.target.p)

        for i in range(len(x_target)):
            loss_p_sum += torch.tensor([p_target[i]]) * self.model(x_target[i].reshape((1, 1)))

        # Sum loss
        loss = loss_V + loss_p_sum

        # Print loss
        print("Loss = {:.6f}".format(loss.detach().numpy()[0, 0]))

        # Backprop to compute gradients
        loss.backward()

        # Make an optimizer step
        self.optimizer.step()

        # Save updated model as torchscript
        # Overwrite last model
        module = torch.jit.script(self.model)
        module.save(self.model_loc)

        # Archive model for future retrieval
        module.save(self.model_loc + ".iter{}".format(traj.shape[0]))
