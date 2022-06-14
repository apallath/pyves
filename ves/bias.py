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
        # Biases are static by default.
        # Implementing the update method is optional.
        raise NotImplementedError("This bias cannot be updated.")

    @property
    def force(self):
        pass


################################################################################
# Neural network biases
################################################################################


class NNBias(Bias):
    """
    Abstract class defining neural network biases.

    Child classes must set the self.model parameter to a TorchScript compatible
    neural network instance before calling super().__init__().
    """
    def __init__(self, model_loc="model.pt"):
        self.model_loc = model_loc

        # self.model parameter needs to be set by the child class.
        module = torch.jit.script(self.model)
        module.save(self.model_loc)
        super().__init__()

    @property
    def force(self):
        torch_force = TorchForce(self.model_loc)
        return torch_force


class HarmonicBias_SingleParticle_x(NNBias):
    """
    Applies a harmonic bias potential along the x coordinate of a single particle.
    """
    def __init__(self, k, x0, model_loc="model.pt"):
        self.model = HarmonicBias_SingleParticle_x_ForceModule(k, x0)
        super().__init__(model_loc)


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


class StaticBias_SingleParticle_x(NNBias):
    """
    Applies a **static** basis set expanded potential along the x coordinate of a single particle.

    The potential (V_module) can either be:
        a) an expansion over a basis set (Valsson and Parrinello 2014).
        b) a neural network (Valsson and Parrinello 2014).

    Note:
        Some basis sets (such as the Legendre basis set) can be slow to work with due to the large
        overhead associated with backpropagation for gradient computation.
    """
    def __init__(self, V_module, model_loc):
        # Bias potential V(x)
        self.model = V_module
        super().__init__(model_loc)


class VESBias_SingleParticle_x(NNBias):
    """
    Applies a **dynamic** basis set expanded potential along the x coordinate of a single particle.
    The parameters of the basis set can be updated by calling the `update` function with
    a trajectory.

    The potential (V_module) can either be:
        a) an expansion over a basis set (Valsson and Parrinello 2014).
        b) a neural network (Valsson and Parrinello 2014).

    Note:
        Some basis sets (such as the Legendre basis set) can be slow to work with due to the large
        overhead associated with backpropagation for gradient computation.
    """
    def __init__(self, V_module, target, beta, optimizer_type, optimizer_params,
                 model_loc, update_steps=5000):
        # Bias potential V(x)
        self.model = V_module

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

        self.update_steps = update_steps

        super().__init__(model_loc)

    def update(self, traj):
        ########################################################################
        # Training loop with one optimizer step
        ########################################################################

        # Zero gradients
        self.optimizer.zero_grad()

        # Evaluate biased part of loss
        # Accumulate loss over last X steps
        eBV = torch.tensor([[0.0]], requires_grad=True)
        eBV_sum = eBV.clone()
        print(max(traj.shape[0] - self.update_steps, 0), traj.shape[0])
        for t in range(max(traj.shape[0] - self.update_steps, 0), traj.shape[0]):
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
