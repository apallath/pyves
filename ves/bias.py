"""
Classes defining bias potentials and update mechanisms.
"""
from openmmtorch import TorchForce
import torch
import torch.optim as optim


torch.set_default_tensor_type(torch.DoubleTensor)

################################################################################
# Base classes for biases
################################################################################


class Bias:
    """
    Base class defining biases.
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


class NNBias(Bias):
    """
    Base class defining neural network biases.

    Child classes must set the self.model parameter to a TorchScript compatible
    neural network instance before calling super().__init__().

    Warning:
        Failing to set the self.model parameter will throw an exception.

    Attributes:
        model: TorchScript-compatible neural network instance which returns bias energy
            from particle coordinates.
        model_loc: Path to save module to.
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

################################################################################
# Analytical biases
################################################################################

# TODO: Implement

################################################################################
# Neural network biases
################################################################################

# Static biases
# -------------
# These cannot be updated


class HarmonicBias_SingleParticle_1D(NNBias):
    """
    Applies a harmonic bias potential $k/2 (s - s0)^2$ along the x- or y- coordinate
    of a single particle.

    Attributes:
        k (float): Strength of harmonic bias.
        s0 (float): Location (x-/y- value) of harmonic bias' center.
        axis (str): Axis (x/y) along which bias is applied.
        model_loc: Path to save module to.
    """
    def __init__(self, k, s0, axis='x', model_loc="model.pt"):
        if axis == 'x':
            self.model = HarmonicBias_SingleParticle_1D_ForceModule(k, s0, axis='x')
        elif axis == 'y':
            self.model = HarmonicBias_SingleParticle_1D_ForceModule(k, s0, axis='y')
        else:
            raise ValueError("Invalid axis")
        super().__init__(model_loc)


class HarmonicBias_SingleParticle_1D_ForceModule(torch.nn.Module):
    """
    A harmonic potential $k/2 (s - s0)^2$ as a static compute graph.

    The `axis` attribute specifies whether the potential is applied to the x-
    or the y- coordinate of the particle.

    Attributes:
        k (float): Strength of harmonic bias.
        s0 (float): Location (x-/y- value) of harmonic bias' center.
        axis (str): Axis (x/y) along which bias is applied.
    """
    def __init__(self, k, s0, axis='x'):
        self.k = k
        self.s0 = s0
        self.axis = axis
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
        # Extract coordinate
        if self.axis == 'x':
            s = positions[:, 0]
        elif self.axis == 'y':
            s = positions[:, 1]
        else:
            raise ValueError("Invalid axis")
        return self.k / 2 * torch.sum((s - self.s0) ** 2)


class StaticBias_SingleParticle(NNBias):
    """
    Applies a **static** basis set expanded potential to a single particle.

    The potential (V_module) can be an instance of:
        a) ves.basis.LegendreBasis1D: an expansion over a basis set (Valsson and Parrinello 2014) along the x- or the y- coordinate.
        b) ves.basis.LegendreString2D: an expansion over a basis set (Valsson and Parrinello 2014) along a string in the x- and y- coordinates.
        c) ves.basis.NNBasis1D: a neural network (Valsson and Parrinello 2014) along the x- or the y- coordinate.

    Attributes:
        V_module: TorchScript compatible neural network instance.
        model_loc: Path to save module to.
    """
    def __init__(self, V_module, model_loc):
        # Bias potential V(x) or V(y)
        self.model = V_module
        super().__init__(model_loc)


# Dynamic biases
# --------------
# These can be updated


class VESBias_SingleParticle_1D(NNBias):
    """
    Applies a **dynamic** basis set expanded potential along the x- or y- coordinate of a single particle.
    The parameters of the basis set can be updated by calling the `update` function with
    a trajectory.

    The potential (V_module) can be an instance of:
        a) ves.basis.LegendreBasis1D: an expansion over a basis set (Valsson and Parrinello 2014).
        b) ves.basis.NNBasis1D: a neural network (Valsson and Parrinello 2014).

    The `axis` attribute of the V_module instance specifies whether the potential
    acts along the x- or y- axis.

    Attributes:
        V_module:
        target:
        beta:
        optimizer_type:
        optimizer_params:
        model_loc:
        update_steps:
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
