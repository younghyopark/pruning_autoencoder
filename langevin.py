import numpy as np
import torch
import torch.autograd as autograd

def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x


def sample_langevin(x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False,
                    clip_x=None, clip_grad=None, reflect=False, noise_anneal=None,
                    spherical=False):
    """Draw samples using Langevin dynamics
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    """
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    noise_scale_ = noise_scale

    l_samples = []
    l_dynamics = []
    x.requires_grad = True
    for i_step in range(n_steps):
        l_samples.append(x.detach().to('cpu'))
        noise = torch.randn_like(x) * noise_scale_
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        if clip_grad is not None:
            grad = clip_vector_norm(grad, max_norm=clip_grad)
        dynamics = - stepsize * grad + noise  # negative!
        x = x + dynamics
        if clip_x is not None and not reflect:
            x = torch.clamp(x, clip_x[0], clip_x[1])

        if spherical:
            if len(x.shape) == 4:
                x = x / x.view(len(x), -1).norm(dim=1)[:, None, None ,None]
            else:
                x = x / x.norm(dim=1, keepdim=True)

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)
        # elif clip_x is not None and reflect:
        #     x = torch.min(x, 2 * clip_x[1] - x)
        #     x = torch.max(x, 2 * clip_x[0] - x)
        l_dynamics.append(dynamics.detach().to('cpu'))
    l_samples.append(x.detach().to('cpu'))

    if intermediate_samples:
        return l_samples, l_dynamics
    else:
        return x.detach()




def HMCwithAccept(energy, x, length, steps, epsilon, m=1, bound=None, T=1):
    '''from https://github.com/li012589/HMC_pytorch/blob/master/hmc.py
    length: the number of leap frog trajectories '''
    shape = [i if no==0 else 1 for no,i in enumerate(x.shape)]
    def grad(z):
        return autograd.grad(energy(z),z,grad_outputs=z.new_ones(z.shape[0]))[0]
    torch.set_grad_enabled(False)
    E = energy(x)
    torch.set_grad_enabled(True)
    g = grad(x.requires_grad_())
    torch.set_grad_enabled(False)
    g = g.detach()
    for l in range(length):
        p = x.new_empty(size=x.size()).normal_() * np.sqrt(m) * T
        H = ((0.5*p*p/m).reshape(p.shape[0], -1).sum(dim=1) + E)
        xnew = x
        gnew = g
        for _ in range(steps):
            p = p- epsilon* gnew/2.
            xnew = (xnew + epsilon * p / m)
            if bound is not None:
                xnew = torch.clamp(xnew, bound[0], bound[1])
            torch.set_grad_enabled(True)
            gnew = grad(xnew.requires_grad_())
            torch.set_grad_enabled(False)
            xnew = xnew.detach()
            gnew = gnew.detach()
            p = p- epsilon* gnew/2.
        Enew = energy(xnew)
        Hnew = (0.5*p*p/m).reshape(p.shape[0], -1).sum(dim=1) + Enew
        diff = (H-Hnew) / T
        accept = (diff.exp() >= diff.uniform_()).to(x)

        E = accept*Enew + (1.-accept)*E
        acceptMask = accept.reshape(shape)
        x = acceptMask*xnew + (1.-acceptMask)*x
        g = acceptMask*gnew + (1.-acceptMask)*g
    torch.set_grad_enabled(True)

    return x, accept
