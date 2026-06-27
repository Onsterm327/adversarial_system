from .utils import *



class EBMModel:
    def __init__(self, model, times = 10):
        self.model = model
        self.times = times

    def purify(self, x):
        x = (x - 0.5) / 0.5
        # iterative langevin MCMC updates
        X_purified = torch.autograd.Variable(x.clone(), requires_grad=True)
        for ell in range(self.times):
            U_prime = torch.autograd.grad(self.model(X_purified).sum(), [X_purified])[0]
            X_purified.data += - U_prime + 1e-2 * torch.randn_like(U_prime)
        X_purified = (X_purified + 1) / 2
        return X_purified 

class GANModel:
    def __init__(self, model):
        self.model = model

    def purify(self, x):
        return self.model(x)

class CleanModel:
    def __init__(self):
        pass

    def purify(self, x):
        return x

class GLOWModel:
    def __init__(self, model):
        self.model = model

    def purify(self, inputs):
        with torch.no_grad():
            z, nll, _ = self.model(x=preprocess(inputs), y_onehot=None)
            purified_inputs = postprocess(self.model(y_onehot=None, z=z, temperature=1, reverse=True))
        return purified_inputs

class NCSNV2Model:
    def __init__(self, model):
        self.model = model

    def purify(self, x):
        min_step_lr = 0.00001
        lr_min = 1.0e-3
        images = [] # From noisy initialized image to purified image
        step_sizes = [] # Step sizes
        max_iter = 10
        transform_raw_to_ebm = raw_to_ebm("CIFAR10")
        transform_ebm_to_raw = ebm_to_raw("CIFAR10")

        with torch.no_grad():
            smoothing_level = 0.25
            
            x_pur = torch.clamp(x + torch.randn_like(x)*smoothing_level, 0.0, 1.0)
            x_pur = transform_raw_to_ebm(x_pur).to("cuda")
            
            images.append(x_pur.clone().detach())
            cont_purification = torch.ones(x_pur.shape[0], dtype=torch.bool).to("cuda")
            # Stopping criterion
            for i in range(max_iter):
                labels = torch.ones(x_pur.shape[0], device=x_pur.device)
                labels = labels.long().to("cuda")
                grad = self.model(x_pur, labels) # Get gradients
                # Get adaptive step size
                x_eps = x_pur + lr_min*grad
                grad_eps = self.model(x_eps, labels)
                z1 = torch.bmm(grad.view(grad.shape[0], 1, -1), grad_eps.view(grad_eps.shape[0], -1, 1))
                z2 = torch.bmm(grad.view(grad.shape[0], 1, -1), grad.view(grad.shape[0], -1, 1))
                z = torch.div(z1, z2)
                step_lambda = 0.05
                step_size = torch.clamp(step_lambda*lr_min/(1.-z), min=min_step_lr, max=min_step_lr*10000.).view(-1)
                cont_purification = torch.logical_and(cont_purification, (step_size>0.001))
                if torch.sum(cont_purification)==0:
                    break
                step_size *= cont_purification
                x_pur_t = x_pur.clone().detach()
                x_pur = torch.clamp(transform_ebm_to_raw(x_pur_t+grad*step_size[:, None, None, None]), 0.0, 1.0)
                step_sizes.append(step_size)
                images.append(x_pur)
        return images[-1]
    
class DiffusionModel:
    def __init__(self, model):
        self.model = model

    def purify(self, x):
        return self.model(x, 10)