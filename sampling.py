import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils
from utils import dprint

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        dprint(">> EulerPredictor")
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        dprint(">> NonePredictor")
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        dprint(">> AnalyticPredictor")
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma
        dprint(f"dsigma({dsigma}) = curr_sigma({curr_sigma}) - next_sigma({next_sigma})")

        score = score_fn(x, curr_sigma)  # [batch_size, seq_len, vocab_size]
        dprint("score:", score.shape)
        dprint(score)
        dprint(score[:,0,:])
        dprint(score[:,1,:])
        dprint(score[:,2,:])

        stag_score = self.graph.staggered_score(score, dsigma)  # [batch_size, seq_len, vocab_size]
        dprint("stag_score(score, dsigma):", stag_score.shape)
        dprint(stag_score.shape)
        
        probs = stag_score * self.graph.transp_transition(x, dsigma)  # [batch_size, seq_len, vocab_size]
        
        dprint("transp: ", self.graph.transp_transition(x, dsigma).shape)
        dprint(self.graph.transp_transition(x, dsigma))
        
        dprint("probs = stag_score ", probs.shape)
        dprint(probs.shape)

        return sample_categorical(probs) # x: [batch_size, seq_len]

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise
        
        dprint("graph")
        dprint(graph)
        dprint("noise")
        dprint(noise)

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]
        
        dprint(f'>> Denoiser.update_fn(score_fn={score_fn}, x={x}, t={t}) => sigma: {sigma}')

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    dprint('predictor')
    dprint(predictor)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        dprint(">> pc_sampler")
        
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        dprint("sampling_score_fn")
        dprint(sampling_score_fn)
        
        x = graph.sample_limit(*batch_dims).to(device)
        dprint("x", x)
        
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dprint("timesteps", timesteps)
        
        dt = (1 - eps) / steps
        dprint("dt", dt)

        for i in range(steps): # predictor
            dprint("i:", i)
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)            

        dprint("x", x)

        if denoise: # denoisor = basically predictor without next sigma and [MASK] token prob.
            dprint(">> denoise")
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        dprint("x", x)
        
        return x
    
    return pc_sampler

