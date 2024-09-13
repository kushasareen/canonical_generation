import torch
import copy

#torchvision ema implementation
#https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class CustomExponentialMovingAverage:
    def __init__(self, model, decay, device="cpu"):
        self.model = model
        self.decay = decay
        self.device = device
        self.ema_model = self._deepcopy_model(model)

    def _deepcopy_model(self, model):
        """Creates a deep copy of the model to maintain EMA parameters."""
        return copy.deepcopy(model)

    def update_parameters(self, model):
        """Update the EMA parameters using the provided model."""
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            if ema_param.shape == model_param.shape:
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=(1 - self.decay))
            else:
                print(f"Skipping EMA update due to shape mismatch: {ema_param.shape} vs {model_param.shape}")

    def sample(self, *args, **kwargs):
        """Call the sampling method from the EMA model."""
        self.ema_model.eval()
        self.ema_model.to(self.device)
        return self.ema_model.sampling(*args, **kwargs)