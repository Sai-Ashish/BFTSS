from transformers import TrainingArguments
import torch

class customTrainingArguments(TrainingArguments):
    def __init__(self,*args, **kwargs):
        self._device = torch.device("cuda:0")

        super(customTrainingArguments, self).__init__(*args, **kwargs)

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        Name the device the number you use.
        """
        return self._device

    @device.setter
    def device(self, value):
        """
        The device used by this process.
        Name the device the number you use.
        """
        self._device = torch.device(value)

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        # _ = self._setup_devices
        # I set to one manullay
        self._n_gpu = 1
        return self._n_gpu