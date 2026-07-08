import torch
import segmentation_models_pytorch as smp
import numpy as np


class Segmenter:
    def __init__(self, UNet=True,
        encoder_name: str = 'resnet18', 
        encoder_depth : int = 5, 
        encoder_weights : str = 'imagenet', 
        classes : int =  1,
        channels : int = 1,
        checkpoint_path = None):

        self.UNet = UNet
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights
        self.classes = classes
        self.in_channels = channels
        # TODO: Add other models, let's work with UNET for now
        if self.UNet == True:
            self.model = smp.Unet(encoder_name = self.encoder_name, 
                                encoder_depth = self.encoder_depth, 
                                encoder_weights = self.encoder_weights,
                                 in_channels = self.in_channels )

    def _make_tensors_(self, candidates, batch = 1, dimensions = 1):
        self.tensors = []
        for candidate in candidates:
            array = np.expand_dims(np.expand_dims(candidate.roi, 0), 0)
            # if batch or dimensions != 1:
            #     # TODO: Do this
            #     raise ValueError("I'll fix this later twin i'm sorry")
            tensor = torch.from_numpy(array).float()
            self.tensors.append(tensor)
        return self.tensors

    def segment(self, candidates=None, threshold=0.5):
        self.candidates = candidates
        self.threshold = threshold
        self.model.eval()
        tensors = self._make_tensors_(self.candidates)
        for tensor, candidate in zip(tensors, self.candidates):
            with torch.inference_mode():
                probability_map = self.model(tensor)
                candidate.mask = probability_map > threshold
        
        
