import torch
import numpy as np
from torch.nn.functional import softmax
import cv2


class SaveFeatures(object):
    def __init__(self, m):
        self.features = None
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.data.cpu()

    def remove(self):
        self.hook.remove()


class FeaturesExtractor(object):
    def __init__(self, model, layer_name, data_size):
        self.model = model
        self.final_layer = self.model._modules.get(layer_name)
        self.data_size = data_size
        assert self.final_layer
        self.activated_features = None

    def connect(self):
        self.activated_features = SaveFeatures(self.final_layer)

    def remove(self):
        self.activated_features.remove()

    def get_cam(self, model_output):
        probability_ = softmax(model_output.cpu(), dim=1).data.squeeze()
        class_idx = torch.topk(probability_, 1)[1].int().numpy()[0]
        weight_softmax_params = self.model._modules.get('fc').parameters()
        weight_softmax_params = list(weight_softmax_params)
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        _, nc, n_size = self.activated_features.features.shape
        cam = weight_softmax[class_idx].dot(self.activated_features.features.reshape((nc, n_size)))
        cam = cam.reshape(1, n_size)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam_img = cv2.resize(cam[0], (1, self.data_size))
        return cam_img
