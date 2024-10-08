import pathlib
import statistics
import numpy as np
import torchvision
#import cv2

from model.lane_keeping.dave.dave_model import Dave2
# import pygame
# import torch
# import torchvision

from .action import UdacityAction
from .observation import UdacityObservation


class UdacityAgent:

    def __init__(self, before_action_callbacks=None, after_action_callbacks=None):
        self.after_action_callbacks = after_action_callbacks
        self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
        self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []

    def on_before_action(self, observation: UdacityObservation):
        for callback in self.before_action_callbacks:
            callback(observation)

    def on_after_action(self, observation: UdacityObservation):
        for callback in self.after_action_callbacks:
            callback(observation)

    def action(self, observation: UdacityObservation, *args, **kwargs):
        raise NotImplementedError('UdacityAgent does not implement __call__')

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        # TODO: the image should never be none (by design)
        if observation.input_image is None:
            return UdacityAction(steering_angle=0.0, throttle=0.0)

        for callback in self.before_action_callbacks:
            callback(observation)

        action = self.action(observation, *args, **kwargs)

        for callback in self.after_action_callbacks:
            callback(observation, action=action)

        return action


class PIDUdacityAgent(UdacityAgent):

    def __init__(self, kp, kd, ki, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.ki = ki  # Integral gain
        self.prev_error = 0.0
        self.total_error = 0.0

    def action(self, observation: UdacityObservation, *args, **kwargs):
        error = (observation.next_cte + observation.cte) / 2
        diff_err = error - self.prev_error

        # Calculate steering angle
        steering_angle = - (self.kp * error) - (self.kd * diff_err) - (self.ki * self.total_error)
        steering_angle = max(-1, min(steering_angle, 1))

        # Calculate throttle
        throttle = 1

        # Save error for next prediction
        self.total_error += error
        self.total_error = self.total_error * 0.99
        self.prev_error = error

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)

class DaveUdacityAgent(UdacityAgent):

    def __init__(self, checkpoint_path, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.model = Dave2.load_from_checkpoint(self.checkpoint_path)
        # TODO: I probably need to check where the model is

    def action(self, observation: UdacityObservation, *args, **kwargs):

        # Cast input to right shape
        # input_image = torchvision.transforms.functional.pil_to_tensor(observation.input_image)
        input_image = torchvision.transforms.ToTensor()(observation.input_image)

        # Calculate steering angle
        steering_angle = self.model(input_image).item()
        # Calculate throttle
        throttle = 1

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)
    

class DaveUdacityAgentWithDropout(UdacityAgent):

    def __init__(self, checkpoint_path, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.model = Dave2.load_from_checkpoint(self.checkpoint_path)
        # TODO: I probably need to check where the model is

    def action(self, observation: UdacityObservation, *args, **kwargs):

        # Cast input to right shape
        # input_image = torchvision.transforms.functional.pil_to_tensor(observation.input_image)
        input_image = torchvision.transforms.ToTensor()(observation.input_image)

        # Calculate steering angle
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        N = 32
        # make the predictions
        predictions = []
        for _ in range(N):
            predictions.append(self.model(input_image).item())
        
        mean_value = statistics.mean(predictions)
        variance_value = statistics.variance(predictions)

        steering_angle = mean_value
        # Calculate throttle
        throttle = 1

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)
    

class DaveUdacityAgentWithAugmentation(UdacityAgent):

    def __init__(self, checkpoint_path, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.model = Dave2.load_from_checkpoint(self.checkpoint_path)
        # TODO: I probably need to check where the model is

    def action(self, observation: UdacityObservation, *args, **kwargs):

        input_image = torchvision.transforms.ToTensor()(observation.input_image)

        #image= np.array(observation.input_image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #transform = A.Compose([
        #   A.Blur(p=0.2),
        #])

        #transformed = transform(image=image)
        #transformed_image = transformed["image"]
        #input_image = torchvision.transforms.ToTensor()(transformed_image)
        self.model.eval()
        
        steering_angle = self.model(input_image).item()
        # Calculate throttle
        throttle = 1

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)