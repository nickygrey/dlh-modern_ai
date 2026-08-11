#!/usr/bin/env python3
"""Module to create SGD optimizer with learning rate schedules."""
from tensorflow import keras


def get_optimizer_SGD_with_schedule(
    schedule_type, initial_lr, decay_steps, decay_rate, momentum
):
    """Create an SGD optimizer with a specified learning rate schedule.

    Args:
        schedule_type (str): Schedule type ('exponential' or 'inverse_time').
        initial_lr (float): Initial learning rate.
        decay_steps (int): Decay steps before applying decay.
        decay_rate (float): Decay rate factor.
        momentum (float): Momentum factor for SGD.

    Returns:
        tuple: (optimizer, lr_schedule) where optimizer is a Keras SGD instance
               and lr_schedule is the LearningRateSchedule object.
    """
    if schedule_type == 'exponential':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    elif schedule_type == 'inverse_time':
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
    else:
        lr_schedule = initial_lr

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum
    )

    return optimizer, lr_schedule
