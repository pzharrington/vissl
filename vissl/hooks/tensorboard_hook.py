# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import is_primary
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.utils.activation_statistics import (
    ActivationStatistics,
    ActivationStatisticsMonitor,
    ActivationStatisticsObserver,
)


# NEW: imports for plotting sample imgs
import numpy as np
import matplotlib.pyplot as plt
from vissl.utils.decals_rgb import dr2_rgb

def visualize_img_batch(view1, view2):
    # Only viz up to 4 from batch
    bs = view1.shape[0]
    vbs = min(bs, 4)
    f = plt.figure(figsize=(vbs*2, 4))
    for i in range(vbs):
        plt.subplot(2,vbs,i+1)
        plt.imshow(convert_decals_rgb(view1[i,:,:,:]))
        plt.axis('off')
        plt.subplot(2,vbs,i+1+vbs)
        plt.imshow(convert_decals_rgb(view2[i,:,:,:]))
        plt.axis('off')
    return f

def convert_decals_rgb(img):
    img = img.transpose((1,2,0))
    imgs = [img[:,:,i] for i in range(img.shape[-1])]
    return dr2_rgb(imgs, ['g', 'r', 'z'])


try:
    from torch.utils.tensorboard import SummaryWriter  # noqa F401

    tb_available = True
except ImportError:
    # Make sure that the type hint is not blocking
    # on a non-TensorBoard aware platform
    from typing import TypeVar

    SummaryWriter = TypeVar("SummaryWriter")
    tb_available = False

BYTE_TO_MiB = 2 ** 20


class ActivationStatisticsTensorboardWatcher(ActivationStatisticsObserver):
    """
    Implementation of ActivationStatisticsObserver which logs the
    activation statistics to tensorboard.
    """

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def consume(self, stat: ActivationStatistics):
        self.writer.add_scalar(
            tag="activations/" + stat.name + "/mean",
            scalar_value=stat.mean,
            global_step=stat.iteration,
        )
        self.writer.add_scalar(
            tag="activations/" + stat.name + "/spread",
            scalar_value=stat.spread,
            global_step=stat.iteration,
        )


class SSLTensorboardHook(ClassyHook):
    """
    SSL Specific variant of the Classy Vision tensorboard hook
    """

    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop

    def __init__(
        self,
        tb_writer: SummaryWriter,
        log_params: bool = False,
        log_params_every_n_iterations: int = -1,
        log_params_gradients: bool = False,
        log_activation_statistics: int = 0,
        visualize_samples: bool = False,
    ) -> None:
        """The constructor method of SSLTensorboardHook.

        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
                        readthedocs.io/en/latest/tensorboard.html#tensorboardX.
                        SummaryWriter>`_ instance
            log_params (bool): whether to log model params to tensorboard
            log_params_every_n_iterations (int): frequency at which parameters
                        should be logged to tensorboard
            log_params_gradients (bool): whether to log params gradients as well
                        to tensorboard.
            visualize_samples (bool): whether to visualize batch samples for debugging
        """
        super().__init__()
        if not tb_available:
            raise RuntimeError(
                "tensorboard not installed, cannot use SSLTensorboardHook"
            )
        logging.info("Setting up SSL Tensorboard Hook...")
        self.tb_writer = tb_writer
        self.log_params = log_params
        self.log_params_every_n_iterations = log_params_every_n_iterations
        self.log_params_gradients = log_params_gradients
        self.log_activation_statistics = log_activation_statistics
        self.visualize_samples = visualize_samples
        if self.log_activation_statistics > 0:
            self.activation_watcher = ActivationStatisticsMonitor(
                observer=ActivationStatisticsTensorboardWatcher(tb_writer),
                log_frequency=self.log_activation_statistics,
                sample_feature_map=True,
            )
        logging.info(
            f"Tensorboard config: log_params: {self.log_params}, "
            f"log_params_freq: {self.log_params_every_n_iterations}, "
            f"log_params_gradients: {self.log_params_gradients}, "
            f"log_activation_statistics: {self.log_activation_statistics}"
        )

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of training.
        """
        if self.log_activation_statistics and is_primary():
            self.activation_watcher.monitor(task.base_model)
            self.activation_watcher.set_iteration(task.iteration)

    def on_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of training.
        """
        if self.log_activation_statistics and is_primary():
            self.activation_watcher.stop()

    def on_forward(self, task: "tasks.ClassyTask") -> None:
        """
        Called after every forward if tensorboard hook is enabled.
        Logs the model parameters if the training iteration matches the
        logging frequency.
        """
        if not is_primary():
            return

        if self.log_activation_statistics:
            self.activation_watcher.set_iteration(task.iteration + 1)

        if (
            self.log_params
            and self.log_params_every_n_iterations > 0
            and task.train
            and task.iteration % self.log_params_every_n_iterations == 0
        ):
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=task.iteration
                )

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of every epoch if the tensorboard hook is
        enabled.
        Logs the model parameters once at the beginning of training only.
        """
        if not self.log_params:
            return

        # log the parameters just once, before training starts
        if is_primary() and task.train and task.train_phase_idx == 0:
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=-1
                )

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of every epoch if the tensorboard hook is
        enabled.
        Log model parameters and/or parameter gradients as set by user
        in the tensorboard configuration. Also resents the CUDA memory counter.
        """
        # Log train/test accuracy
        if is_primary():
            phase_type = "Training" if task.train else "Testing"
            for meter in task.meters:
                if "accuracy" in meter.name:
                    for top_n, accuracies in meter.value.items():
                        for i, acc in accuracies.items():
                            tag_name = f"{phase_type}/Accuracy_" f" {top_n}_Output_{i}"
                            self.tb_writer.add_scalar(
                                tag=tag_name,
                                scalar_value=round(acc, 5),
                                global_step=task.train_phase_idx,
                            )
                if 'photoz' in meter.name:
                    for name, val in meter.value.items():
                        tag_name = f"{phase_type}/photoz_" f" {name}"
                        self.tb_writer.add_scalar(
                            tag=tag_name,
                            scalar_value=val,
                            global_step=task.train_phase_idx,
                        )

        if not (self.log_params or self.log_params_gradients):
            return

        if is_primary() and task.train:
            # Log the weights and bias at the end of the epoch
            if self.log_params:
                for name, parameter in task.base_model.named_parameters():
                    self.tb_writer.add_histogram(
                        f"Parameters/{name}",
                        parameter,
                        global_step=task.train_phase_idx,
                    )
            # Log the parameter gradients at the end of the epoch
            if self.log_params_gradients:
                for name, parameter in task.base_model.named_parameters():
                    if parameter.grad is not None:
                        try:
                            self.tb_writer.add_histogram(
                                f"Gradients/{name}",
                                parameter.grad,
                                global_step=task.train_phase_idx,
                            )
                        except ValueError:
                            logging.info(
                                f"Gradient histogram empty for {name}, "
                                f"iteration {task.iteration}. Unable to "
                                f"log gradient."
                            )

            # Reset the GPU Memory counter
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after every parameters update if tensorboard hook is enabled.
        Logs the parameter gradients if they are being set to log,
        log the scalars like training loss, learning rate, average training
        iteration time, batch size per gpu, img/sec/gpu, ETA, gpu memory used,
        peak gpu memory used.
        """

        if not is_primary():
            return

        iteration = task.iteration

        if (
            self.log_params_every_n_iterations > 0
            and self.log_params_gradients
            and task.train
            and iteration % self.log_params_every_n_iterations == 0
        ):
            logging.info(f"Logging Parameter gradients. Iteration {iteration}")
            for name, parameter in task.base_model.named_parameters():
                if parameter.grad is not None:
                    try:
                        self.tb_writer.add_histogram(
                            f"Gradients/{name}",
                            parameter.grad,
                            global_step=task.iteration,
                        )
                    except ValueError:
                        logging.info(
                            f"Gradient histogram empty for {name}, "
                            f"iteration {task.iteration}. Unable to "
                            f"log gradient."
                        )

        if iteration % task.config["LOG_FREQUENCY"] == 0 or (
            iteration <= 100 and iteration % 5 == 0
        ):
            logging.info(f"Logging metrics. Iteration {iteration}")
            self.tb_writer.add_scalar(
                tag="Training/Loss",
                scalar_value=round(task.last_batch.loss.data.cpu().item(), 5),
                global_step=iteration,
            )

            self.tb_writer.add_scalar(
                tag="Training/Learning_rate",
                scalar_value=round(task.optimizer.options_view.lr, 5),
                global_step=iteration,
            )

            # Plot some images from the batch to tensorboard
            if self.visualize_samples:
                assert 'data_momentum' in task.last_batch.sample.keys(), 'Tensorboard sample visualizing currently only supported for MoCo'
                view1 = task.last_batch.sample["input"]
                view2 = task.last_batch.sample["data_momentum"]
                fig = visualize_img_batch(view1.detach().cpu().numpy(), view2[0].detach().cpu().numpy())
                self.tb_writer.add_figure('Samples/Augmented_views', fig, global_step=iteration, close=True)
                

            # Batch processing time
            if len(task.batch_time) > 0:
                batch_times = task.batch_time
            else:
                batch_times = [0]

            batch_time_avg_s = sum(batch_times) / max(len(batch_times), 1)
            self.tb_writer.add_scalar(
                tag="Speed/Batch_processing_time_ms",
                scalar_value=int(1000.0 * batch_time_avg_s),
                global_step=iteration,
            )

            # Images per second per replica
            pic_per_batch_per_gpu = task.config["DATA"]["TRAIN"][
                "BATCHSIZE_PER_REPLICA"
            ]
            pic_per_batch_per_gpu_per_sec = (
                int(pic_per_batch_per_gpu / batch_time_avg_s)
                if batch_time_avg_s > 0
                else 0.0
            )
            self.tb_writer.add_scalar(
                tag="Speed/img_per_sec_per_gpu",
                scalar_value=pic_per_batch_per_gpu_per_sec,
                global_step=iteration,
            )

            # ETA
            avg_time = sum(batch_times) / len(batch_times)
            eta_secs = avg_time * (task.max_iteration - iteration)
            self.tb_writer.add_scalar(
                tag="Speed/ETA_hours",
                scalar_value=eta_secs / 3600.0,
                global_step=iteration,
            )

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                self.tb_writer.add_scalar(
                    tag="Memory/Peak_GPU_Memory_allocated_MiB",
                    scalar_value=torch.cuda.max_memory_allocated() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                # Memory reserved by PyTorch's memory allocator
                self.tb_writer.add_scalar(
                    tag="Memory/Peak_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.max_memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )

                self.tb_writer.add_scalar(
                    tag="Memory/Current_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )
