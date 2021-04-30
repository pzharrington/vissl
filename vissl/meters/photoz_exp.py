from classy_vision.meters import ClassyMeter, register_meter
from classy_vision.generic.distributed_util import all_reduce_sum, gather_from_all
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.config import AttrDict
import logging

import torch
import torch.nn.functional as F
import numpy as np
from astropy.stats import mad_std

@register_meter("photoz_exp")
class ZphotMeter(ClassyMeter):
    """
    
    Compute z_phot prediction from expectation over class bins, then get sigma_mad.
    Note ground truth specz used for sigma_mad computation is only accurate up to 
    photoz bin size, since train targets come from data loader already binned.
    Args:
        specz_range: range of photoz's spanned by class bins
        specz_nbins: number of bins
    """

    def __init__(self, meters_config: AttrDict):
        super().__init__()
        self.range = meters_config["specz_range"]
        self.nbins = meters_config["specz_nbins"]
        self.maxsamp = meters_config["max_num_samples"]

        lo, hi = self.range
        bins = np.linspace(lo, hi, num=self.nbins+1, endpoint=True)
        self.zbin_ctrs = np.expand_dims((bins[1:] + bins[:-1])/2., axis=0)
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        """
        Get the ZphotMeter instance from the user defined config
        """
        return cls(meters_config)

    @property
    def name(self):
        """
        Name of the meter
        """
        return "photoz_exp"

    @property
    def value(self):
        """
        Value of the meter which has been globally synced. This is the value printed and
        recorded by user.
        """
        _, distributed_rank = get_machine_local_and_dist_rank()
        logging.info(
            f"Rank: {distributed_rank} Photoz sigma MAD meter: "
            f"scores: {self._scores.shape}, target: {self._targets.shape}"
        )
        pred_pdf = F.softmax(self._scores, dim=-1).detach().numpy()
        targets = self._targets.detach().numpy()
        
        photoz = np.sum(pred_pdf*self.zbin_ctrs, axis=-1)
        specz = self.zbin_ctrs[0,targets]

        delz = (photoz - specz)/(1.+ specz)
        meandelz = np.mean(delz)
        madstd = mad_std(delz)
        th = 0.1
        eta = np.sum(abs(delz)>th)/delz.shape[0]
        return {"mean_delz":meandelz, "sigma_mad": madstd, "eta_pct":eta*100}

    def gather_scores(self, scores: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            scores_gathered = gather_from_all(scores)
        else:
            scores_gathered = scores
        return scores_gathered

    def gather_targets(self, targets: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            targets_gathered = gather_from_all(targets)
        else:
            targets_gathered = targets
        return targets_gathered

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        We gather scores, targets, total sampled
        """
        # Communications
        self._curr_sample_count = all_reduce_sum(self._curr_sample_count)
        self._scores = self.gather_scores(self._scores)
        self._targets = self.gather_targets(self._targets)

        # Store results
        self._total_sample_count += self._curr_sample_count

        # Reset values until next sync
        self._curr_sample_count.zero_()

    def reset(self):
        """
        Reset the meter. Should reset all the meter variables, values.
        """
        self._scores = torch.zeros(0, self.nbins, dtype=torch.float32)
        self._targets = torch.zeros(0, dtype=torch.int8)
        self._total_sample_count = torch.zeros(1)
        self._curr_sample_count = torch.zeros(1)

    def __repr__(self):
        # implement what information about meter params should be
        # printed by print(meter). This is helpful for debugging
        return repr({"name": self.name, "value": self.value})

    def set_classy_state(self, state):
        """
        Set the state of meter. This is the state loaded from a checkpoint when the model
        is resumed
        """
        assert (
            self.name == state["name"]
        ), f"State name {state['name']} does not match meter name {self.name}"
        assert self.nbins == state['nbins'], (
            f"nbins of state {state['nbins']} "
            f"does not match object's nbins {self.nbins}"
        )
        assert self.range[0] == state['range'][0] and self.range[1] == state['specz_range'][1], (
            f"range of state {state['range']} "
            f"does not match object's range {self.range}"
        )
        assert self.maxsamp == state['maxsamp'], (
            f"maxsamp of state {state['maxsamp']} "
            f"does not match object's maxsamp {self.maxsamp}"
        )


        # Restore the state -- correct_predictions and sample_count.
        self.reset()
        self._total_sample_count = state["total_sample_count"].clone()
        self._curr_sample_count = state["curr_sample_count"].clone()
        self._scores = state["scores"]
        self._targets = state["targets"]

    def get_classy_state(self):
        """
        Returns the states of meter that will be checkpointed. This should include
        the variables that are global, updated and affect meter value.
        """
        return {
            "name": self.name,
            "nbins": self.nbins,
            "range": self.range,
            "maxsamp": self.maxsamp,
            "scores": self._scores,
            "targets": self._targets,
            "total_sample_count": self._total_sample_count,
            "curr_sample_count": self._curr_sample_count,
        }

    def update(self, model_output, target):
        """
        Update the meter every time meter is calculated
        """
        #self.validate(model_output, target)
        self._curr_sample_count += model_output.shape[0]

        curr_scores, curr_targets = self._scores, self._targets
        sample_count_so_far = curr_scores.shape[0]
        if sample_count_so_far>= self.maxsamp:
            # Stop recording after gathering maxsamp number of samples to save time
            del curr_scores, curr_targets
            return
        self._scores = torch.zeros(
            int(self._curr_sample_count[0]), self.nbins, dtype=torch.float32
        )
        self._targets = torch.zeros(
            int(self._curr_sample_count[0]), dtype=torch.int8
        )

        if sample_count_so_far > 0:
            self._scores[:sample_count_so_far, :] = curr_scores
            self._targets[:sample_count_so_far] = curr_targets
        self._scores[sample_count_so_far:, :] = model_output
        self._targets[sample_count_so_far:] = target
        del curr_scores, curr_targets

    def validate(self, model_output, target):
        """
        Validate that the input to meter is valid
        """
        #assert len(model_output.shape) == 2, "model_output should be a 2D tensor"
        #assert len(target.shape) == 2, "target should be a 2D tensor"
        #assert (
        #    model_output.shape[0] == target.shape[0]
        #), "Expect same shape in model output and target"
        #assert (
        #    model_output.shape[1] == target.shape[1]
        #), "Expect same shape in model output and target"
        #num_classes = target.shape[1]
        #assert num_classes == self.num_classes, "number of classes is not consistent"



