"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR

from ..core import register


class Warmup(object):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int=-1) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg['lr'] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        self.step()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'lr_scheduler'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_warmup_factor(self, step, **kwargs):
        raise NotImplementedError

    def step(self, ):
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            return
        factor = self.get_warmup_factor(self.last_step)
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            pg['lr'] = factor * self.warmup_end_values[i]

    def finished(self, ):
        if self.last_step >= self.warmup_duration:
            return True
        return False


@register()
class LinearWarmup(Warmup):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        super().__init__(lr_scheduler, warmup_duration, last_step)

    def get_warmup_factor(self, step):
        return min(1.0, (step + 1) / self.warmup_duration)


@register()
class LinearEpochLR(LambdaLR):
    """Epoch-indexed linear decay; advance even while batch warmup is active."""
    advance_during_warmup = True

    def __init__(self, optimizer, total_epochs=20, final_ratio=.01, last_epoch=-1):
        if total_epochs < 1 or not 0 < final_ratio <= 1:
            raise ValueError("Invalid linear epoch schedule")
        super().__init__(optimizer,lambda epoch: max(1-epoch/total_epochs,0)*(1-final_ratio)+final_ratio,last_epoch)


@register()
class GroupLinearWarmup(Warmup):
    """Interpolate each group's declared start LR to its current epoch LR."""
    def step(self):
        self.last_step += 1
        self.prepare_step()

    def prepare_step(self):
        # Epoch schedulers overwrite group LRs at epoch boundaries. Reapply
        # the current batch's interpolation before its optimizer update.
        if self.last_step > self.warmup_duration:
            return
        factor = self.last_step / max(1,self.warmup_duration)
        for group, scheduled in zip(self.lr_scheduler.optimizer.param_groups,self.lr_scheduler.get_last_lr()):
            start = group.get("warmup_start_lr",0.)
            group["lr"] = start + factor*(scheduled-start)
