"""Tinygrad-compatible learning-rate schedulers."""

from typing import List

from polygrad import Tensor
from polygrad.nn.optim import Optimizer


class LR_Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.epoch_counter = Tensor([0], device=self.optimizer.device, _ctx=self.optimizer._ctx)

    def get_lr(self):
        raise NotImplementedError

    def schedule_step(self) -> List[Tensor]:
        return [
            self.epoch_counter.assign(self.epoch_counter + 1),
            self.optimizer.lr.assign(self.get_lr()),
        ]

    def step(self) -> None:
        scheduled = self.schedule_step()
        scheduled[0].realize(*scheduled[1:])


class LRSchedulerGroup:
    def __init__(self, *schedulers: LR_Scheduler):
        self.schedulers = schedulers

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()


class OneCycleLR(LR_Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        div_factor: float,
        final_div_factor: float,
        total_steps: int,
        pct_start: float,
        anneal_strategy: str = 'linear',
        cycle_momentum: bool = False,
    ):
        super().__init__(optimizer)
        self.initial_lr = max_lr / div_factor
        self.max_lr = max_lr
        self.min_lr = self.initial_lr / final_div_factor
        self.total_steps = total_steps
        self.pct_start = pct_start
        assert anneal_strategy == 'linear', 'only linear annealing supported'
        assert not cycle_momentum, 'cycle momentum not supported'
        self.optimizer.lr.assign(self.get_lr()).realize()

    @staticmethod
    def _annealing_linear(start: float, end: float, pct: Tensor) -> Tensor:
        return pct * (end - start) + start

    def get_lr(self) -> Tensor:
        up_steps = self.total_steps * self.pct_start
        down_steps = self.total_steps * (1 - self.pct_start)
        return (self.epoch_counter < up_steps).where(
            self._annealing_linear(self.initial_lr, self.max_lr, self.epoch_counter / up_steps),
            self._annealing_linear(self.max_lr, self.min_lr, (self.epoch_counter - up_steps) / down_steps),
        ).cast(self.optimizer.lr.dtype)


__all__ = ['LR_Scheduler', 'LRSchedulerGroup', 'OneCycleLR']
