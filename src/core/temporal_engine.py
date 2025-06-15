from dataclasses import dataclass

from omegaconf import DictConfig

from src.utils.logging import log_call


@dataclass
class TemporalEngine:
    cfg: DictConfig

    @log_call
    def run(self) -> None:
        """Placeholder run method."""
        months = self.cfg.simulation.total_months
        for _ in range(months):
            pass
