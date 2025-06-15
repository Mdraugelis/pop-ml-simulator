"""Simple Hydra application tutorial.

This demonstrates the basics of Hydra configuration.
"""
import hydra  # type: ignore
from omegaconf import DictConfig, OmegaConf  # type: ignore
import logging
from pathlib import Path


# This decorator tells Hydra where to find configs
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main function that receives configuration.

    Args:
        cfg: Configuration object from Hydra
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Print the entire configuration
    logger.info("Full configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Access configuration values
    logger.info(f"\nApp name: {cfg.app.name}")
    logger.info(f"Version: {cfg.app.version}")
    logger.info(f"Number of steps: {cfg.simulation.n_steps}")
    logger.info(f"Random seed: {cfg.simulation.seed}")

    # Show current working directory
    logger.info(f"\nCurrent directory: {Path.cwd()}")

    # Create some output
    output_file = Path("simulation_result.txt")
    with open(output_file, "w") as f:
        f.write(f"Simulation with {cfg.simulation.n_steps} steps completed!\n")
        f.write(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    logger.info(f"Results saved to: {output_file.absolute()}")

    return {"status": "success", "n_steps": cfg.simulation.n_steps}


if __name__ == "__main__":
    main()
