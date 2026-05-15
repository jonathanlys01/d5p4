from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from d5p4.config import Config

    from ._greedy_map import _GreedyMAP
    from .base import BaseSelector
    from .baseline import BaselineSelection
    from .beam import DiverseBeamSearch, GreedyBeamSearch
    from .dpp_selector import DPP
    from .exhaustive import Exhaustive
    from .greedy_map import GreedyMAP
    from .random_selector import RandomSelection

AVAIL = {
    "dpp": ("d5p4.subsample.dpp_selector", "DPP"),
    "exhaustive": ("d5p4.subsample.exhaustive", "Exhaustive"),
    "greedy_map": ("d5p4.subsample.greedy_map", "GreedyMAP"),
    "_greedy_map": ("d5p4.subsample._greedy_map", "_GreedyMAP"),
    "greedy_beam": ("d5p4.subsample.beam", "GreedyBeamSearch"),
    "diverse_beam": ("d5p4.subsample.beam", "DiverseBeamSearch"),
    "random": ("d5p4.subsample.random_selector", "RandomSelection"),
    "baseline": ("d5p4.subsample.baseline", "BaselineSelection"),
}


def get_subsample_selector(
    config: Config,
) -> "GreedyBeamSearch | BaseSelector | DiverseBeamSearch | DPP | Exhaustive | GreedyMAP | _GreedyMAP | RandomSelection | BaselineSelection":  # noqa: E501
    """
    Factory function to dynamically load and instantiate a subset selector.
    """
    name = config.method.lower()
    import_info = AVAIL.get(name)

    if import_info is None:
        raise ValueError(f"Unknown subset selector: {name}")

    module_path, class_name = import_info

    try:
        module = importlib.import_module(module_path)
        selector_class = getattr(module, class_name)

    except ImportError as e:
        raise ImportError(
            f"Could not lazily import subset selector '{name}'. "
            f"Failed to import {class_name} from {module_path}. Error: {e}",
        )
    except AttributeError:
        raise AttributeError(
            f"Could not find class '{class_name}' in module '{module_path}'.",
        )

    return selector_class(config)
