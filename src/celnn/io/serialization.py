"""JSON serialization helpers for celnn objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.network import CellularNetwork
from ..core.simulation import SimulationConfig
from ..core.templates import Template
from ..templates.registry import TemplateRegistry


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON-serializable dictionary to disk."""
    target = Path(path)
    target.write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON dictionary from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_template_json(template: Template, path: str | Path) -> Path:
    """Serialize a template to JSON."""
    return save_json(template.to_dict(), path)


def load_template_json(path: str | Path) -> Template:
    """Load a template from JSON."""
    return Template.from_dict(load_json(path))


def save_config_json(config: SimulationConfig, path: str | Path) -> Path:
    """Serialize a simulation configuration to JSON."""
    return save_json(config.to_dict(), path)


def load_config_json(path: str | Path) -> SimulationConfig:
    """Load a simulation configuration from JSON."""
    return SimulationConfig.from_dict(load_json(path))


def save_registry_json(registry: TemplateRegistry, path: str | Path) -> Path:
    """Serialize a registry to JSON."""
    return save_json(registry.to_dict(), path)


def load_registry_json(path: str | Path) -> TemplateRegistry:
    """Load a template registry from JSON."""
    return TemplateRegistry.from_dict(load_json(path))


def save_network_json(network: CellularNetwork, path: str | Path) -> Path:
    """Serialize a network to JSON."""
    return save_json(network.to_dict(), path)


def load_network_json(path: str | Path) -> CellularNetwork:
    """Load a network from JSON."""
    return CellularNetwork.from_dict(load_json(path))
