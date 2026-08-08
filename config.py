#!/usr/bin/env python
"""
config.py — loads config.yaml into a plain dict, with light validation
against environment_multi.MARKER_ID_BY_SHAPE (the actual source of truth
for marker IDs — config.yaml's marker_ids section is documentation/
override surface, not a second source of truth, so a mismatch is an error
rather than something that silently diverges).

Usage:
    from config import load_config
    cfg = load_config()                       # config.yaml in cwd
    cfg = load_config("my_experiment.yaml")    # explicit path
"""
import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path: str = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)

    from environment_multi import MARKER_ID_BY_SHAPE
    for shape, marker_id in cfg.get("marker_ids", {}).items():
        if shape not in MARKER_ID_BY_SHAPE:
            raise ValueError(f"config.yaml marker_ids has unknown shape {shape!r}")
        if MARKER_ID_BY_SHAPE[shape] != marker_id:
            raise ValueError(
                f"config.yaml marker_ids[{shape!r}]={marker_id} does not match "
                f"environment_multi.MARKER_ID_BY_SHAPE[{shape!r}]={MARKER_ID_BY_SHAPE[shape]} "
                f"— these must stay in sync; fix config.yaml or environment_multi.py."
            )
    for shape in cfg["scene"]["object_shapes"]:
        if shape not in MARKER_ID_BY_SHAPE:
            raise ValueError(f"config.yaml scene.object_shapes has unknown shape {shape!r}")

    return cfg


def target_marker_ids_from_config(cfg: dict):
    """Resolves task.target_marker_ids (null -> every marker in
    scene.object_shapes, ascending) into a concrete list."""
    from environment_multi import MARKER_ID_BY_SHAPE
    explicit = cfg["task"].get("target_marker_ids")
    if explicit is not None:
        return list(explicit)
    return sorted(MARKER_ID_BY_SHAPE[s] for s in cfg["scene"]["object_shapes"])


if __name__ == "__main__":
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2))
    print("resolved target_marker_ids:", target_marker_ids_from_config(cfg))
