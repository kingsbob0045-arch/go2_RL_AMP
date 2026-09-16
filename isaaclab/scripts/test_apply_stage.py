#!/usr/bin/env python3
"""Check that --stage disables exactly the difficulty groups above the named stage.

train.py cannot be imported without booting Isaac Sim, so apply_stage is lifted out of the
source with ast and run against a stub config.  Worth having: a stage that silently leaves a
group enabled makes a whole curriculum arm meaningless, and the run still looks healthy.

    python isaaclab/scripts/test_apply_stage.py
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

WANTED = {"apply_stage", "STAGES"}


def _load_apply_stage() -> types.ModuleType:
    tree = ast.parse((Path(__file__).with_name("train.py")).read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in WANTED)]
    assert len(body) == len(WANTED), f"expected {WANTED} in train.py, found {len(body)} of them"
    module = types.ModuleType("stage_only")
    exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), module.__dict__)
    return module


class _Box:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _full_difficulty_cfg() -> _Box:
    """A stub carrying only the fields apply_stage touches, at full difficulty."""
    return _Box(
        terrain=_Box(terrain_type="generator", terrain_generator=_Box()),
        events=_Box(
            link_mass=_Box(), link_com=_Box(), joint_parameters=_Box(),
            actuator_gains=_Box(params={"stiffness_distribution_params": (0.8, 1.2),
                                        "damping_distribution_params": (0.8, 1.2)}),
        ),
        gravity_bias=0.03,
        action_latency_steps=(0, 1),
        observation_latency_steps=(0, 0),
    )


EXPECTED = {
    "core":     dict(terrain="plane",     latency=(0, 0), bias=0.0,  mass=False, gains=(0.9, 1.1)),
    "latency":  dict(terrain="plane",     latency=(0, 1), bias=0.0,  mass=False, gains=(0.8, 1.2)),
    "dynamics": dict(terrain="plane",     latency=(0, 1), bias=0.03, mass=True,  gains=(0.8, 1.2)),
    "terrain":  dict(terrain="generator", latency=(0, 1), bias=0.03, mass=True,  gains=(0.8, 1.2)),
}


def test_stages() -> None:
    module = _load_apply_stage()
    assert module.STAGES == tuple(EXPECTED), f"stage list changed: {module.STAGES}"
    for stage, want in EXPECTED.items():
        cfg = _full_difficulty_cfg()
        module.apply_stage(cfg, stage)
        got = dict(
            terrain=cfg.terrain.terrain_type,
            latency=cfg.action_latency_steps,
            bias=cfg.gravity_bias,
            mass=cfg.events.link_mass is not None,
            gains=cfg.events.actuator_gains.params["stiffness_distribution_params"],
        )
        assert got == want, f"stage {stage!r}: {got} != {want}"


if __name__ == "__main__":
    test_stages()
    print("apply_stage: all stages disable exactly the groups above them")
