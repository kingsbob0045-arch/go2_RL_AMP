"""Make this checkout's go2_amp_isaaclab and rsl_rl win over the editable install.

go2_amp_isaaclab is pip-installed in editable mode pointing at whichever checkout was
installed from, which is not necessarily the one these scripts live in.  Import this
module first and the tree containing it is used instead, so a change can be tested before
it is merged anywhere.

    import _local_source  # noqa: F401  (must precede go2_amp_isaaclab)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Each entry is the directory that *contains* the importable package, not the project
# folder of the same name -- the outer folder has no __init__.py, so putting it on the
# path yields a namespace package whose __file__ is None.
for candidate in (ROOT / "isaaclab" / "source" / "go2_amp_isaaclab", ROOT / "rsl_rl"):
    if (candidate / candidate.name / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))


def report() -> None:
    import go2_amp_isaaclab

    loaded = Path(go2_amp_isaaclab.__file__).resolve()
    expected = (ROOT / "isaaclab/source/go2_amp_isaaclab/go2_amp_isaaclab/__init__.py").resolve()
    mark = "this checkout" if loaded == expected else f"ELSEWHERE (expected {expected})"
    print(f"go2_amp_isaaclab loaded from {loaded}  -> {mark}", flush=True)
