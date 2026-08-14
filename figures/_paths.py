"""Put the repository root and the sibling script folders on ``sys.path``.

The project is a set of flat modules, not an installed package, so a script
living in ``figures/`` cannot import ``images_generator`` or ``pbr_solver``
(which sit at the root) without this. Import it before any project import:

    import _paths  # noqa: F401
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

for _p in (_ROOT, _ROOT / "scripts", _ROOT / "figures"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
