"""Single-call deterministic seeding across every RNG that matters.

Audit H-80: passing `--seed 1337` to the bench used to call
`random.seed(1337)` and call it a day -- numpy, torch (CPU + CUDA),
sentence-transformers (which routes through torch), the BGE reranker
(also torch), and HuggingFace `transformers.set_seed` were all untouched.
Two `--seed 1337` runs therefore did NOT produce identical retrieval
pools, identical rerank orders, or even identical embeddings under
mixed-precision math. Reproducibility was nominal.

`seed_everything(N)` is the single entry point. It walks the optional
deps and seeds whatever is importable; missing libraries are silently
skipped (transformers is in the bench extras, not core). Two calls
with the same `N` produce bit-identical state across every RNG we
touch.
"""

from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int) -> dict[str, bool]:
    """Seed every RNG we know about. Returns a `{lib_name: seeded?}` map.

    Order matches the dependency cone: stdlib first, numpy next, torch
    (which sentence-transformers + BGE reranker both depend on), then
    transformers. PYTHONHASHSEED is set for the current process but does
    not change Python's own hash randomization in-flight (that requires
    a process restart) -- we set it so child processes inherit it.
    """
    seeded: dict[str, bool] = {
        "python": False,
        "PYTHONHASHSEED": False,
        "numpy": False,
        "torch": False,
        "torch.cuda": False,
        "transformers": False,
    }

    random.seed(seed)
    seeded["python"] = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    seeded["PYTHONHASHSEED"] = True

    try:
        import numpy as np
        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass

    try:
        import torch
        torch.manual_seed(seed)
        seeded["torch"] = True
        cuda: Any = getattr(torch, "cuda", None)
        if cuda is not None and getattr(cuda, "is_available", lambda: False)():
            torch.cuda.manual_seed_all(seed)
            seeded["torch.cuda"] = True
    except ImportError:
        pass

    try:
        from transformers import set_seed as _hf_set_seed

        _hf_set_seed(seed)
        seeded["transformers"] = True
    except ImportError:
        pass

    return seeded
