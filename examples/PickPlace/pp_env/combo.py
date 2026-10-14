"""Pick-and-place task helpers: prompt sampling and combination validity.

Both functions take their per-setting data tables as explicit parameters
rather than relying on module globals; this lets them be reused by future
settings without monkey-patching.
"""

import random
from typing import Mapping, Sequence


def enrich_pick_prompt(class_name: str, prompts: Mapping[str, Sequence[str]]) -> str:
    """Return a random natural-language alias for `class_name`.

    Args:
        class_name: Pickable-object class name (e.g. "Can7up01").
        prompts:    Mapping from class name to a list of alias strings; one is
                    sampled uniformly using the current `random` state.
    """
    return random.sample(prompts[class_name], k=1)[0]


def is_valid_combination(comb, conflicts: Mapping[str, Sequence[str]]) -> bool:
    """True if no two members of `comb` conflict per the YAML rules.

    Args:
        comb:      Iterable of class names (e.g. a 4-tuple from itertools.combinations).
        conflicts: Mapping from class name to its conflicting class names.
                   A combination is invalid as soon as any pair is mutually listed.
    """
    for obj in comb:
        for conflict_obj in conflicts.get(obj, []):
            if conflict_obj in comb:
                return False
    return True
