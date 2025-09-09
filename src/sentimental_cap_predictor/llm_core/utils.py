from __future__ import annotations

from typing import Any, Dict


def clean_generate_kwargs(**kwargs: Any) -> Dict[str, Any]:
    """Normalize keyword arguments for ``generate`` calls.

    Parameters
    ----------
    **kwargs: Any
        Arbitrary keyword arguments passed to
        :meth:`~transformers.PreTrainedModel.generate`.

    Returns
    -------
    Dict[str, Any]
        A new dictionary with conflicting or missing options adjusted.

    Notes
    -----
    - If ``max_new_tokens`` is provided, any ``max_length`` argument is
      removed.
    - If ``temperature`` is greater than zero and ``do_sample`` is not
      specified, ``do_sample`` is forced to ``True`` to ensure stochastic
      decoding.
    """

    cleaned = dict(kwargs)
    if "max_new_tokens" in cleaned and "max_length" in cleaned:
        cleaned.pop("max_length")
    temperature = cleaned.get("temperature")
    # fmt: off
    if (
        temperature is not None
        and temperature > 0
        and "do_sample" not in cleaned
    ):
        cleaned["do_sample"] = True
    # fmt: on
    return cleaned
