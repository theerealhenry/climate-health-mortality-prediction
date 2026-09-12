"""Shared, cross-cutting utilities used by more than one `features` module.

Extracted during Stage 6 from `climate.py` (where these three checks originated,
Stage 8) once a second and third module (`temporal.py`, `demographic.py`) needed the
exact same column/null/collision validation — duplicating them a third and fourth
time would have made the four copies drift silently, which is exactly the kind of
thing this project's own reviews have been catching in other forms."""
