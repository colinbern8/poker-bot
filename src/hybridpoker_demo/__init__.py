"""Phase 5: playable human-versus-AI heads-up games for the demonstration.

This package sits **outside** the frozen ``hybridpoker`` package. It never
changes the Phase 4 rules, agents or model; it only drives them:

* :mod:`hybridpoker_demo.human` and :mod:`hybridpoker_demo.recording` let the
  unchanged :func:`hybridpoker.simulator.play_hand` pause when the human must
  act and resume later by replaying the hand from the start (deterministic
  interrupt-and-replay).
* :mod:`hybridpoker_demo.controller` turns that into a heads-up match against
  one AI, with input checks and all-or-nothing state updates.
* :mod:`hybridpoker_demo.seeds` and :mod:`hybridpoker_demo.resources` provide
  seeds, the frozen model (SHA-256 checked) and cache warm-up.

Nothing here imports Streamlit; the user interface is a later stage.
"""
