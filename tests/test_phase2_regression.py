"""Regression guard for Phase 4 changes to Phase 2 code.

The SHA-256 hashes below were recorded from the Phase 2/3 code (commit
a79526f) **before** the Phase 4 refactor of ``hand_log`` and the
end-of-hand notification were added. Any change to gameplay, action
legality, chip amounts, payouts or log formatting changes these bytes.
"""

import hashlib
from pathlib import Path

import pytest

from hybridpoker.agents import make_agent
from hybridpoker.hand_log import write_run
from hybridpoker.match import run_match

GOLDEN = {
    ("probability", "maniac", 10, 21): {
        "decisions": "2c1dda84b1a2dcf5d3bff9c3ced83de38d79902c7f80e888ccfe332199e2f9ad",
        "hands": "8579bad7912640e35260a548237167959897a40aa5e814e994fa6bd19f2489b9",
    },
    ("random", "solid", 12, 5): {
        "decisions": "9ec32b59ee5c33e001d7241dfe36c5338707c0721283e44f7255ddf08358ec6f",
        "hands": "1db6ad1600a8f565010d639f0096530e3058d904accbae07f65d1b23660efeb3",
    },
    ("nit", "calling_station", 8, 77): {
        "decisions": "ba4896362c41e45cc7986d93dc67945d95143e97cbcc77132a51bd4a927e2b83",
        "hands": "5959c30cc3d0e05004e2dcdc9a8772da7ae93762117cb8ff069d5adbdfedd610",
    },
}


@pytest.mark.parametrize("case", list(GOLDEN))
def test_logs_are_byte_identical_to_phase2(case, tmp_path: Path) -> None:
    hero_name, villain_name, deals, seed = case
    hero, villain = make_agent(hero_name, 30), make_agent(villain_name, 30)
    match = run_match(hero, villain, deal_count=deals, seed=seed)
    paths = write_run(match, hero, villain, tmp_path)
    for kind in ("decisions", "hands"):
        assert hashlib.sha256(paths[kind].read_bytes()).hexdigest() == GOLDEN[case][kind], kind
