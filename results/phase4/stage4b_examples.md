# Stage 4B: example decisions (illustrative only)

Hand-picked spots, each decided by both EV agents with the same seed (identical equity). The ML agent has seen no hands of this opponent here (history = training priors). Not a tournament; not evidence of playing strength.

## Button, ace-king suited, preflop open

Pot 3, to call 1, board `-`.

- **ev_frequency** chose **all_in** (equity 0.630)
  - EVs: fold +0.00, check_call +1.52, raise_half_pot +0.83, raise_pot +0.69, all_in +50.11
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)
- **ev_ml** chose **all_in** (equity 0.630)
  - EVs: fold +0.00, check_call +1.52, raise_half_pot +1.03, raise_pot +1.40, all_in +45.20
  - if raise_half_pot: opponent fold 0.03 / call 0.76 / raise 0.21; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.13 / call 0.71 / raise 0.16; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.16 / call 0.84 / raise 0.00; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)

## Button, seven-two offsuit, preflop open

Pot 3, to call 1, board `-`.

- **ev_frequency** chose **check_call** (equity 0.352)
  - EVs: fold +0.00, check_call +0.41, raise_half_pot -0.75, raise_pot -1.67, all_in -54.66
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)
- **ev_ml** chose **check_call** (equity 0.352)
  - EVs: fold +0.00, check_call +0.41, raise_half_pot -0.67, raise_pot -0.97, all_in -48.66
  - if raise_half_pot: opponent fold 0.03 / call 0.76 / raise 0.21; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.13 / call 0.71 / raise 0.16; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.16 / call 0.84 / raise 0.00; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)

## Big blind, pair of queens, flop first to act

Pot 4, to call 0, board `Kc7h2d`.

- **ev_frequency** chose **all_in** (equity 0.766)
  - EVs: check_call +3.06, raise_half_pot +2.60, raise_pot +2.85, all_in +102.37
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00; training coverage `flop|facing_all_in=1|raises=1` = 55 rows
- **ev_ml** chose **all_in** (equity 0.766)
  - EVs: check_call +3.06, raise_half_pot +3.00, raise_pot +3.62, all_in +73.79
  - if raise_half_pot: opponent fold 0.18 / call 0.64 / raise 0.18; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if raise_pot: opponent fold 0.28 / call 0.59 / raise 0.14; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if all_in: opponent fold 0.33 / call 0.67 / raise 0.00; training coverage `flop|facing_all_in=1|raises=1` = 55 rows

## Big blind, king-high flop with a weak hand, facing a half-pot bet

Pot 6, to call 2, board `Kc9h2d`.

- **ev_frequency** chose **fold** (equity 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -3.26, raise_pot -5.98, all_in -112.34
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)
- **ev_ml** chose **fold** (equity 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -2.74, raise_pot -4.59, all_in -86.65
  - if raise_half_pot: opponent fold 0.10 / call 0.66 / raise 0.24; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.17 / call 0.58 / raise 0.25; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.26 / call 0.74 / raise 0.00; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)

## History matters: the facing-a-bet spot after watching 8 deals of an opponent

The ML agent first watches a short match between a solid bot and the named opponent (public end-of-hand summaries only), then decides the same spot. Only the opponent-history features differ.

**After watching a nit** (55 of its actions):
- **ev_ml** chose **fold** (equity 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -1.67, raise_pot -0.91, all_in -52.53
  - if raise_half_pot: opponent fold 0.16 / call 0.76 / raise 0.08; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.42 / call 0.51 / raise 0.07; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.53 / call 0.47 / raise 0.00; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)

**After watching a maniac** (66 of its actions):
- **ev_ml** chose **fold** (equity 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -4.23, raise_pot -7.32, all_in -85.05
  - if raise_half_pot: opponent fold 0.02 / call 0.48 / raise 0.50; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.02 / call 0.49 / raise 0.49; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.28 / call 0.72 / raise 0.00; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)
