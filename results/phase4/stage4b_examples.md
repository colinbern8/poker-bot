# Stage 4B: example decisions (illustrative only)

Hand-picked spots, each decided by both EV agents with the same seed (identical equity). Bets are valued with the hero's equity against the opponent's estimated continuing range (strongest P(call)+P(re-raise) fraction of its possible hands). The ML agent has seen no hands of this opponent here (history = training priors). Not a tournament; not evidence of playing strength.

## Button, ace-king suited, preflop open

Pot 3, to call 1, board `-`.

- **ev_frequency** chose **all_in** (equity vs a random hand 0.630)
  - EVs: fold +0.00, check_call +1.52, raise_half_pot +0.78, raise_pot +0.63, all_in +47.62
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.623; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.623; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00 -> continues 0.94, equity if called 0.623; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)
- **ev_ml** chose **all_in** (equity vs a random hand 0.630)
  - EVs: fold +0.00, check_call +1.52, raise_half_pot +0.99, raise_pot +1.28, all_in +40.15
  - if raise_half_pot: opponent fold 0.03 / call 0.76 / raise 0.21 -> continues 0.97, equity if called 0.623; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.13 / call 0.71 / raise 0.16 -> continues 0.87, equity if called 0.616; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.16 / call 0.84 / raise 0.00 -> continues 0.84, equity if called 0.615; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)

## Button, seven-two offsuit, preflop open

Pot 3, to call 1, board `-`.

- **ev_frequency** chose **check_call** (equity vs a random hand 0.352)
  - EVs: fold +0.00, check_call +0.41, raise_half_pot -0.86, raise_pot -1.83, all_in -62.87
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.333; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.333; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00 -> continues 0.94, equity if called 0.330; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)
- **ev_ml** chose **check_call** (equity vs a random hand 0.352)
  - EVs: fold +0.00, check_call +0.41, raise_half_pot -0.77, raise_pot -1.19, all_in -57.64
  - if raise_half_pot: opponent fold 0.03 / call 0.76 / raise 0.21 -> continues 0.97, equity if called 0.335; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if raise_pot: opponent fold 0.13 / call 0.71 / raise 0.16 -> continues 0.87, equity if called 0.326; training coverage `preflop|facing_all_in=0|raises=1` = 1302 rows
  - if all_in: opponent fold 0.16 / call 0.84 / raise 0.00 -> continues 0.84, equity if called 0.325; training coverage `preflop|facing_all_in=1|raises=1` = 0 rows (LOW)

## Big blind, pair of queens, flop first to act

Pot 4, to call 0, board `Kc7h2d`.

- **ev_frequency** chose **all_in** (equity vs a random hand 0.766)
  - EVs: check_call +3.06, raise_half_pot +2.55, raise_pot +2.78, all_in +98.99
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.758; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.758; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00 -> continues 0.94, equity if called 0.757; training coverage `flop|facing_all_in=1|raises=1` = 55 rows
- **ev_ml** chose **all_in** (equity vs a random hand 0.766)
  - EVs: check_call +3.06, raise_half_pot +2.84, raise_pot +3.24, all_in +56.79
  - if raise_half_pot: opponent fold 0.18 / call 0.64 / raise 0.18 -> continues 0.82, equity if called 0.735; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if raise_pot: opponent fold 0.28 / call 0.59 / raise 0.14 -> continues 0.72, equity if called 0.713; training coverage `flop|facing_all_in=0|raises=1` = 1381 rows
  - if all_in: opponent fold 0.33 / call 0.67 / raise 0.00 -> continues 0.67, equity if called 0.702; training coverage `flop|facing_all_in=1|raises=1` = 55 rows

## Big blind, king-high flop with a weak hand, facing a half-pot bet

Pot 6, to call 2, board `Kc9h2d`.

- **ev_frequency** chose **fold** (equity vs a random hand 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -3.49, raise_pot -6.32, all_in -120.49
  - if raise_half_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.176; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.04 / call 0.71 / raise 0.25 -> continues 0.96, equity if called 0.176; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.06 / call 0.94 / raise 0.00 -> continues 0.94, equity if called 0.174; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)
- **ev_ml** chose **fold** (equity vs a random hand 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -3.01, raise_pot -5.05, all_in -102.07
  - if raise_half_pot: opponent fold 0.10 / call 0.66 / raise 0.24 -> continues 0.90, equity if called 0.170; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.17 / call 0.58 / raise 0.25 -> continues 0.83, equity if called 0.163; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.26 / call 0.74 / raise 0.00 -> continues 0.74, equity if called 0.144; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)

## History matters: the facing-a-bet spot after watching 8 deals of an opponent

The ML agent first watches a short match between a solid bot and the named opponent (public end-of-hand summaries only), then decides the same spot. Only the opponent-history features differ.

**After watching a nit** (55 of its actions):
- **ev_ml** chose **fold** (equity vs a random hand 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -2.06, raise_pot -1.82, all_in -70.50
  - if raise_half_pot: opponent fold 0.16 / call 0.76 / raise 0.08 -> continues 0.84, equity if called 0.165; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.42 / call 0.51 / raise 0.07 -> continues 0.58, equity if called 0.122; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.53 / call 0.47 / raise 0.00 -> continues 0.47, equity if called 0.100; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)

**After watching a maniac** (66 of its actions):
- **ev_ml** chose **fold** (equity vs a random hand 0.196)
  - EVs: fold +0.00, check_call -0.43, raise_half_pot -4.31, raise_pot -7.47, all_in -99.50
  - if raise_half_pot: opponent fold 0.02 / call 0.48 / raise 0.50 -> continues 0.98, equity if called 0.186; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if raise_pot: opponent fold 0.02 / call 0.49 / raise 0.49 -> continues 0.98, equity if called 0.183; training coverage `flop|facing_all_in=0|raises=2` = 209 rows
  - if all_in: opponent fold 0.28 / call 0.72 / raise 0.00 -> continues 0.72, equity if called 0.146; training coverage `flop|facing_all_in=1|raises=2` = 8 rows (LOW)
