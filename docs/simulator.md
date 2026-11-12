# The Heads-Up Simulator (Phase 2)

This document explains, in plain language, how HybridPoker plays and
measures heads-up poker matches.

## 1. The game

| Setting | Value |
|---|---|
| Variant | Heads-up (2-player) No-Limit Texas Hold'em |
| Blinds | 1 / 2 chips (small / big) |
| Antes, rake | None |
| Starting stacks | 200 chips = 100 big blinds, **reset every hand** |

Resetting stacks makes every hand an independent experiment, so a lucky
early hand cannot affect later ones.

**Seats.** PokerKit seats heads-up players as seat 0 = big blind and
seat 1 = button (small blind). The button acts first before the flop and
last after it.

## 2. Who does what

| Part | Job | File |
|---|---|---|
| PokerKit | Enforces the rules: betting order, minimum raises, all-ins, pot and stack updates, showdown payouts | (library) |
| Simulator | Deals from a seeded deck, asks agents for decisions, records everything | `simulator.py` |
| Actions | Turns the five abstract choices into legal chip amounts | `actions.py` |
| Observation | What the acting player may see | `observation.py` |
| Agents | Choose actions | `agents/` |
| Match runner | Plays many deals, swapping seats | `match.py` |
| Metrics | bb/100 and confidence intervals | `metrics.py` |
| Hand log | CSV files for analysis and the Phase 3 dataset | `hand_log.py` |

## 3. The five actions

Every agent chooses one of five **abstract action categories**:

| Category | Meaning |
|---|---|
| `fold` | Give up. Only offered when facing a bet, because folding when you can check for free is never better. |
| `check_call` | Check, or call the current bet. |
| `raise_half_pot` | Bet or raise by half the pot. |
| `raise_pot` | Bet or raise by the whole pot. |
| `all_in` | Bet or raise everything (up to what the opponent can match). |

**Raise sizes** use the standard "pot-size raise" rule: first imagine
calling, then add a fraction of the resulting pot.

```
raise_to = largest_bet + fraction × (pot_size + to_call)
```

Example: before the flop, the pot is 3 (blinds 1 + 2) and the button
needs 1 more to call. A pot-size raise goes to 2 + 1 × (3 + 1) = **6**,
and a half-pot raise goes to 2 + 0.5 × 4 = **4**.

The amount is rounded to whole chips and kept inside PokerKit's legal
range. If a pot-size raise would need more chips than the player has, it
becomes an all-in. The log records both the **category the agent chose**
and the **actual chips moved**.

## 4. Reproducibility: seeds

One **master seed** controls a whole match. Every source of randomness gets
its own seed derived from it with SHA-256, for example
`derive_seed(master, "deal", 17)` for deal 17's deck. The results are
identical on every computer, and the tests check this on Linux and Windows.

The simulator shuffles its **own** deck and hands PokerKit the exact cards
to deal. PokerKit's built-in dealing could not be seeded.

## 5. The agents

| Agent | How it plays |
|---|---|
| **Probability** (baseline hero) | Uses the Phase 1 math. Facing a bet: raise the pot if equity ≥ 0.70, call if equity ≥ pot odds + 0.02, otherwise fold. With no bet: bet half the pot if equity ≥ 0.60, otherwise check. Every decision is logged with its reason, e.g. `call: equity 0.42 >= pot odds 0.25 + margin 0.02`. |
| **Random** | Picks uniformly among the legal categories. |
| **Calling Station** | Always checks or calls. Never folds, never raises. |
| **Maniac** | Raises and bluffs very often and calls wide. |
| **Nit** | Plays only strong hands, rarely raises, never bluffs. |
| **Solid** | Calls close to break-even, raises good hands, bluffs occasionally. |

The four styles are presets of one `StyleAgent` with three settings
between 0 and 1:

- **looseness:** calls when equity ≥ pot odds × 2 × (1 − looseness).
- **aggression:** how often it bets or raises strong hands, and how low its
  bar for "strong" is.
- **bluff_rate:** how often it bets or raises regardless of its cards.

| Preset | looseness | aggression | bluff_rate |
|---|---|---|---|
| calling_station | 1.00 | 0.00 | 0.00 |
| maniac | 0.75 | 0.85 | 0.35 |
| nit | 0.10 | 0.30 | 0.00 |
| solid | 0.45 | 0.60 | 0.08 |

All agents measure hand strength as **equity against one random hand**.
The probability agent uses 500 Monte Carlo samples and the bots use 200,
which makes the bots deliberately noisy. Preflop equities are cached for
each of the 169 starting-hand classes (e.g. `AKs`, `72o`), so they are
computed only once.

The bot settings are **private**. They are stored in the run's
`metadata.json`, never in the per-decision rows that a model will learn
from.

## 6. Seat-swapped ("duplicate") matches

Poker results are dominated by luck: who gets the good cards. To cancel
much of that luck, every deal is played **twice from the same seed**:

1. The hero is on the button and the villain is in the big blind.
2. The seats are swapped, with the **same cards for each seat**.

Both agents therefore play both sides of every deal. If the hero wins only
because seat 1 had aces, the villain gets those same aces in the second
hand.

A check that demonstrates this: when two Calling Stations play each other,
every pair of hands nets **exactly zero**. A test confirms it.

## 7. Win rate: bb/100 with a 95% confidence interval

**bb/100** means big blinds won per 100 hands, the standard poker win-rate
unit.

The two hands of a deal pair share cards, so their results are
**correlated**. Treating them as independent would misstate the
uncertainty. The confidence interval therefore uses the **deal pair** as
its unit:

1. For each deal, add up the hero's chips over both seatings.
2. Convert to bb/100: `pair_net / 2 hands / 2 chips per big blind × 100`.
3. bb/100 is the average over pairs, and the standard error is the sample
   standard deviation divided by √(number of pairs).
4. 95% CI = bb/100 ± 1.96 × standard error. This is a normal approximation,
   which is fine for hundreds of pairs.

How to read it: if the whole interval is above 0, the hero is very likely
a winner against that opponent. If the interval contains 0, the match was
too short to tell.

The script also prints a "naive" standard error that treats every hand as
independent. It is shown only for comparison and should not be used for
conclusions.

## 8. The hand log

`python scripts/run_match.py` writes three files per match to
`data/raw/simulations/<run>/`, a folder that is not committed to git:

- `decisions.csv`: one row per decision, with the game state **just
  before** the decision.
- `hands.csv`: one row per hand (cards, board, payouts, showdown).
- `metadata.json`: seeds, game settings, agent settings, software
  versions and the summary result.

The decision columns are grouped by prefix, so it is always clear what a
model may use:

| Prefix | Contents | May a model use it? |
|---|---|---|
| `id_` | Hand, deal and decision numbers, seeds | No (identifiers) |
| `obs_` | Street, position, board, pot, amount to call, pot odds, stacks, effective stack, bets, legal-action mask and amounts, betting history | **Yes**: public at decision time |
| `act_` | Category chosen, `bet_to`, chips added, all-in flag | It is the **target** to predict |
| `hidden_` | Hole cards, the agent's equity estimate and reason | No (private) |
| `label_` | Which agent acted (e.g. `maniac`), hero/villain role | Evaluation only |
| `out_` | Net chips won, showdown, final board | No (future information) |

## 9. Simplifying assumptions and limitations

- **Heads-up only**, with fixed 100 BB stacks, no rake, and stacks reset
  every hand.
- **Five abstract actions.** Real players choose any bet size.
- **Random-hand equity.** Every agent ignores what the opponent's actions
  reveal. The ML opponent model is meant to fix that for the hero.
- **Synthetic opponents.** The bots follow rules we wrote, so a model
  trained on them learns *our* rules. This is stated openly. It also lets
  us check a learned model against each bot's known behavior.
- **Thresholds chosen by reasoning.** The probability agent's thresholds
  (0.70, 0.60, margin 0.02) were not tuned on results.
- **Speed.** Equity is recomputed on every decision after the flop, which
  limits matches to a few hands per second (see `docs/progress.md`).
