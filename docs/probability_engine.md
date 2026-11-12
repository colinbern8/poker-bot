# The Probability Engine (Phase 1)

This document explains the math in `src/hybridpoker/equity.py` and
`src/hybridpoker/pot_odds.py` in plain language.

## 1. Equity: "what share of the pot do I win on average?"

Imagine the hand is played to the end with no more betting. For each
possible way the remaining cards could fall, the hero gets a **pot share**:

| Showdown outcome | Hero's pot share |
|---|---|
| Hero has the single best hand | 1 |
| Hero ties for best with *n − 1* other players (an *n*-way split) | 1 / *n* |
| Someone beats the hero | 0 |

**Equity** is the average pot share over all possible outcomes.

Example: with a royal flush on the board, everyone plays the board and the
pot is always split equally. Against 1 opponent, equity is 1/2. Against 2
opponents it is 1/3, and against 5 it is 1/6. In general it is
1 / (number of players).

Assumption: the opponents' hidden cards are random. Any two cards the hero
cannot see are equally likely. This is the baseline that the ML opponent
model will later improve on, by predicting which hands an opponent really
holds.

## 2. Monte Carlo simulation: estimating equity by sampling

There are too many possible outcomes to check them all before the flop
(about 2 billion for heads-up). So we **sample**:

1. Remove the cards we can see (our hole cards and the board) from a
   52-card deck.
2. Repeat `sample_count` times:
   1. Randomly deal 2 cards to each opponent and enough cards to complete
      the 5-card board.
   2. Use PokerKit to find every player's best 5-card hand.
   3. Record the hero's pot share (1, 1/*n*, or 0).
3. Equity estimate = average of the recorded shares.

It is like estimating how often a weighted coin lands heads by flipping it
thousands of times.

### Reproducibility (the seed)

The random dealing uses Python's `random.Random(seed)`. The same inputs and
the same seed always deal the same cards and give exactly the same answer.
A test checks this on both Linux and Windows. The seed is a **required**
argument, so every experiment records it.

### How accurate is the estimate? (standard error)

Each simulation's pot share is a number between 0 and 1. The **standard
error (SE)** measures how much the average would wobble if we re-ran the
simulation with a different seed:

```
SE = (sample standard deviation of the pot shares) / sqrt(sample_count)
```

Important details:

- The SE uses the **fractional pot shares**, not just win/lose. In a hand
  where the board plays, almost every outcome is a 0.5 split. Treating
  splits as wins or losses would badly misstate the uncertainty.
- Rule of thumb: the true equity is within about **±2 SE** of the
  estimate roughly 95% of the time.
- Accuracy improves slowly. To halve the SE you need **4×** as many
  samples. In typical heads-up spots, 1,000 samples give an SE of
  about ±1.5% and 5,000 samples about ±0.7%.

### Checking the simulation against the exact answer

On the river all 5 board cards are known. Against one opponent there are
only C(45, 2) = **990** possible opponent hands, so
`exact_river_equity` checks every one of them and gives the exact equity.
The tests require the Monte Carlo estimate to be within 4 SE of this
exact answer. This validates the simulation without relying on equity
figures copied from elsewhere.

Exact enumeration is limited to **one opponent**. Two opponents would need
about 450,000 combinations, which takes minutes at the measured speed.

## 3. Pot odds: "how often do I need to win to call?"

Project-wide definitions:

- **`pot_size`**: all chips in the middle, **including the opponent's
  bet**, but **not** the chips the hero would add by calling.
- **`call_amount`**: the extra chips the hero must add to call.

```
pot_odds = call_amount / (pot_size + call_amount)
```

The denominator is the size of the **final pot** after the call. Pot odds
are the **break-even equity**: the minimum share of the pot the hero must
win on average for calling to be worth it.

Worked example: the pot is 20 and the opponent bets 10. Then `pot_size`
is 30 (20 + 10) and `call_amount` is 10. The final pot is 40, so pot odds
are 10 / 40 = **25%**. A hand with more than 25% equity profits from
calling, under the assumptions below.

## 4. Simplified expected value (EV) of calling

```
EV(call) = equity × (pot_size + call_amount) − call_amount
         = equity × pot_size − (1 − equity) × call_amount
```

In words: the hero pays `call_amount`, and on average gets back `equity`
times the final pot. EV is measured in chips **compared with folding**,
which is worth 0.

Example: with 40% equity, a pot of 30, and a call of 10:
0.40 × 40 − 10 = **+6 chips** per call on average. With 20% equity the
result is 0.20 × 40 − 10 = **−2 chips**.

EV is exactly 0 when equity equals pot odds, which is why pot odds are
called the break-even point.

### Assumptions (state these in the presentation)

1. **No future betting.** The call ends the action, e.g. the opponent is
   all-in or it is the river. Real hands may have more bets later, which
   this formula ignores.
2. **Equity is fully realized.** The hero always reaches showdown and
   collects their share.
3. **Heads-up pot.** After the call, the final pot of
   `pot_size + call_amount` is contested by the hero and the bettor.
4. **No rake.**
5. **Sunk chips are ignored.** Chips the hero already put in are lost by
   folding too, so they do not count against calling.
6. **Random opponent hands.** The equity input assumes opponents hold
   random cards. Real opponents' betting tells us something about their
   hands. Using that information is the purpose of the ML opponent model.
