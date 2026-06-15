"""
Brick 9: Fractional Kelly bet sizing.

Given the model probability and offered decimal odds, returns the
recommended stake as a fraction of bankroll.

Uses fractional Kelly (default 0.25x) for safety, since full Kelly
is too aggressive for a model with estimation error.

Usage:
    python sizing.py
"""


def kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    kelly_cap: float = 0.25,
) -> float:
    """
    Compute fractional Kelly stake as a fraction of bankroll.

    Args:
        model_prob: model's estimated probability of the outcome
        decimal_odds: the decimal odds offered by the bookmaker
        kelly_cap: fraction of full Kelly to use (default 0.25 = quarter Kelly)

    Returns:
        Recommended stake as a fraction of bankroll (0.0 if no edge).
    """
    if model_prob <= 0 or model_prob >= 1 or decimal_odds <= 1:
        return 0.0

    # Full Kelly: f* = (b*p - q) / b
    # where b = decimal_odds - 1, p = model_prob, q = 1 - p
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = (b * model_prob - q) / b

    if full_kelly <= 0:
        return 0.0

    return full_kelly * kelly_cap


def kelly_stake(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    kelly_cap: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """
    Compute the dollar stake for a bet.

    Args:
        model_prob: model's estimated probability
        decimal_odds: decimal odds offered
        bankroll: current bankroll in dollars
        kelly_cap: fraction of full Kelly (default 0.25)
        max_stake_pct: hard cap on stake as fraction of bankroll (default 5%)

    Returns:
        Dollar amount to stake (0.0 if no edge).
    """
    frac = kelly_fraction(model_prob, decimal_odds, kelly_cap)
    frac = min(frac, max_stake_pct)
    return round(frac * bankroll, 2)


def main():
    print("Fractional Kelly Bet Sizing Examples\n")

    examples = [
        ("Strong edge", 0.55, 2.10, 1000),
        ("Moderate edge", 0.40, 3.00, 1000),
        ("Small edge", 0.35, 3.20, 1000),
        ("No edge", 0.30, 3.00, 1000),
        ("Negative edge", 0.20, 3.00, 1000),
        ("Long shot edge", 0.10, 15.00, 1000),
        ("Heavy favorite", 0.80, 1.30, 1000),
    ]

    print(f"{'Scenario':<20} {'Prob':>6} {'Odds':>6} {'Full K':>8} {'1/4 K':>8} {'Stake':>8} {'EV%':>7}")
    print("-" * 65)

    for label, prob, odds, bank in examples:
        b = odds - 1.0
        full_k = (b * prob - (1 - prob)) / b
        frac = kelly_fraction(prob, odds)
        stake = kelly_stake(prob, odds, bank)
        ev = (prob * odds - 1.0) * 100

        print(f"  {label:<18} {prob:>5.0%} {odds:>6.2f} {max(0,full_k):>7.2%} "
              f"{frac:>7.2%} ${stake:>6.2f} {ev:>+6.1f}%")

    print(f"\nAll stakes use quarter Kelly with 5% bankroll cap.")
    print("Stake is $0.00 when there is no edge (Kelly <= 0).")


if __name__ == "__main__":
    main()
