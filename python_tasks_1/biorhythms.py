import math
from datetime import date, datetime, timedelta


def read_birth_date(s: str) -> date | None:
    """Read date in DD.MM.YYYY format."""
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except ValueError:
        return None


def days_lived(birth: date, today: date) -> int:
    return (today - birth).days


def bio(days: int, period: int) -> float:
    return math.sin(2 * math.pi * days / period)


def state_text(v: float, name: str) -> str:
    if v > 0.5:
        return f"{name}: strong phase"
    if v < -0.5:
        return f"{name}: low phase"
    return f"{name}: medium phase"


def trend_text(today_v: float, tomorrow_v: float, name: str) -> str:
    if tomorrow_v > today_v:
        return f"{name}: tomorrow should be better"
    if tomorrow_v < today_v:
        return f"{name}: tomorrow may be weaker"
    return f"{name}: no major change"


def main() -> None:
    print("Biorhythm calculator")
    name = input("Enter your name: ").strip()
    raw = input("Enter birth date (DD.MM.YYYY): ").strip()

    birth = read_birth_date(raw)
    if birth is None:
        print("Wrong date format")
        return

    today = date.today()
    if birth > today:
        print("Birth date cannot be in the future")
        return

    d_today = days_lived(birth, today)
    d_tomorrow = days_lived(birth, today + timedelta(days=1))

    periods: dict[str, int] = {
        "Physical": 23,
        "Emotional": 28,
        "Intellectual": 33,
    }

    print(f"\n{name}, you have lived {d_today} days")

    for rhythm_name, p in periods.items():
        v1 = bio(d_today, p)
        v2 = bio(d_tomorrow, p)

        print(f"\n{rhythm_name}: {v1:.3f}")
        print(state_text(v1, rhythm_name))
        print(trend_text(v1, v2, rhythm_name))


if __name__ == "__main__":
    main()
