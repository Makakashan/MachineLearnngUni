import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

h: float = 100.0
v0: float = 50.0
g: float = 9.81
PLOT_DIR = Path(__file__).resolve().parent / "output" / "trajectories"
PLOT_FILE = PLOT_DIR / "trajektoria.png"


def calc_range(angle_deg: float) -> float | None:
    """Calculate projectile range for a given angle."""
    a = math.radians(angle_deg)
    c = math.cos(a)

    if c == 0:
        return None

    k1 = -g / (2 * (v0 ** 2) * (c ** 2))
    k2 = math.tan(a)
    k3 = h

    d = k2 * k2 - 4 * k1 * k3
    if d < 0:
        return None

    x1 = (-k2 + math.sqrt(d)) / (2 * k1)
    x2 = (-k2 - math.sqrt(d)) / (2 * k1)

    good: list[float] = []
    if x1 > 0:
        good.append(x1)
    if x2 > 0:
        good.append(x2)

    if len(good) == 0:
        return None

    return max(good)


def find_best_angle() -> tuple[float, float]:
    """Simple angle scan to find near-maximum range."""
    best_a = 0.0
    best_x = 0.0

    cur = 0.1
    while cur < 89.9:
        x = calc_range(cur)
        if x is not None and x > best_x:
            best_x = x
            best_a = cur
        cur += 0.1

    return best_a, best_x


def draw_plot(angle_deg: float, hit_x: float, target_x: float) -> None:
    """Draw trajectory and save it as PNG."""
    a = math.radians(angle_deg)
    c = math.cos(a)

    k1 = -g / (2 * (v0 ** 2) * (c ** 2))
    k2 = math.tan(a)

    xs: list[float] = []
    ys: list[float] = []

    steps = 300
    dx = hit_x / steps

    x = 0.0
    for _ in range(steps + 1):
        y = k1 * x * x + k2 * x + h
        xs.append(x)
        ys.append(y)
        x += dx

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, label=f"Trajectory, angle {angle_deg:.1f}°")
    plt.axhline(0, color="black")
    plt.axvline(target_x, color="red", linestyle="--", label=f"Target {target_x:.2f} m")
    plt.scatter([hit_x], [0], color="green", label=f"Hit point {hit_x:.2f} m")
    plt.title("Trebuchet shot")
    plt.xlabel("x, meters")
    plt.ylabel("y, meters")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()


def main() -> None:
    target: float = random.uniform(50, 340)
    best_a, best_x = find_best_angle()
    tries: int = 0

    print("Game: trebuchet target practice")
    print("Enter an angle from 0 to 90 degrees.")
    print("Hit tolerance: +- 5 m")
    print(f"Model max range: {best_x:.2f} m (around {best_a:.1f}°)")

    while True:
        s: str = input("Enter angle: ").strip().replace(",", ".")

        try:
            angle = float(s)
        except ValueError:
            print("Please enter a number")
            continue

        if angle <= 0 or angle >= 90:
            print("Angle must be in range (0, 90)")
            continue

        x: float | None = calc_range(angle)
        tries += 1

        if x is None:
            print("Calculation error, try another angle")
            continue

        diff = x - target
        print(f"Projectile landed at {x:.2f} m")

        if abs(diff) <= 5:
            print(f"Hit! Target was at {target:.2f} m")
            print(f"Attempts: {tries}")
            draw_plot(angle, x, target)
            print(f"Plot saved as {PLOT_FILE}")
            break

        if diff < 0:
            if angle < best_a:
                print("Short shot. You can increase the angle")
            else:
                print("Short shot. You can decrease the angle")
        else:
            if angle < best_a:
                print("Overshot. You can decrease the angle")
            else:
                print("Overshot. You can increase the angle")


if __name__ == "__main__":
    main()
