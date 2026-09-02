from __future__ import annotations

import time


class CountProgress:
    """Print completed/total progress whenever a count interval is crossed."""

    def __init__(self, total: int, every: int):
        if every <= 0:
            raise ValueError("progress_every must be greater than zero.")
        self.total = total
        self.every = every
        self.next_report = every
        self.started_at = time.time()

    def update(self, completed: int, suffix: str = "") -> None:
        is_final = completed >= self.total
        if completed < self.next_report and not is_final:
            return

        elapsed = time.time() - self.started_at
        speed = completed / max(elapsed, 1e-9)
        remaining_minutes = (self.total - completed) / max(speed, 1e-9) / 60
        extra = f"  {suffix}" if suffix else ""
        print(
            f"progress: {completed}/{self.total}  {speed:.0f} img/s  "
            f"ETA {remaining_minutes:.1f} min{extra}"
        )
        while self.next_report <= completed:
            self.next_report += self.every
