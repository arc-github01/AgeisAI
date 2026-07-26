"""Verify the Phase 11 live demonstration end-to-end.

Runs every attack scenario through the real simulator service (generator
injectors -> process_injection -> StreamingEngine.process_event) and reports
what the SOC dashboard would show.

    python scripts/verify_live_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import live_state  # noqa: E402
from dashboard.simulator_service import SimulatorService  # noqa: E402
from dashboard.state import DashboardContext  # noqa: E402
from src.schema import ATTACK_CLASSES  # noqa: E402


def main() -> int:
    ctx = DashboardContext.build()
    service = SimulatorService(ctx)

    print("prerequisites:")
    for item in service.prerequisites():
        print(f"  [{'x' if item.ready else ' '}] {item.key:<20} phase {item.phase}")
    if not service.is_ready():
        print("\nSIMULATOR LOCKED - run the offline pipeline first.")
        return 1

    entity_ids = service.list_entities()
    target = next((e for e in entity_ids if e.startswith("USR")), entity_ids[0])
    print(f"\ntarget entity: {target}\n")

    failures = 0
    for attack in ATTACK_CLASSES:
        started = time.perf_counter()
        outcome = service.run(target, attack, intensity=4)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not outcome.success:
            failures += 1
            print(f"  FAIL  {attack:<28} {outcome.error}")
            continue

        results = outcome.result.results if outcome.result else []
        alerted = sum(1 for r in results if r.alerted)
        peak = max((r.risk_score for r in results), default=0.0)
        severity = max(
            (r.severity for r in results),
            key=lambda s: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s),
            default="LOW",
        )
        named = {r.predicted_attack_type for r in results if r.alerted}
        print(
            f"  ok    {attack:<28} events={len(outcome.events):<3} "
            f"alerts={alerted:<2} peak_risk={peak:5.1f} {severity:<8} "
            f"posted={outcome.alerts_posted} {elapsed_ms:6.0f}ms "
            f"named={sorted(named) if named else '-'}"
        )

    print(f"\nlive overlay alerts held for dashboard: {live_state.live_alert_count()}")
    print("FAILURES:" if failures else "ALL SCENARIOS PASSED", failures or "")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
