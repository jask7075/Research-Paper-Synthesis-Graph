"""Token accounting for LLM calls.

Provider-neutral and thread-safe, because extraction now runs sections concurrently
(`rpsg.extraction.extractor`) and several threads record into the same counters.

Tokens are always reported. Dollar amounts are reported ONLY if you configure rates
in `configs/settings.yaml` under `models.pricing` — the rates are not hardcoded
because a stale price table that silently under-reports spend is worse than no
price table at all. Look the current rates up for your provider and fill them in:

    models:
      pricing:
        gpt-5.4-nano: {input_per_mtok: 0.05, output_per_mtok: 0.40}

Usage is process-local: it measures one pipeline run, not a billing period. Treat
it as an estimate for planning, and the provider's dashboard as the truth.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_input_tokens += cached_input_tokens


@dataclass
class UsageTracker:
    """Accumulates token counts per model across a process."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _by_model: dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._by_model[model].add(input_tokens, output_tokens, cached_input_tokens)

    def reset(self) -> None:
        with self._lock:
            self._by_model.clear()

    def snapshot(self) -> dict[str, ModelUsage]:
        with self._lock:
            return {m: ModelUsage(u.calls, u.input_tokens, u.output_tokens, u.cached_input_tokens)
                    for m, u in self._by_model.items()}

    def summary(self) -> str:
        """Human-readable table. Includes cost only where a rate is configured."""
        from rpsg.config import get_settings

        snap = self.snapshot()
        if not snap:
            return "token usage: no LLM calls recorded"

        pricing = get_settings().models.pricing or {}
        lines = [
            f"{'model':22} {'calls':>7} {'in_tok':>11} {'out_tok':>10} {'cached':>9} {'cost':>10}",
            "-" * 74,
        ]
        total_cost, priced_all = 0.0, True
        for model, u in sorted(snap.items()):
            rate = pricing.get(model)
            if rate:
                cost = (
                    u.input_tokens / 1e6 * rate.get("input_per_mtok", 0.0)
                    + u.output_tokens / 1e6 * rate.get("output_per_mtok", 0.0)
                )
                total_cost += cost
                cost_str = f"${cost:,.2f}"
            else:
                priced_all = False
                cost_str = "n/a"
            lines.append(
                f"{model:22} {u.calls:>7,} {u.input_tokens:>11,} "
                f"{u.output_tokens:>10,} {u.cached_input_tokens:>9,} {cost_str:>10}"
            )
        if priced_all:
            lines.append("-" * 74)
            lines.append(f"{'TOTAL':22} {'':>7} {'':>11} {'':>10} {'':>9} ${total_cost:>9,.2f}")
        else:
            lines.append(
                "\n  cost 'n/a': add rates under models.pricing in configs/settings.yaml"
            )
        return "token usage\n" + "\n".join(lines)


#: Process-wide tracker. Adapters record into it; scripts print it when they finish.
USAGE = UsageTracker()