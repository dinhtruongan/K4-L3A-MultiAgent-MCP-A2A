"""Typed state shared by the coordinator and specialist agents.

Everything here is in-memory and scoped to exactly one ``case_id``. Nothing is shared
between cases, so an evidence reference can never leak into another case's output.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

MAX_HOPS = 12

_message_counter = itertools.count(1)


@dataclass(frozen=True)
class Evidence:
    """One validated MCP evidence envelope, exactly as returned by the gateway."""

    ref: str
    tool: str
    domain: str
    data: Any
    actor: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentMessage:
    """A2A envelope. ``case_id`` is the correlation key; ``hop`` bounds the conversation."""

    case_id: str
    sender: str
    recipient: str
    intent: str
    hop: int
    payload: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: f"msg_{next(_message_counter):06d}")


@dataclass
class Finding:
    """Result a specialist hands back to the coordinator."""

    agent: str
    facts: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def refs(self) -> list[str]:
        return [item.ref for item in self.evidence]


@dataclass
class Decision:
    """Policy-agent decision, later checked (and possibly revised) by the verifier."""

    primary_issue: str
    case_status: str
    confidence: float
    ranked_causes: list[str]
    responsible_parties: list[tuple[str, str | None]]
    refund_lines: list[tuple[str, float, str | None]]
    resolution_actions: list[str]
    evidence_domains: list[str]
    rule: str
    data_conflicts: list[dict[str, Any]] = field(default_factory=list)
    confidence_notes: list[str] = field(default_factory=list)

    @property
    def refund_total(self) -> float:
        return round(sum(amount for _, amount, _ in self.refund_lines), 2)


@dataclass
class CaseState:
    case_id: str
    case: dict[str, Any]
    hints: dict[str, Any] = field(default_factory=dict)
    findings: dict[str, Finding] = field(default_factory=dict)
    ledger: dict[str, Evidence] = field(default_factory=dict)
    decision: Decision | None = None
    verifier_notes: list[str] = field(default_factory=list)
    hops: int = 0

    def record(self, finding: Finding) -> None:
        self.findings[finding.agent] = finding
        for item in finding.evidence:
            self.ledger[item.ref] = item

    def evidence_for(self, *domains: str) -> list[Evidence]:
        wanted = set(domains)
        return [item for item in self.ledger.values() if item.domain in wanted]

    def facts(self, agent: str) -> dict[str, Any]:
        finding = self.findings.get(agent)
        return finding.facts if finding else {}
