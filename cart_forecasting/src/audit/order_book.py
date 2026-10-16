"""Explicit replenishment order book with lifecycle tracking.

The original simulator tracked pending retrievals in an anonymous list and
blocked *any* new decision while that list was non-empty. That prevents double
counting but also prevents a legitimate additional top-up, and it leaves no
auditable record of order identity, status or delivered quantity.

This module implements the order lifecycle required by the audit:

    PENDING -> COMPLETED            (delivered after the lead time)
    PENDING -> CANCELLED            (no longer required / infeasible)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

LOGGER = logging.getLogger(__name__)

PENDING = "PENDING"
COMPLETED = "COMPLETED"
CANCELLED = "CANCELLED"


@dataclass
class ReplenishmentOrder:
    order_id: str
    decision_time: pd.Timestamp
    quantity: int
    expected_completion_time: pd.Timestamp
    status: str = PENDING
    delivered_quantity: float = 0.0
    delivered_time: pd.Timestamp | None = None
    delivery_record_time: pd.Timestamp | None = None
    cancelled_reason: str | None = None
    origin_index: int | None = None
    urgency: str | None = None
    risk_at_decision: float | None = None
    predicted_min_at_decision: float | None = None
    capacity_rejected: float = 0.0

    def as_row(self) -> dict:
        return {
            "order_id": self.order_id,
            "decision_time": self.decision_time,
            "quantity": self.quantity,
            "expected_completion_time": self.expected_completion_time,
            "status": self.status,
            "delivered_quantity": self.delivered_quantity,
            "delivered_time": self.delivered_time,
            "delivery_record_time": self.delivery_record_time,
            "delivery_delay_min": (
                (self.delivery_record_time - self.expected_completion_time).total_seconds() / 60.0
                if self.delivery_record_time is not None
                else None
            ),
            "capacity_rejected": self.capacity_rejected,
            "cancelled_reason": self.cancelled_reason,
            "origin_index": self.origin_index,
            "urgency": self.urgency,
            "risk_at_decision": self.risk_at_decision,
            "predicted_min_at_decision": self.predicted_min_at_decision,
        }


@dataclass
class OrderBook:
    """Chronological order ledger. All quantities are whole carts."""

    orders: list = field(default_factory=list)
    _seq: int = 0

    # ---------------------------------------------------------------- create
    def place(
        self,
        decision_time: pd.Timestamp,
        quantity: int,
        lead_time_min: float,
        **meta,
    ) -> ReplenishmentOrder:
        self._seq += 1
        order = ReplenishmentOrder(
            order_id=f"ORD-{self._seq:04d}",
            decision_time=pd.Timestamp(decision_time),
            quantity=int(quantity),
            expected_completion_time=pd.Timestamp(decision_time)
            + pd.Timedelta(minutes=float(lead_time_min)),
            **meta,
        )
        self.orders.append(order)
        return order

    # ----------------------------------------------------------------- state
    @property
    def pending(self) -> list:
        return [o for o in self.orders if o.status == PENDING]

    @property
    def completed(self) -> list:
        return [o for o in self.orders if o.status == COMPLETED]

    def pending_quantity(self) -> float:
        """Carts already ordered but not yet delivered.

        The optimiser subtracts this from the computed requirement so that the
        same shortfall is never ordered twice.
        """
        return float(sum(o.quantity for o in self.pending))

    def pending_arriving_before(self, ts: pd.Timestamp) -> float:
        return float(
            sum(o.quantity for o in self.pending if o.expected_completion_time <= ts)
        )

    # --------------------------------------------------------------- lifecycle
    def deliver_due(self, record_time: pd.Timestamp, record_start_time: pd.Timestamp) -> list:
        """Complete orders whose lead time elapsed before the record's interval.

        A delivery at 10:10 cannot influence a cart measurement aggregated over
        an interval that *ended* at 10:10. Orders are therefore only credited to
        records whose interval **starts** at or after the expected completion
        time, which removes the sub-interval look-ahead present in the original
        simulator.
        """
        delivered = []
        for o in self.pending:
            if o.expected_completion_time <= record_start_time:
                o.status = COMPLETED
                o.delivered_quantity = float(o.quantity)
                o.delivered_time = o.expected_completion_time
                o.delivery_record_time = pd.Timestamp(record_time)
                delivered.append(o)
        return delivered

    def cancel(self, order: ReplenishmentOrder, reason: str) -> None:
        if order.status == PENDING:
            order.status = CANCELLED
            order.cancelled_reason = reason

    def to_frame(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame(
                columns=list(
                    ReplenishmentOrder("", pd.Timestamp("2000-01-01"), 0,
                                       pd.Timestamp("2000-01-01")).as_row().keys()
                )
            )
        return pd.DataFrame([o.as_row() for o in self.orders])

    def audit(self) -> pd.DataFrame:
        """Per-order invariant checks used by pending_orders_audit.csv."""
        rows = []
        for o in self.orders:
            lead_ok = o.expected_completion_time > o.decision_time
            delivery_ok = (
                o.delivery_record_time is None
                or o.delivery_record_time >= o.expected_completion_time
            )
            qty_ok = isinstance(o.quantity, int) and o.quantity > 0
            status_ok = o.status in (PENDING, COMPLETED, CANCELLED)
            delivered_ok = (
                (o.status == COMPLETED and o.delivered_quantity == o.quantity)
                or (o.status != COMPLETED and o.delivered_quantity == 0.0)
            )
            rows.append(
                {
                    **o.as_row(),
                    "check_lead_time_positive": lead_ok,
                    "check_delivery_not_early": delivery_ok,
                    "check_quantity_valid": qty_ok,
                    "check_status_valid": status_ok,
                    "check_delivered_matches_order": delivered_ok,
                    "all_checks_pass": bool(
                        lead_ok and delivery_ok and qty_ok and status_ok and delivered_ok
                    ),
                }
            )
        return pd.DataFrame(rows)


def overlap_audit(book: OrderBook) -> pd.DataFrame:
    """Detect orders issued while an earlier order was still in transit."""
    rows = []
    orders = sorted(book.orders, key=lambda o: o.decision_time)
    for i, o in enumerate(orders):
        overlapping = [
            p.order_id
            for p in orders[:i]
            if p.status != CANCELLED and p.expected_completion_time > o.decision_time
        ]
        rows.append(
            {
                "order_id": o.order_id,
                "decision_time": o.decision_time,
                "quantity": o.quantity,
                "expected_completion_time": o.expected_completion_time,
                "status": o.status,
                "overlapping_orders": ";".join(overlapping),
                "n_overlapping": len(overlapping),
                "pending_quantity_at_decision": float(
                    sum(
                        p.quantity
                        for p in orders[:i]
                        if p.status != CANCELLED and p.expected_completion_time > o.decision_time
                    )
                ),
            }
        )
    return pd.DataFrame(rows)
