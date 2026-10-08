"""Guards: simultaneous approve+reject on one pending order -> exactly one
transition succeeds (the loser matches nothing == the route's 409), and the
order is acted on at most once. Models the conditional update both routes issue
(UPDATE ... WHERE status='pending'), which is the atomicity guarantee."""

import threading

from fakes import FakeSupabase


def _transition(sb, oid, new_status):
    # Mirrors approve/reject routes: PATCH gated on status=eq.pending.
    res = (
        sb.table("trader_pending_orders")
        .update({"status": new_status})
        .eq("id", oid)
        .eq("status", "pending")
        .execute()
    )
    return res.data


def test_approve_reject_exactly_one_wins():
    sb = FakeSupabase()
    sb.seed("trader_pending_orders", [{"id": "o1", "status": "pending"}])
    outcomes = {}

    def approve():
        outcomes["approve"] = _transition(sb, "o1", "approved")

    def reject():
        outcomes["reject"] = _transition(sb, "o1", "rejected")

    ta, tr = threading.Thread(target=approve), threading.Thread(target=reject)
    ta.start()
    tr.start()
    ta.join()
    tr.join()

    wins = [k for k, v in outcomes.items() if v]
    assert len(wins) == 1  # exactly one transition succeeded
    loser = "reject" if wins[0] == "approve" else "approve"
    assert outcomes[loser] == []  # loser matched nothing == 409-equivalent
    assert sb.tables["trader_pending_orders"][0]["status"] in ("approved", "rejected")


def test_double_reject_no_double_act():
    sb = FakeSupabase()
    sb.seed("trader_pending_orders", [{"id": "o2", "status": "pending"}])
    first = _transition(sb, "o2", "rejected")
    second = _transition(sb, "o2", "rejected")
    assert first and not second  # second reject is a clean no-op (409 in the route)
