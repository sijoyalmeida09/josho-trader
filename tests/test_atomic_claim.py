"""Guards: two concurrent claim_command() on one row -> exactly one wins and the
other gets nothing; a consumed row is never re-emitted on a later drain; and
finish_command is one-way (a failed row cannot be re-claimed)."""

import threading

from fakes import FakeSupabase


def _seed_pending(sb, cid="cmd1"):
    sb.seed("trader_commands", [{"id": cid, "command": "pause", "status": "pending", "payload": {}}])


def test_concurrent_claim_exactly_one_winner(db_factory):
    sb = FakeSupabase()
    db = db_factory(sb)
    _seed_pending(sb)
    results = []

    def worker():
        results.append(db.claim_command("cmd1"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r is not None) == 1  # exactly one win
    assert sum(1 for r in results if r is None) == 1  # the other got nothing


def test_consumed_row_not_reemitted(db_factory):
    sb = FakeSupabase()
    db = db_factory(sb)
    _seed_pending(sb, "c")
    assert db.claim_command("c") is not None
    assert db.fetch_pending_commands() == []  # second drain sees nothing
    assert db.claim_command("c") is None  # cannot re-claim a consumed row


def test_finish_command_is_one_way(db_factory):
    sb = FakeSupabase()
    db = db_factory(sb)
    _seed_pending(sb, "c")
    db.claim_command("c")
    db.finish_command("c", {"ok": False}, ok=False)  # mark failed
    assert sb.tables["trader_commands"][0]["status"] == "failed"
    assert db.claim_command("c") is None  # failed never re-opens to pending
