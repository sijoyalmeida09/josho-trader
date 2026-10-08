# Test harness (STEP 0) — shared signatures. No test defines its own mock.

- `FakeSupabase(fail_reads=False)` — `.table(name)`, `.seed(table, rows)`; chained `select/insert/update/upsert(on_conflict)/delete/eq/gt/contains/order(desc)/limit/execute()`; `execute()` is lock-atomic so `update().eq("status","pending")` wins for exactly one caller; `fail_reads=True` makes SELECT raise.
- `make_trader_db(fake_sb, settings=None) -> TraderDB` — REAL TraderDB on the fake, seeds `trader_settings` singleton.
- `make_control_plane(db, kill_path, wallet=10000.0) -> ControlPlane` — REAL ControlPlane, no-op notifier.
- `FakeGrowwConn()` — `.buy()/.sell()` record to `.buys/.sells`; never networks; no kill logic (real SmartConnection owns the gate).
- `FakeClock(start=None)` — `.now(tz=None)`, `.advance(seconds)`.
- `KillFile(path)` — `.arm()`, `.disarm()`, `.exists()`, os.PathLike.
- Fixtures (conftest): `fake_sb`, `fake_groww`, `fake_clock`, `kill_file`, `db_factory`(=make_trader_db), `cp_factory`(=make_control_plane).
