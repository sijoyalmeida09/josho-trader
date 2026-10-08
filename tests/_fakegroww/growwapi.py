"""Hermetic fake of the `growwapi` package, injected via PYTHONPATH for the
smoke subprocess ONLY (it shadows the real package). Never networks; returns
minimal, fast, deterministic data so `--scan-once` connects and exits 0 with no
broker and no secrets."""


class GrowwAPI:
    def __init__(self, token=None, *args, **kwargs):
        self.token = token

    @staticmethod
    def get_access_token(*args, **kwargs):
        return "fake-token"

    def get_available_margin_details(self, *args, **kwargs):
        return {"clear_cash": 10000.0}

    def get_expiries(self, *args, **kwargs):
        # Falsy -> the F&O scan loop skips immediately (fast, no option-chain calls).
        return {}

    def get_option_chain(self, *args, **kwargs):
        return {}

    def get_quote(self, *args, **kwargs):
        return {"last_price": 100.0, "ohlc": {"close": 100.0, "high": 101.0, "low": 99.0}}

    def get_positions_for_user(self, *args, **kwargs):
        return {}

    def place_order(self, *args, **kwargs):
        return {"groww_order_id": "FAKE", "order_status": "FAKE"}
