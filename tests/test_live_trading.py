import tempfile
import unittest
from pathlib import Path

import trading_bot_main as tb


class LiveTradingPayloadTests(unittest.TestCase):
    def test_build_live_trading_payload_reports_defaults(self):
        tb.CONFIG["live_trading_enabled"] = False
        tb.CONFIG["live_account_connected"] = False
        tb.CONFIG["live_balance"] = 0.0
        tb.CONFIG["live_exchange"] = "binance"
        tb.CONFIG.setdefault("live_accounts", {})["CRYPTO"] = {"exchange": "binance", "connected": False, "balance": 0.0}
        payload = tb.build_live_trading_payload()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["exchange"], "binance")
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["balance"], 0.0)

    def test_build_live_trading_payload_uses_current_state(self):
        tb.CONFIG["live_trading_enabled"] = True
        tb.CONFIG["live_exchange"] = "bybit"
        tb.CONFIG["live_account_connected"] = True
        tb.CONFIG["live_balance"] = 1234.5
        tb.CONFIG.setdefault("live_accounts", {})["CRYPTO"] = {"exchange": "bybit", "connected": True, "balance": 1234.5}
        payload = tb.build_live_trading_payload()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["exchange"], "bybit")
        self.assertTrue(payload["connected"])
        self.assertEqual(payload["balance"], 1234.5)

    def test_database_persists_admin_activity_and_exchange_account(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trade_bot.db"
            tb.init_database(str(db_path))
            tb.record_admin_activity("admin", "login", "Signed in", session_id="sess-1")
            tb.save_exchange_account("CRYPTO", "binance", "key", "secret", "", True, 1234.5, "ok")

            activities = tb.get_recent_admin_activity("admin", limit=5)
            self.assertTrue(any(item["activity_type"] == "login" for item in activities))

            account = tb.get_saved_exchange_account("CRYPTO")
            self.assertEqual(account["exchange"], "binance")
            self.assertEqual(account["balance"], 1234.5)
            self.assertTrue(account["connected"])

    def test_trade_manager_can_open_and_close_without_crashing(self):
        original_paper = tb.CONFIG["paper_trading"]
        tb.CONFIG["paper_trading"] = True
        try:
            tm = tb.TradeManager()
            trade = tm.open_trade(
                "BTC/USDT",
                {"price": 100.0, "sl": 95.0, "tp": 105.0, "confidence": 90.0, "side": "BUY", "reasons": []},
                {"action": "CONFIRM", "analysis": "ok", "risk_level": "LOW"},
            )
            self.assertIsNotNone(trade)
            self.assertEqual(trade["status"], "OPEN")
            closed = tm.force_close(trade["id"], {"BTC/USDT": 102.0})
            self.assertIsNotNone(closed)
            self.assertEqual(closed["status"], "MANUAL")
        finally:
            tb.CONFIG["paper_trading"] = original_paper


if __name__ == "__main__":
    unittest.main()
