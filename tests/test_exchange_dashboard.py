import unittest

import trading_bot_main as tb


class ExchangeDashboardTests(unittest.TestCase):
    def setUp(self):
        tb.CONFIG.setdefault("live_accounts", {})["CRYPTO"] = {
            "exchange": "bybit",
            "api_key": "",
            "api_secret": "",
            "passphrase": "",
            "connected": False,
            "balance": 0.0,
            "last_sync": None,
            "last_error": "",
        }
        tb.CONFIG["live_trading_enabled"] = False

    def test_dashboard_payload_reports_disconnected_state(self):
        payload = tb.build_exchange_dashboard_payload("CRYPTO")
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["status"], "disconnected")
        self.assertIn("exchange", payload)
        self.assertIn("summary", payload)

    def test_live_trading_settings_include_dashboard_snapshot(self):
        client = tb.app.test_client()
        with client.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "admin"

        response = client.get("/api/settings/live-trading?market=CRYPTO")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("exchange_dashboard", payload)
        self.assertIn("summary", payload["exchange_dashboard"])


if __name__ == "__main__":
    unittest.main()
