import os
import unittest

from dspy_agent.config import get_settings, Settings


class TestConfig(unittest.TestCase):
    def test_default_settings_types(self):
        s = get_settings()
        self.assertIsInstance(s, Settings)
        self.assertIsInstance(s.model_name, str)
        self.assertIsInstance(s.local_mode, bool)
        self.assertIsInstance(s.max_log_bytes, int)
        self.assertIn(s.tool_approval_mode, {"auto", "manual", "true", "false", os.getenv("TOOL_APPROVAL", "auto").lower()})

    def test_env_overrides(self):
        # Use os.environ directly for compatibility with unittest
        old = dict(os.environ)
        try:
            os.environ["MODEL_NAME"] = "my-model"
            os.environ["LOCAL_MODE"] = "TrUe"
            os.environ["MAX_LOG_BYTES"] = "12345"
            os.environ["TOOL_APPROVAL"] = "manual"
            os.environ["OPENAI_API_KEY"] = "sk-test"
            os.environ["OPENAI_BASE_URL"] = "http://localhost:1234"
            s = get_settings()
            self.assertEqual(s.model_name, "my-model")
            self.assertTrue(s.local_mode)
            self.assertEqual(s.max_log_bytes, 12345)
            self.assertEqual(s.tool_approval_mode, "manual")
            self.assertEqual(s.openai_api_key, "sk-test")
            self.assertEqual(s.openai_base_url, "http://localhost:1234")
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
