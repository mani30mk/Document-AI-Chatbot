import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
from main import app, save_session
from fastapi.testclient import TestClient

client = TestClient(app)

class TestManualDeleteSession(unittest.TestCase):
    def test_delete_session_dev_fallback(self):
        # 1. Create a session in-memory
        sid = "test-manual-del-123"
        dev_id = "test-device-del"
        save_session(sid, files=["test.pdf"], history=[{"role": "user", "content": "hello"}], device_id=dev_id)
        
        # Verify it exists in memory
        self.assertIn(sid, main.session_files)
        self.assertIn(sid, main.session_histories)
        self.assertIn(sid, main.session_device_ids)

        # 2. Call DELETE /session/{sid} without supabase (mock None)
        with patch.object(main, "supabase_client", None):
            res = client.delete(f"/session/{sid}")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["deleted"], sid)

        # 3. Verify evicted from memory
        self.assertNotIn(sid, main.session_files)
        self.assertNotIn(sid, main.session_histories)
        self.assertNotIn(sid, main.session_device_ids)
        self.assertNotIn(sid, main.session_updated_at)
        self.assertNotIn(sid, main.session_access_order)

    def test_delete_session_with_supabase(self):
        mock_supabase = MagicMock()
        sid = "test-supabase-del-456"
        dev_id = "test-device-supabase"

        # Mock table calls
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_del = MagicMock()
        mock_table.delete.return_value = mock_del
        mock_eq = MagicMock()
        mock_del.eq.return_value = mock_eq
        mock_eq.execute.return_value = MagicMock(data=[{"session_id": sid}])

        # Mock storage listing & removal
        mock_bucket = MagicMock()
        mock_supabase.storage.from_.return_value = mock_bucket
        mock_bucket.list.return_value = [{"name": "fileA.pdf"}, {"name": "fileB.docx"}]

        # Populate memory
        save_session(sid, files=["fileA.pdf", "fileB.docx"], history=[], device_id=dev_id)

        with patch.object(main, "supabase_client", mock_supabase):
            res = client.delete(f"/session/{sid}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), {"deleted": sid})

        # Verify documents table deletion
        mock_table.delete().eq.assert_any_call("metadata->>session_id", sid)

        # Verify storage removal
        mock_bucket.remove.assert_called_with([f"{sid}/fileA.pdf", f"{sid}/fileB.docx"])

        # Verify chat_sessions table deletion
        mock_table.delete().eq.assert_any_call("session_id", sid)

        # Verify in-memory eviction
        self.assertNotIn(sid, main.session_files)
        self.assertNotIn(sid, main.session_device_ids)

if __name__ == "__main__":
    unittest.main()
