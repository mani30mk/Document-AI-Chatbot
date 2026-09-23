import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADMIN_CLEANUP_KEY"] = "super-secret-cleanup-key"

import main
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestCleanupMock(unittest.TestCase):
    def test_cascading_cleanup_execution(self):
        os.environ["ADMIN_CLEANUP_KEY"] = "super-secret-cleanup-key"
        main.ADMIN_CLEANUP_KEY = "super-secret-cleanup-key"
        mock_supabase = MagicMock()
        old_sid = "session-old-123"
        recent_sid = "session-recent-456"

        # Mock chat_sessions query response
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        
        # When querying chat_sessions for old sessions
        mock_select = MagicMock()
        mock_table.select.return_value = mock_select
        mock_lt = MagicMock()
        mock_select.lt.return_value = mock_lt
        mock_lt.execute.return_value = MagicMock(data=[{"session_id": old_sid}])

        # Mock storage listing
        mock_storage_bucket = MagicMock()
        mock_supabase.storage.from_.return_value = mock_storage_bucket
        mock_storage_bucket.list.return_value = [{"name": "doc1.pdf"}, {"name": "doc2.txt"}]

        # Set in-memory data
        main.session_histories[old_sid] = [{"role": "user", "content": "hi"}]
        main.session_files[old_sid] = ["doc1.pdf"]
        main.session_docs[old_sid] = {"doc1.pdf": "hello"}
        main.session_slides[old_sid] = {}
        main.session_vectorstores[old_sid] = {"chunks": []}
        main.session_device_ids[old_sid] = "dev-1"
        main.session_updated_at[old_sid] = "2020-01-01T00:00:00Z"
        main.session_access_order.append(old_sid)

        # Recent session in-memory
        main.session_histories[recent_sid] = [{"role": "user", "content": "hello"}]

        with patch.object(main, "supabase_client", mock_supabase):
            res = client.delete(
                f"/admin/sessions/cleanup?older_than_days=5&key=super-secret-cleanup-key"
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["deleted"], 1)
            self.assertEqual(len(data["errors"]), 0)

        # Verify storage remove was called with correct paths
        mock_storage_bucket.remove.assert_called_with([f"{old_sid}/doc1.pdf", f"{old_sid}/doc2.txt"])

        # Verify documents vector rows deleted
        mock_table.delete().eq.assert_any_call("metadata->>session_id", old_sid)

        # Verify chat_sessions row deleted
        mock_table.delete().eq.assert_any_call("session_id", old_sid)

        # Verify in-memory eviction
        self.assertNotIn(old_sid, main.session_histories)
        self.assertNotIn(old_sid, main.session_files)
        self.assertNotIn(old_sid, main.session_docs)
        self.assertNotIn(old_sid, main.session_vectorstores)
        self.assertNotIn(old_sid, main.session_device_ids)
        self.assertNotIn(old_sid, main.session_updated_at)
        self.assertNotIn(old_sid, main.session_access_order)

        # Recent session remains untouched
        self.assertIn(recent_sid, main.session_histories)

if __name__ == "__main__":
    unittest.main()
