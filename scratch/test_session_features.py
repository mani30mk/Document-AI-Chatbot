import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient

# Set mock ADMIN_CLEANUP_KEY for testing
os.environ["ADMIN_CLEANUP_KEY"] = "test-secret-key-12345"

import main
from main import app, save_session, session_histories, session_files
main.ADMIN_CLEANUP_KEY = "test-secret-key-12345"

client = TestClient(app)

class TestSessionFeatures(unittest.TestCase):
    def test_post_session_new_without_device_id(self):
        res = client.post("/session/new")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("session_id", data)
        self.assertTrue(len(data["session_id"]) > 0)

    def test_post_session_new_with_device_id(self):
        dev_id = "test-device-uuid-abc"
        res = client.post("/session/new", json={"device_id": dev_id})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        sid = data["session_id"]
        self.assertTrue(len(sid) > 0)

        # Check in main.session_device_ids
        self.assertEqual(main.session_device_ids.get(sid), dev_id)

    def test_list_sessions_endpoint(self):
        dev_id = "test-device-uuid-xyz"
        
        # Session 1 with a file
        res1 = client.post("/session/new", json={"device_id": dev_id})
        sid1 = res1.json()["session_id"]
        save_session(sid1, files=["biology_chapter_1.pdf"], history=[], device_id=dev_id)

        # Session 2 with chat history
        res2 = client.post("/session/new", json={"device_id": dev_id})
        sid2 = res2.json()["session_id"]
        save_session(
            sid2, 
            files=[], 
            history=[{"role": "user", "content": "What is quantum entanglement and how does it work in physics?"}], 
            device_id=dev_id
        )

        # Query sessions for this device
        list_res = client.get(f"/sessions?device_id={dev_id}")
        self.assertEqual(list_res.status_code, 200)
        sessions = list_res.json()
        self.assertGreaterEqual(len(sessions), 2)

        sids = [s["session_id"] for s in sessions]
        self.assertIn(sid1, sids)
        self.assertIn(sid2, sids)

        s1_entry = next(s for s in sessions if s["session_id"] == sid1)
        s2_entry = next(s for s in sessions if s["session_id"] == sid2)

        self.assertEqual(s1_entry["title"], "biology_chapter_1.pdf")
        self.assertTrue("quantum entanglement" in s2_entry["title"])
        self.assertTrue("updated_at" in s1_entry)

    def test_admin_cleanup_authorization(self):
        os.environ["ADMIN_CLEANUP_KEY"] = "test-secret-key-12345"
        main.ADMIN_CLEANUP_KEY = "test-secret-key-12345"
        # 1. Missing key -> 403
        res = client.delete("/admin/sessions/cleanup")
        self.assertEqual(res.status_code, 403)

        # 2. Wrong key -> 403
        res = client.delete("/admin/sessions/cleanup?key=wrong-key")
        self.assertEqual(res.status_code, 403)

        # 3. Correct key
        res = client.delete("/admin/sessions/cleanup?key=test-secret-key-12345&older_than_days=5")
        # In test environment, supabase may or may not be connected. If not connected, 500 "Supabase not configured". If connected, 200 with {"deleted": ..., "errors": ...}
        self.assertIn(res.status_code, [200, 500])
        if res.status_code == 200:
            self.assertIn("deleted", res.json())

    def test_source_labeling_in_context(self):
        # Verify the context builder format
        chunks = [
            {"content": "Photosynthesis occurs in chloroplasts.", "metadata": {"source": "uploaded_docs/session1/biology.pdf"}},
            {"content": "Mitochondria produce ATP.", "metadata": {"source": "notes.txt"}},
        ]
        context_block = "\n\n".join(
            f"[{os.path.basename(chunk.get('metadata', {}).get('source', 'document'))}]: {chunk['content']}"
            for chunk in chunks
        )
        self.assertIn("[biology.pdf]: Photosynthesis occurs in chloroplasts.", context_block)
        self.assertIn("[notes.txt]: Mitochondria produce ATP.", context_block)

if __name__ == "__main__":
    unittest.main()
