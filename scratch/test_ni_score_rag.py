import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
import main
from main import app, save_session, session_docs, session_vectorstores, session_histories

client = TestClient(app)

class TestNIScoreRAG(unittest.TestCase):
    def test_ni_score_retrieval_hybrid(self):
        sid = "test-ni-score-session"
        sample_doc = (
            "Introduction to medical metrics.\n\n"
            "Chapter 1: Standard scoring systems and baseline evaluations.\n\n"
            "Chapter 2: Diagnostic criteria and methodology.\n\n"
            "Chapter 3: Laboratory tests and observation logs.\n\n"
            "Chapter 4: Special Indices.\n"
            "Section 4.2: The NI Score (Normalized Impact Score) is a quantitative clinical index "
            "ranging from 0 to 100 that measures patient neural integration and recovery velocity.\n\n"
            "Chapter 5: Discussion and future work."
        )
        
        # Save session in memory with document text
        save_session(sid, files=["neuro_metrics.pdf"], docs={"neuro_metrics.pdf": sample_doc}, history=[])
        session_docs[sid] = {"neuro_metrics.pdf": sample_doc}
        
        # Mock embeddings
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.05] * 384
        
        # Mock Gemini generate_chat to inspect the system_prompt context it receives
        captured_contexts = []
        def mock_generate_chat(model, key, system_prompt, history, question, timeout_s=25.0):
            captured_contexts.append(system_prompt)
            return (
                "The NI Score (Normalized Impact Score) is a quantitative clinical index ranging from 0 to 100 "
                "that measures patient neural integration and recovery velocity.\n"
                "<!-- yt_search: NI Score medical index -->"
            )
            
        with patch("routers.chat.get_embeddings", return_value=mock_emb), \
             patch("routers.chat.get_available_models", return_value=["gemini-2.5-flash"]), \
             patch("routers.chat.get_gemini_api_keys", return_value=["fake-key-123"]), \
             patch("routers.chat.generate_chat", side_effect=mock_generate_chat):
            
            res = client.post("/ask", json={"session_id": sid, "question": "what is NI Score?"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("Normalized Impact Score", data["answer"])
            
            # Verify the prompt context actually received the NI Score passage!
            self.assertTrue(len(captured_contexts) > 0)
            self.assertIn("NI Score", captured_contexts[0])
            self.assertIn("neural integration", captured_contexts[0])
            print("Successfully verified: NI Score was captured in prompt context and answered!")

if __name__ == "__main__":
    unittest.main()
