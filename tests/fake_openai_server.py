"""Minimal OpenAI-compatible server for verifying evalkit's custom-endpoint
support against real HTTP, with no vLLM install and no GPU.

    python3 tests/fake_openai_server.py 8731
    evalkit run examples/local_model.yaml --base-url http://127.0.0.1:8731/v1

Used by CI. Replies are canned but the transport, auth, payload shape and token
accounting are the real code paths.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        msgs = body.get("messages", [])
        system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        last = msgs[-1]["content"] if msgs else ""
        low = last.lower()

        if "json" in system.lower():
            txt = '{"category": "auth", "severity": "high"}'
        elif "yes or no" in low:
            txt = "yes"
        else:
            txt = "HTTP 429 means the client sent too many requests and must back off."

        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": body.get("model", "fake"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": txt},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(last) // 4 + 8,
                      "completion_tokens": max(1, len(txt) // 4),
                      "total_tokens": len(last) // 4 + 8 + max(1, len(txt) // 4)},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
    print(f"listening on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
