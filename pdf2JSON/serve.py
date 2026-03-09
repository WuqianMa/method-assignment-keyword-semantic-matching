"""Minimal server for viewer.html – auto-serves output/ and middle/raw/ JSON files."""
import http.server, json, os, webbrowser

PORT = 8787
BASE = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/files":
            self._json_response(self._list_files())
        elif self.path.startswith("/api/file/clean/"):
            self._serve_json("output", self.path[len("/api/file/clean/"):])
        elif self.path.startswith("/api/file/raw/"):
            self._serve_json(os.path.join("middle", "raw"), self.path[len("/api/file/raw/"):])
        else:
            super().do_GET()

    def _list_files(self):
        clean, raw = [], []
        out_dir = os.path.join(BASE, "output")
        raw_dir = os.path.join(BASE, "middle", "raw")
        if os.path.isdir(out_dir):
            clean = sorted(f for f in os.listdir(out_dir) if f.endswith(".json"))
        if os.path.isdir(raw_dir):
            raw = sorted(f for f in os.listdir(raw_dir) if f.endswith(".json"))
        return {"clean": clean, "raw": raw}

    def _serve_json(self, folder, filename):
        # Sanitize filename
        filename = os.path.basename(filename)
        path = os.path.join(BASE, folder, filename)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet

os.chdir(BASE)
print(f"Viewer running at  http://localhost:{PORT}/viewer.html")
webbrowser.open(f"http://localhost:{PORT}/viewer.html")
http.server.HTTPServer(("", PORT), Handler).serve_forever()
