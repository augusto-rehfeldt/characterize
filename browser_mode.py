"""Browser mode: serve the fully client-side app in docs/ and open it.

Everything (character ranking, image mapping, rendering) runs in the
browser; this only hands out the three static files once. Equivalent to
`python -m http.server -d docs` plus opening the page. The same folder is
published via GitHub Pages.
"""

import functools
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "docs"


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(port=8765):
    handler = functools.partial(Quiet, directory=str(WEB_DIR))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:  # default port busy: let the OS pick one
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://localhost:{server.server_address[1]}"
    print(f"Browser mode running at {url} (Ctrl+C to stop)", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    serve()
