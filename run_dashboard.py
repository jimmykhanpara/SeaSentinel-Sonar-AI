"""
Launch Script for the SeaSentinel Marine Debris Sonar Dashboard
==============================================================
Robust local launcher with auto-port detection, readiness polling,
and automatic browser opening.
"""

import os
import sys
import time
import socket
import threading
import webbrowser
import urllib.request

# Ensure workspace directory is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sonar_debris.server.app import app
import uvicorn


def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Finds an available TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start_port


def wait_and_open_browser(host: str, port: int, timeout: float = 12.0):
    """Waits until the FastAPI server responds with 200 OK before launching browser."""
    url = f"http://{host}:{port}"
    health_url = f"{url}/api/system-status"
    start_time = time.time()

    print(f"[*] Initializing server & waiting for readiness at: {url} ...")
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=0.8) as response:
                if response.status == 200:
                    print(f"\n[✓] Server is READY! Launching browser at: {url}\n")
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.3)

    # Fallback if timeout reached
    print(f"[!] Launching browser directly at: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = find_free_port(8000)

    print("\n" + "=" * 65)
    print("  🌊 SeaSentinel: AI Sonar Marine Debris & Ghost Net System")
    print(f"  Local Mission Dashboard URL: http://{host}:{port}")
    print("=" * 65 + "\n")

    # Start browser opener in background thread
    launcher_thread = threading.Thread(
        target=wait_and_open_browser,
        args=(host, port),
        daemon=True
    )
    launcher_thread.start()

    # Run Uvicorn server directly with app instance
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n[✓] SeaSentinel server stopped cleanly. Goodbye!")
