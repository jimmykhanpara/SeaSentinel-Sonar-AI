"""
Launch Script for the AEGIS-SSS Marine Debris Sonar Dashboard
============================================================
Starts the FastAPI server with Uvicorn and launches the web interface.
"""

import uvicorn
import webbrowser
import threading
import time

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"

    print("\n" + "=" * 60)
    print("  AEGIS-SSS: Marine Debris & Ghost Net Detection System")
    print(f"  Starting local dashboard server at: http://{host}:{port}")
    print("=" * 60 + "\n")

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("sonar_debris.server.app:app", host=host, port=port, log_level="info")
