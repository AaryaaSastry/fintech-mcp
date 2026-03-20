import subprocess
import threading
import time
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Configurations
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
BACKEND_DIR = os.path.join(os.getcwd(), "backend")
FRONTEND_DIR = os.path.join(os.getcwd(), "frontend")

def run_backend():
    print(f"🚀 Starting Backend on http://localhost:{BACKEND_PORT}...")
    try:
        # We run the command as a module to handle imports correctly
        # Assuming the root is in PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run([sys.executable, "backend/main.py"], env=env)
    except Exception as e:
        print(f"❌ Backend failed to start: {e}")

class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

def run_frontend():
    print(f"🌐 Starting Frontend on http://localhost:{FRONTEND_PORT}...")
    server_address = ('', FRONTEND_PORT)
    httpd = HTTPServer(server_address, FrontendHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("      FINTECH AI ANALYTICS - LAUNCHER")
    print("="*50 + "\n")

    # 1. Start Backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Give backend a moment to initialize
    time.sleep(2)

    # 2. Start Frontend in a separate thread
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()

    # 3. Open the browser
    print(f"\n✨ System check complete!")
    print(f"📊 Dashboard available at: http://localhost:{FRONTEND_PORT}")
    print(f"⚙️  API Gateway available at: http://localhost:{BACKEND_PORT}")
    print("\nPress Ctrl+C to stop the servers.")
    
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down servers. See you next time!")
        sys.exit(0)
