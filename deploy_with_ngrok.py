# deploy_with_pyngrok.py
import subprocess
import time
from pyngrok import ngrok, conf

# Setting auth token 
NGROK_AUTHTOKEN = "33FGjOHflpwRo63KpGmjPfP0QxY_27PmHgXQ6kMAa8MFuts5r"
ngrok.set_auth_token(NGROK_AUTHTOKEN)

# Start streamlit
proc = subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.address", "127.0.0.1"
])

# Wait a bit for Streamlit to start 
time.sleep(3)

# Create HTTP tunnel to port 8501
tunnel = ngrok.connect(addr="8501", proto="http")
print("Public URL:", tunnel.public_url)
print("Press Ctrl+C here or close this window to stop both tunnel and Streamlit.")

try:
    proc.wait()
except KeyboardInterrupt:
    pass
finally:
    # cleanup
    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
    proc.terminate()
