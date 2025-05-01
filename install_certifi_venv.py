# install_certifi_venv.py (revised)
import os
import subprocess
import sys
import certifi  # Added import

def main():
    python_exe = sys.executable

    # Step 1: Install/upgrade certifi
    print("\n[1] Installing certifi...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "certifi"])

    # Step 2: Use certifi's CA bundle path
    certifi_path = certifi.where()
    print(f"\n[2] Certifi CA bundle path: {certifi_path}")

    # Step 3: Set environment variable
    print("\n[3] Setting SSL_CERT_FILE permanently")
    with open(os.path.join(os.getenv("VIRTUAL_ENV"), ".env"), "a") as f:
        f.write(f"\nSSL_CERT_FILE={certifi_path}")

if __name__ == '__main__':
    main()