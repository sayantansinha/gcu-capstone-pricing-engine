#!/bin/bash
set -euo pipefail

echo "[cleanup] Removing old app files from /opt/app"

# Clear everything under /opt/ppe-app
sudo rm -rf /opt/app/* /opt/app/.[!.]* /opt/app/..?*
