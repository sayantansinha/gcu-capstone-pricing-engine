#!/bin/bash
set -euo pipefail

echo "[install_service] Installing/refreshing ppe.service"

# Copy the unit file from the current revision into systemd
cp /opt/app/deploy_scripts/ppe.service /etc/systemd/system/ppe.service
chmod 644 /etc/systemd/system/ppe.service

systemctl daemon-reload
systemctl enable ppe.service
