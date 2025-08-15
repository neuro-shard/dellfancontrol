# Dell R730xd PID Fan Controller (Enhanced)

This is a modified version of [Dan Smith's dellfancontrol](https://github.com/kk7ds/dellfancontrol) script, with extra features for **Proxmox/Debian** users and better support for **CPU package temperature control**.

## ✨ Features
- **Host CPU temperature sensor** (reads `Package id 0` from `lm-sensors`) for faster fan response to CPU load.
- **Weighted sensors** — bias certain sensors (like CPU) to have more influence than others when determining fan speed.
- **Periodic re-assert of manual fan mode** — prevents iDRAC from silently reverting to failsafe/auto mode.
- **Safe SMART temp handling** — ignores drives with no temperature reading instead of crashing.
- **Hot-reload YAML config** — edits are applied automatically without restarting the service.
- Works on **Proxmox** and other Debian-based systems.

---

## 📦 Requirements

### System packages
```bash
apt update
apt install -y ipmitool lm-sensors smartmontools python3-venv git
```

### Python packages (inside venv)
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install pyyaml pySMART simple_pid
```

---

## ⚙️ Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/neuro-shard/dellfancontrol.git
   cd dellfancontrol
   ```

2. Create and activate virtualenv:
   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install pyyaml pySMART simple_pid
   ```

3. Install `lm-sensors` and detect hardware:
   ```bash
   apt install lm-sensors
   sensors-detect --auto
   ```

4. Create your `fancontrol.yaml` (example below).

---

## 📝 Example `fancontrol.yaml`

```yaml
min_fan: 8
poll_time: 10

sensors:
  - {type: ipmi, name: "Inlet Temp",   target: 30, panic: 45, weight: 1.0}
  - {type: ipmi, name: "Exhaust Temp", target: 42, panic: 65, weight: 1.0}
  - {type: host, name: "Package id 0", target: 65, panic: 85, weight: 2.0}  # CPU bias

pid:
  kp: 2.2
  ki: 0.04
  kd: 0.10
  sample_time: 10
  pom: true
  dom: true
  fan_scale: 2
```

### Sensor types:
- **`ipmi`** — Reads from iDRAC/IPMI sensors (`ipmitool sensor`).
- **`smart`** — Reads from drive SMART temps (requires `smartmontools`).
- **`host`** — Reads from `lm-sensors` output on the host OS (e.g., CPU temps).

### Weights:
- Higher `weight` means that sensor’s temperature delta is more important in the final fan speed calculation.

---

## 🚀 Running

### Test mode:
```bash
. .venv/bin/activate
python3 fancontrol.py --config fancontrol.yaml --debug
```

### Install as a service:
Create `/etc/systemd/system/dellfancontrol.service`:
```ini
[Unit]
Description=Dell R730xd Fan Controller (PID)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/dellfancontrol
ExecStart=/root/dellfancontrol/.venv/bin/python /root/dellfancontrol/fancontrol.py --config /root/dellfancontrol/fancontrol.yaml
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable + start:
```bash
systemctl daemon-reload
systemctl enable --now dellfancontrol
journalctl -u dellfancontrol -f
```

---

## 🔍 Monitoring

Live watch with IPMI fans + temps + CPU package temp:
```bash
watch -n2 '
echo "== IPMI ==";
ipmitool sensor | egrep "Inlet Temp|Exhaust Temp|Fan[0-9] RPM";
echo;
echo "== CPU (lm-sensors) ==";
sensors | egrep "Package id [0-9]"
'
```

---

## ⚠️ Safety Notes
- Keep `min_fan >= 8` on R730xd to avoid fan faults and ensure VRM/HBA cooling.
- Panic thresholds instantly revert fans to iDRAC auto mode for safety.
- If you add HBAs or GPUs, consider raising `min_fan` to 10–12 and/or adding their temps.

---

## 📄 License
GPLv3 — see original [upstream project](https://github.com/kk7ds/dellfancontrol).
