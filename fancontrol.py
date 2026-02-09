#!/usr/bin/env python3
"""
Dell PID Fan Controller (hardened)

Fixes:
- Fans ramp up when hot AND reliably ramp down when cool again.
- Prevents PID integral "ratchet" from pinning fans high.
- Prevents a missing/failed sensor reading from holding a hot value forever.
- Periodically re-asserts manual mode + target to avoid iDRAC silent mode flips.

Config notes (fancontrol.yaml):
- min_fan: minimum % (default 10)
- poll_time: seconds between loops (default 10)
- units: c|f (default c)
- sensors: list of sensors with type/ipmi|smart|host, name/target/panic/weight

pid:
  kp: 2
  ki: 0.02
  kd: 0.0
  sample_time: 30
  fan_scale: 1
  deadband: 1.0            # °C above target where we aim for min_fan
  cooldown_reset_s: 60      # seconds delta<=deadband before PID reset
  cooldown_step: 2          # % step-down each loop when cooling (optional)
  max_output: 100           # optional cap of PID contribution (default 100)
"""

import argparse
import atexit
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import yaml
import pySMART
from simple_pid import PID

LOG = logging.getLogger("controller")
HOSTNAME = socket.gethostname()


def report_values(host, port, values):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((host, port))
        now = int(time.time())
        for key, value in values.items():
            key = str(key).replace(" ", "_").replace("/", "_")
            s.send(f"{key} {value} {now}\n".encode())
        s.close()
    except Exception:
        # Graphite should never break control
        pass


def run_ipmitool(*args) -> Optional[bytes]:
    for _ in range(3):
        try:
            return subprocess.check_output(["/usr/bin/ipmitool"] + list(args))
        except Exception as e:
            LOG.warning("Failed ipmitool %s: %s; retrying", " ".join(map(str, args)), e)
            time.sleep(2)
    LOG.error("PANIC: Unable to run ipmitool %s", " ".join(map(str, args)))
    return None


class Panic(Exception):
    pass


class IgnoreSensor(Exception):
    pass


class Sensor:
    """
    Base class for sensors.

    Key reliability behavior:
    - If we miss readings repeatedly, we stop "holding" the old value forever.
      After max_misses, current becomes None (ignored by composite).
    """

    def __init__(
        self,
        name: str,
        target: Optional[float],
        panic: Optional[float],
        sample_time: int = 0,
        max_misses: int = 3,
    ):
        self.name = name
        self.target = target
        self.panic = panic
        self.sample_time = sample_time
        self.max_misses = max_misses

        self._current: Optional[float] = target
        self.last_sample = 0.0
        self.misses = 0

    def get_sample(self) -> Optional[float]:
        raise NotImplementedError

    def sample(self):
        if time.monotonic() - self.last_sample < self.sample_time:
            return

        val = self.get_sample()
        self.last_sample = time.monotonic()

        if isinstance(val, (int, float)):
            self._current = float(val)
            self.misses = 0
            if self.panic is not None and self._current >= self.panic:
                raise Panic(f"Sensor panic: {self.name} is {self._current} > {self.panic}")
        else:
            # Missed/invalid sample
            self.misses += 1
            if self.misses >= self.max_misses:
                self._current = None  # stop pinning the controller on stale/hot values

    @property
    def current(self) -> Optional[float]:
        return self._current

    @property
    def delta(self) -> float:
        """
        delta = current - target
        Positive means hotter than target.
        """
        if self.target is None or not isinstance(self._current, (int, float)):
            return 0.0
        return float(self._current) - float(self.target)


class CompositeSensor:
    """
    Computes the maximum weighted delta across all valid sensors.
    Ignores sensors with current=None.
    """

    def __init__(self, sensors, weights=None):
        self.sensors = sensors
        self.weights = weights or {s: 1.0 for s in sensors}

    def sample(self):
        for s in self.sensors:
            s.sample()

    @property
    def delta(self) -> Optional[float]:
        valid = []
        for s in self.sensors:
            if s.current is None:
                continue
            w = float(self.weights.get(s, 1.0))
            valid.append(s.delta * w)

        if not valid:
            return None  # no valid readings -> fail safe
        return max(valid)


class IPMISensor(Sensor):
    sensors = {}
    _last_get = 0.0

    @classmethod
    def get_sensors(cls):
        if time.monotonic() - cls._last_get < 10:
            return

        out = run_ipmitool("sensor")
        if not out:
            return

        o = out.decode(errors="ignore")
        lines = [line.strip() for line in o.split("\n") if line.strip()]
        cls.sensors = {}

        for line in lines:
            fields = [x.strip() for x in line.split("|")]
            if len(fields) < 2:
                continue

            name = fields[0]
            raw = fields[1]

            val = None
            if raw.startswith("0x"):
                try:
                    val = int(raw, 16)
                except Exception:
                    val = None
            else:
                try:
                    val = float(raw)
                except Exception:
                    val = None

            if name in cls.sensors:
                try:
                    cls.sensors[name] = max(cls.sensors[name], val)
                except Exception:
                    pass
            else:
                cls.sensors[name] = val

        cls._last_get = time.monotonic()

    def get_sample(self) -> Optional[float]:
        self.get_sensors()
        val = self.sensors.get(self.name)
        if isinstance(val, (int, float)):
            return float(val)
        return None


class SMARTSensor(Sensor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        dev = pySMART.Device(self.name)
        if not getattr(dev, "interface", None):
            raise IgnoreSensor(f"SMART device {self.name} not accessible!")

    def get_sample(self) -> Optional[float]:
        dev = pySMART.Device(self.name)
        temp = None

        if getattr(dev, "if_attributes", None):
            temp = getattr(dev.if_attributes, "temperature", None)

        if temp is None and getattr(dev, "attributes", None):
            try:
                if 194 in dev.attributes:
                    temp = dev.attributes[194].raw
            except Exception:
                temp = None

        if isinstance(temp, (int, float)):
            return float(temp)
        return None


class HostCPUSensor(Sensor):
    """
    Reads CPU package temperature from lm-sensors output.
    "name" is a regex that matches a label like "Package id 0".
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        pattern = self.name or r"Package id 0"
        self._rx = re.compile(rf"^{pattern}:\s*\+?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    def get_sample(self) -> Optional[float]:
        try:
            out = subprocess.check_output(["sensors"], text=True, errors="ignore")
        except Exception as e:
            LOG.warning("Failed to read lm-sensors: %s", e)
            return None

        for line in out.splitlines():
            m = self._rx.search(line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None

        # Fallback
        m = re.search(r"CPU Temp:\s*\+?([0-9]+(?:\.[0-9]+)?)", out, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None

        return None


class Controller:
    def __init__(self, conf_file: str):
        self.conf_file = conf_file
        self.run = True

        self.current_target = None
        self._last_apply = 0.0

        self.errors = 0
        self.config_ts = 0.0

        self.load_config()

        atexit.register(self.fanmode_default)
        signal.signal(signal.SIGTERM, self.signal)
        signal.signal(signal.SIGINT, self.signal)

    def signal(self, signum, frame):
        LOG.warning("Signal received, stopping")
        self.run = False

    def temp(self, tempC: float) -> str:
        if self.config.get("units", "c").lower() == "f":
            return f"{(tempC * 9 / 5) + 32:.0f}F"
        return f"{tempC:.1f}C"

    def load_config(self):
        with open(self.conf_file, "r") as f:
            self.config = yaml.safe_load(f) or {}

        self.min_fan = int(self.config.get("min_fan", 10))
        self.poll_time = int(self.config.get("poll_time", 10))

        self.monitor_sensors = []
        self.sensor_weights = {}

        for data in self.config.get("sensors", []):
            try:
                stype = data["type"]
                name = data["name"]
                target = data.get("target")
                panic = data.get("panic")
                weight = float(data.get("weight", 1.0))

                if stype == "ipmi":
                    s = IPMISensor(name, target, panic, sample_time=0, max_misses=3)
                elif stype == "smart":
                    s = SMARTSensor(name, target, panic, sample_time=60, max_misses=3)
                elif stype == "host":
                    s = HostCPUSensor(name, target, panic, sample_time=5, max_misses=3)
                else:
                    raise IgnoreSensor(f"Unknown sensor type {stype!r}")

            except IgnoreSensor as e:
                LOG.warning("%s", e)
                continue
            except Exception as e:
                LOG.warning("Failed to init sensor %s: %s", data, e)
                continue

            self.monitor_sensors.append(s)
            self.sensor_weights[s] = weight

        self.monitor_temp = CompositeSensor(self.monitor_sensors, self.sensor_weights)
        self.config_ts = os.lstat(self.conf_file).st_mtime

    def check_config(self):
        try:
            current = os.lstat(self.conf_file).st_mtime
        except Exception:
            return
        if current > self.config_ts:
            LOG.info("Reloading changed config")
            self.load_config()

    def fanmode_default(self):
        # System-controlled
        if self.current_target is not None:
            LOG.info("Changing fan target to default (iDRAC auto)")
            run_ipmitool("raw", "0x30", "0x30", "0x01", "0x01")
            self.current_target = None

    def fanmode_set(self, pct: int):
        pct = int(pct)
        pct = max(self.min_fan, min(100, pct))

        # Re-assert manual mode periodically (in case iDRAC flips to auto)
        if (self.current_target != pct) or (time.monotonic() - self._last_apply > 30):
            LOG.info("Changing/refreshing fan target to %i%%", pct)
            run_ipmitool("raw", "0x30", "0x30", "0x01", "0x00")
            run_ipmitool("raw", "0x30", "0x30", "0x02", "0xff", f"0x{pct:02x}")
            self.current_target = pct
            self._last_apply = time.monotonic()

    def get_sensors(self):
        last = {s.name: s.current for s in self.monitor_temp.sensors}
        self.monitor_temp.sample()

        vals = {}
        for s in self.monitor_temp.sensors:
            if s.current is None:
                continue
            prev = last.get(s.name)
            if prev is None or s.current != prev:
                if s.target is not None and s.delta >= -2:
                    vals[s.name] = f"{int(s.current)}({s.delta:+.0f})"
                elif s.panic is not None and (s.current / s.panic) > 0.90:
                    vals[s.name] = f"{int(s.current)}(!{s.panic})"

        if vals:
            LOG.debug("Sensors: %s", ", ".join([f"{k}={v}" for k, v in vals.items()]))

    def target_logic(self) -> Optional[int]:
        raise NotImplementedError

    def calculate_target(self):
        target = self.target_logic()

        if target is None:
            self.fanmode_default()
        else:
            self.fanmode_set(target)

        # Optional graphite metrics
        if self.config.get("graphite"):
            prefix_mode = self.config["graphite"].get("prefix_hostname")
            prefix = HOSTNAME.split(".")[0] if prefix_mode == "short" else HOSTNAME

            delta = self.monitor_temp.delta
            vals = {"target": target if target is not None else -1, "delta": delta if delta is not None else "nan"}
            for s in self.monitor_temp.sensors:
                vals[f"sensor_{s.name}"] = s.current if s.current is not None else "nan"

            report_values(
                self.config["graphite"]["host"],
                self.config["graphite"].get("port", 2003),
                {f"{prefix}.fancontrol.{k}": v for k, v in vals.items()},
            )

        if self.errors:
            LOG.info("Recovered from %i sensor errors", self.errors)
        self.errors = 0

    def default_on_error(self):
        if self.errors >= 3:
            LOG.error("%i sensor errors - going to default", self.errors)
            self.fanmode_default()
        else:
            LOG.warning("%i sensor errors, retrying before default", self.errors + 1)
            self.errors += 1

    def monitor_loop(self):
        while self.run:
            self.check_config()
            try:
                self.get_sensors()
                self.calculate_target()
            except Panic as e:
                LOG.warning("%s", e)
                self.fanmode_default()
            except Exception as e:
                LOG.exception("Unknown failure: %s", e)
                self.default_on_error()

            time.sleep(self.poll_time)

        self.fanmode_default()


class PIDController(Controller):
    def __init__(self, conf_file: str):
        # Positive-gain PID. We feed (-delta) so error becomes +delta when hot.
        self.pid = PID(setpoint=0)
        self.pid_config = {}
        self._below_deadband_since = None
        super().__init__(conf_file)

    def load_config(self):
        super().load_config()
        self.pid_config = self.config.get("pid", {}) or {}

        kp = float(self.pid_config.get("kp", 2))
        ki = float(self.pid_config.get("ki", 0.02))
        kd = float(self.pid_config.get("kd", 0.0))

        self.pid.Kp = kp
        self.pid.Ki = ki
        self.pid.Kd = kd

        self.pid.sample_time = int(self.pid_config.get("sample_time", 30))
        self.pid.proportional_on_measurement = bool(self.pid_config.get("pom", False))
        self.pid.derivative_on_measurement = bool(self.pid_config.get("dom", False))

        # PID output is the *extra* percentage over min_fan
        max_output = float(self.pid_config.get("max_output", 100))
        self.pid.output_limits = (0, max_output)

        LOG.info(
            "PID params kp=%.3f ki=%.4f kd=%.3f sample=%is (output 0..%.1f)",
            self.pid.Kp, self.pid.Ki, self.pid.Kd, self.pid.sample_time, max_output
        )

    def target_logic(self) -> Optional[int]:
        delta = self.monitor_temp.delta

        # If no valid readings, fail safe to iDRAC control
        if delta is None:
            LOG.warning("No valid sensor readings - defaulting to iDRAC auto")
            return None

        deadband = float(self.pid_config.get("deadband", 1.0))
        cooldown_reset_s = float(self.pid_config.get("cooldown_reset_s", 60))
        cooldown_step = float(self.pid_config.get("cooldown_step", 0))  # % per loop
        fan_scale = float(self.pid_config.get("fan_scale", 1.0))

        # Deadband behavior: if we're not meaningfully above target, aim for min_fan
        if delta <= deadband:
            if self._below_deadband_since is None:
                self._below_deadband_since = time.monotonic()

            # After being cool/near-target for long enough, reset PID (clears windup)
            if time.monotonic() - self._below_deadband_since >= cooldown_reset_s:
                self.pid.reset()

            # Smoothly step down if requested and we're currently above min_fan
            if cooldown_step > 0 and isinstance(self.current_target, int) and self.current_target > self.min_fan:
                return int(max(self.min_fan, self.current_target - cooldown_step))

            return int(self.min_fan)

        # We're above deadband -> actively control
        self._below_deadband_since = None

        # Feed -delta so PID error becomes +delta when hot:
        # error = setpoint - input = 0 - (-delta) = +delta
        output = self.pid(-delta)

        target = self.min_fan + (output * fan_scale)
        target = max(self.min_fan, min(100, target))

        LOG.debug("Max weighted delta=%.2fC => pid_out=%.2f => target=%d%%", delta, output, int(round(target)))
        return int(round(target))


def main():
    conf = os.path.expanduser("~/.config/fancontrol.yaml")

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=conf, required=False)
    p.add_argument("--debug", default=False, action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    c = PIDController(args.config)
    c.monitor_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
