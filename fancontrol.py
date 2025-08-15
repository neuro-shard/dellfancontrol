#!/usr/bin/env python3

# Dell R730xd PID Fan Controller (enhanced)
# - Adds Host CPU package temperature sensor via lm-sensors
# - Periodically re-asserts manual fan mode so iDRAC can’t silently flip to auto
# - More robust SMART temperature handling
#
# Original Copyright 2023 Dan Smith <dellfancontrol@f.danplanet.com>
# Released under the GPLv3

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

try:
    from yaml import CLoader as YamlLoader
except Exception:  # pragma: no cover
    from yaml import SafeLoader as YamlLoader
import yaml

import pySMART
from simple_pid import PID

LOG = logging.getLogger('controller')
HOSTNAME = socket.gethostname()


def report_values(host, port, values):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        for key, value in values.items():
            key = key.replace(' ', '_').replace('/', '_')
            s.send(f"{key} {value} {int(time.time())}\n".encode())
        s.close()
    except Exception:
        ...


def run_ipmitool(*args):
    # LOG.debug('Running ipmitool %s', ' '.join(str(a) for a in args))
    for _ in range(3):
        try:
            return subprocess.check_output(['/usr/bin/ipmitool'] + list(args))
        except Exception as e:
            LOG.exception('Failed to run ipmitool %s: %s; retrying', str(args), e)
            time.sleep(5)
    LOG.error('PANIC: Unable to run ipmitool!')


class Panic(Exception):
    pass


class IgnoreSensor(Exception):
    pass


class Sensor:
    def __init__(self, name, target, panic, sample_time=0):
        self.name = name
        self.target = target
        self.panic = panic
        self._current = target
        self.sample_time = sample_time
        self.last_sample = 0

    def get_sample(self):
        raise NotImplementedError

    def sample(self):
        if time.monotonic() - self.last_sample >= self.sample_time:
            self._current = self.get_sample()
            self.last_sample = time.monotonic()
            # Only do panic check when we have a numeric reading
            if self.panic is not None and isinstance(self.current, (int, float)) and self.current >= self.panic:
                raise Panic(f'Sensor panic: {self.name} is {self.current} > {self.panic}')

    @property
    def current(self):
        return self._current

    @property
    def delta(self):
        if self.target is None or not isinstance(self._current, (int, float)):
            return 0
        return self._current - self.target


class CompositeSensor(Sensor):
    def __init__(self, sensors, weights=None):
        # sensors: list of Sensor objects
        # weights: dict {sensor_obj: float} to bias certain sensors
        self.sensors = sensors
        self.weights = weights or {s: 1.0 for s in sensors}

    def sample(self):
        for i in self.sensors:
            i.sample()

    @property
    def delta(self):
        if not self.sensors:
            return 0
        # Weight hotter sensors more heavily (e.g., CPU package)
        return max([(s.delta * self.weights.get(s, 1.0)) for s in self.sensors])


class IPMISensor(Sensor):
    # Cache of sensor values from the last refresh
    sensors = {}
    # Timestamp of last sensor refresh from IPMI
    _last_get = 0

    @classmethod
    def get_sensors(cls):
        if time.monotonic() - cls._last_get < 10:
            return

        o = run_ipmitool('sensor').decode(errors='ignore')
        lines = [line.strip() for line in o.split('\n') if line.strip()]
        cls.sensors = {}
        for line in lines:
            fields = [x.strip() for x in line.split('|')]
            if len(fields) < 2:
                continue
            val = fields[1]
            if val.startswith('0x'):
                try:
                    val = int(val, 16)
                except Exception:
                    val = None
            else:
                try:
                    val = float(fields[1])
                except ValueError:
                    val = None
            if fields[0] in cls.sensors:
                # Take the higher of duplicate keys when comparable
                try:
                    cls.sensors[fields[0]] = max(cls.sensors[fields[0]], val)
                except Exception:
                    pass
            else:
                cls.sensors[fields[0]] = val
        cls._last_get = time.monotonic()

    def get_sample(self):
        self.get_sensors()
        val = self.sensors.get(self.name)
        if isinstance(val, (int, float)):
            return int(val)
        # If an IPMI value is unavailable, keep the last reading
        return self._current


class SMARTSensor(Sensor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        dev = pySMART.Device(self.name)
        if not getattr(dev, 'interface', None):
            raise IgnoreSensor(f'SMART device {self.name} not accessible!')

    def get_sample(self):
        dev = pySMART.Device(self.name)
        temp = None
        # Prefer interface attributes (common)
        if getattr(dev, 'if_attributes', None):
            temp = getattr(dev.if_attributes, 'temperature', None)
        # Fallback to raw SMART attribute 194 if present
        if temp is None and getattr(dev, 'attributes', None):
            try:
                if 194 in dev.attributes:
                    temp = dev.attributes[194].raw
            except Exception:
                temp = None
        # If still no temp, keep previous reading to avoid None in comparisons
        return int(temp) if isinstance(temp, (int, float)) else self._current


class HostCPUSensor(Sensor):
    """
    Reads CPU package temperature from lm-sensors output.
    "name" is a regex that matches a label like "Package id 0".
    Example `sensors` line: "Package id 0:  +37.0°C  (high = +80.0°C, crit = +100.0°C)"
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        pattern = self.name or r"Package id 0"
        self._rx = re.compile(rf"^{pattern}:\s*\+?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

    def get_sample(self):
        try:
            out = subprocess.check_output(['sensors'], text=True, errors='ignore')
        except Exception as e:
            LOG.warning('Failed to read lm-sensors: %s', e)
            return self._current
        for line in out.splitlines():
            m = self._rx.search(line)
            if m:
                try:
                    return int(float(m.group(1)))
                except Exception:
                    break
        # Fallback: try a generic "CPU Temp" label some chipsets use
        m = re.search(r"CPU Temp:\s*\+?([0-9]+(?:\.[0-9]+)?)", out, re.IGNORECASE)
        if m:
            try:
                return int(float(m.group(1)))
            except Exception:
                ...
        return self._current


class Controller:
    def __init__(self, config):
        self.conf_file = config
        self.load_config()
        self.run = True
        self.current_target = object()
        self.sensors = {}
        self.errors = 0
        self._last_apply = 0  # re-assert manual mode periodically

        # Attempt to always revert to default on exit
        atexit.register(self.fanmode_default)
        signal.signal(signal.SIGTERM, self.signal)

    def temp(self, tempC):
        if self.config.get('units', 'c').lower() == 'f':
            return f"{(tempC * 9 / 5) + 32:.0f}F"
        else:
            return f"{tempC:.1f}C"

    def load_config(self):
        with open(self.conf_file, 'r') as conf_file:
            self.config = yaml.load(conf_file, Loader=YamlLoader)
        self.min_fan = self.config.get('min_fan', 10)

        self.monitor_sensors = []
        self.sensor_weights = {}
        for data in self.config['sensors']:
            try:
                if data['type'] == 'ipmi':
                    s = IPMISensor(data['name'], data.get('target'), data.get('panic'))
                elif data['type'] == 'smart':
                    s = SMARTSensor(data['name'], data.get('target'), data.get('panic'), sample_time=60)
                elif data['type'] == 'host':
                    s = HostCPUSensor(data.get('name', r'Package id 0'), data.get('target'), data.get('panic'), sample_time=5)
                else:
                    raise IgnoreSensor(f'Unknown sensor type {data["type"]!r}')
            except IgnoreSensor as e:
                LOG.warning(e)
                continue
            self.monitor_sensors.append(s)
            self.sensor_weights[s] = float(data.get('weight', 1.0))

        self.monitor_temp = CompositeSensor(self.monitor_sensors, self.sensor_weights)
        self.poll_time = self.config.get('poll_time', 10)
        self.config_ts = os.lstat(self.conf_file).st_mtime

    def check_config(self):
        current = os.lstat(self.conf_file).st_mtime
        if current > self.config_ts:
            LOG.info('Reloading changed config')
            self.load_config()

    def signal(self, signum, frame):
        LOG.warning('Signal received, stopping')
        self.run = False

    def fanmode_default(self):
        # System-controlled
        if self.current_target is not None:
            LOG.info('Changing fan target to default')
            run_ipmitool('raw', '0x30', '0x30', '0x01', '0x01')
            self.current_target = None

    def fanmode_set(self, pct):
        # Manual set to a speed; also re-assert regularly in case iDRAC flips to auto
        pct = min(100, int(pct))
        if (self.current_target != pct) or (time.monotonic() - self._last_apply > 30):
            LOG.info('Changing/refreshing fan target to %i%%', pct)
            run_ipmitool('raw', '0x30', '0x30', '0x01', '0x00')
            run_ipmitool('raw', '0x30', '0x30', '0x02', '0xff', f'0x{pct:02x}')
            self.current_target = pct
            self._last_apply = time.monotonic()

    def get_sensors(self):
        last = {s.name: s.current for s in self.monitor_temp.sensors}
        self.monitor_temp.sample()

        vals = {}
        for s in self.monitor_temp.sensors:
            if (s.current != last.get(s.name) and s.target is not None and s.delta >= -2):
                vals[s.name] = f"{s.current}(+{s.delta:+.0f})"
            elif (s.current != last.get(s.name) and s.panic is not None and isinstance(s.current, (int, float)) and s.current / s.panic > 0.90):
                vals[s.name] = f"{s.current}(!{s.panic})"
        if vals:
            LOG.debug('Sensors: %s', ','.join([f'{k}={v}' for k, v in vals.items()]))

    def calculate_target(self):
        target = self.target_logic()
        if target is None:
            self.fanmode_default()
        else:
            self.fanmode_set(target)

        if self.config.get('graphite'):
            if self.config['graphite'].get('prefix_hostname') == 'short':
                prefix = HOSTNAME.split('.')[0]
            else:
                prefix = HOSTNAME

            vals = {'target': target, 'delta': self.monitor_temp.delta}
            for s in self.monitor_temp.sensors:
                vals[f'sensor_{s.name}'] = s.current

            report_values(self.config['graphite']['host'], self.config['graphite'].get('port', 2003), {f'{prefix}.fancontrol.{k}': v for k, v in vals.items()})

        if self.errors:
            LOG.info('Recovered from %i sensor errors', self.errors)
        self.errors = 0

    def default_on_error(self):
        if self.errors >= 3:
            LOG.error('%i sensor errors - going to default', self.errors)
            self.fanmode_default()
        else:
            LOG.warning('%i sensor errors, retrying before default', self.errors)
            self.errors += 1

    def monitor_loop(self):
        while self.run:
            self.check_config()
            try:
                self.get_sensors()
            except Panic as e:
                LOG.warning(e)
                self.fanmode_default()
            except Exception as e:
                LOG.exception('Unknown failure: %s', e)
                self.default_on_error()
            else:
                self.calculate_target()
            time.sleep(self.poll_time)
        # If we exit the control loop for any reason, back to default.
        self.fanmode_default()


class SimpleController(Controller):
    def __init__(self, config):
        super().__init__(config)
        self.warn_temp = self.config['simple']['warn_temp']

    def target_logic(self, temp):
        LOG.debug('Temp %r is %s', self.monitor_temp, self.temp(temp))
        if temp < self.warn_temp:
            target = self.min_fan
        elif temp < self.panic_temp:
            target = 40
        else:
            target = None
        return target


class PIDController(Controller):
    def __init__(self, config):
        self.pid = PID(setpoint=0)
        super().__init__(config)
        self._last_delta = None
        self._last_target = None

    def load_config(self):
        super().load_config()
        self.pid_config = self.config.get('pid', {})

        self.pid.Kp = self.pid_config.get('kp', 2) * -1
        self.pid.Ki = self.pid_config.get('ki', 0.02) * -1
        self.pid.Kd = self.pid_config.get('kd', 0.0) * -1
        self.pid.sample_time = self.pid_config.get('sample_time', 30)
        self.pid.proportional_on_measurement = bool(self.pid_config.get('pom', False))
        self.pid.derivative_on_measurement = bool(self.pid_config.get('dom', False))
        self.pid.output_limits = (0, None)
        LOG.info('PID params %.2f,%.2f,%.2f (sample %i, pom %s, dom %s)', self.pid.Kp, self.pid.Ki, self.pid.Kd, self.pid.sample_time, self.pid.proportional_on_measurement, self.pid.derivative_on_measurement)

    def target_logic(self):
        delta = self.monitor_temp.delta
        output = self.pid(delta)

        if output < 0:
            target = self.min_fan
        else:
            target = self.min_fan + (output * self.pid_config.get('fan_scale', 1))

        if target != self._last_target or delta != self._last_delta:
            LOG.debug('Max delta is %s Target %i%% want=%.1f', delta, int(target), output)

        self._last_target = int(target)
        self._last_delta = delta
        return int(target)


def main():
    conf = os.path.expanduser('~/.config/fancontrol.yaml')
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=conf, required=False)
    p.add_argument('--debug', default=False, action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
    c = PIDController(args.config)
    c.monitor_loop()


if __name__ == '__main__':
    sys.exit(main())
