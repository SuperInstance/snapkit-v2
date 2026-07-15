# ESP32 MPU6050 IMU Firmware

Reference firmware for the physical sensor layer of the Harmony Governor.

## What This Does

Reads roll, pitch, and yaw rate from an MPU6050 IMU on an ESP32 and outputs JSON over serial for the snapkit `MIDIBridge` to ingest.

The ESP32 is the **nervous system** — it translates physical motion into data the Harmony Governor can measure.

## Hardware

| Component | Connection |
|-----------|------------|
| ESP32 (any variant) | USB serial to computer |
| MPU6050 (GY-521) | I2C: SDA=GPIO21, SCL=GPIO22 |
| MPU6050 VCC | 3.3V |
| MPU6050 GND | GND |

## Output Format

JSON lines over serial at 115200 baud, ~10 Hz:

```json
{"roll": 5.23, "pitch": -1.2, "yaw_rate": 0.3, "roll_rate": 0.5, "temp": 24.5, "t": 12345}
```

Startup sequence:
```json
{"status":"boot"}
{"status":"mpu6050_ok"}
{"status":"calibrating","cycles":200}
{"status":"calibrated","ax_off":0.1,...}
{"status":"running"}
```

Heartbeat every 60 seconds:
```json
{"heartbeat":true,"uptime_s":3600,"samples":36000}
```

## Python Integration

```python
import json
import serial
from snapkit.midi_io import MIDIBridge
from snapkit.governor import HarmonyGovernor

gov = HarmonyGovernor()
gov.register_channel("helm", channel=0)
bridge = MIDIBridge(governor=gov)

# Connect to ESP32
ser = serial.Serial('/dev/ttyUSB0', 115200)

while True:
    line = ser.readline()
    data = json.loads(line)
    
    if 'roll' in data:
        # Feed IMU roll to derive hull tempo
        bpm = bridge.feed_roll(data['roll'])
        
        # Feed other sensors
        if 'heading' in data:
            bridge.feed_sensor('heading', data['heading'])
```

## Calibration

On boot, the firmware runs 200 calibration cycles (~20 seconds). Keep the boat still during calibration. The calibration offsets are stored in RAM (not persisted — recalibrate each power cycle).

## Multiple ESP32s

For a full sensor suite, deploy multiple ESP32s:

| ESP32 | Sensors | MIDI Channels |
|-------|---------|---------------|
| Helm station | IMU, compass, rudder angle | 0-2 |
| Engine room | RPM, temp, oil pressure | 3-5 |
| Deck | Bilge, bait counter, gear tension | 6-8 |
| Navigation | GPS, depth, wind | 9-11 |

Each ESP32 publishes to its own serial port. The Python bridge reads from all of them and routes to the appropriate MIDI channels.

## License

MIT
