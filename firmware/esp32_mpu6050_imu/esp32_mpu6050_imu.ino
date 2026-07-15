/*
 * ESP32 MPU6050 IMU → snapkit-v2 Harmony Governor
 * 
 * Firmware for the physical sensor layer. Reads roll/pitch from an MPU6050
 * and outputs JSON over serial for the MIDIBridge to ingest.
 *
 * Also outputs raw MIDI CC messages over USB-MIDI (optional).
 *
 * Hardware:
 *   - ESP32 (any variant: DevKit, WROOM, S3)
 *   - MPU6050 (GY-521 breakout)
 *   - I2C: SDA=GPIO21, SCL=GPIO22 (default ESP32)
 *   - Optional: USB-MIDI on ESP32-S3
 *
 * Wiring:
 *   MPU6050 VCC → 3.3V
 *   MPU6050 GND → GND
 *   MPU6050 SDA → GPIO21
 *   MPU6050 SCL → GPIO22
 *
 * Output format (serial, 115200 baud, JSON lines):
 *   {"roll": 5.23, "pitch": -1.2, "yaw_rate": 0.3, "temp": 24.5, "t": 12345}
 *
 * License: MIT
 */

#include <Wire.h>
#include <ArduinoJson.h>

// ═══════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════

#define I2C_ADDR_MPU6050   0x68
#define SAMPLE_RATE_HZ     10    // 10 Hz = every 100ms
#define SERIAL_BAUD        115200
#define CALIBRATION_CYCLES 200   // 20 seconds at 10Hz for calibration
#define GRAVITY            16384.0  // MPU6050 1g in ±2g mode

// ═══════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════

struct Calibration {
  float accel_x_offset = 0;
  float accel_y_offset = 0;
  float accel_z_offset = 0;
  float gyro_x_offset = 0;
  float gyro_y_offset = 0;
  float gyro_z_offset = 0;
};

Calibration cal;
unsigned long last_sample = 0;
unsigned long start_time = 0;
unsigned long sample_count = 0;
bool calibrated = false;

// Complementary filter state
float roll_angle = 0.0;
float pitch_angle = 0.0;
float yaw_rate = 0.0;

// Previous values for derivative computation
float prev_roll = 0.0;
float prev_yaw_rate = 0.0;

// ═══════════════════════════════════════════════════════════════
//  MPU6050 Low-Level I/O
// ═══════════════════════════════════════════════════════════════

void mpu_write_byte(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(I2C_ADDR_MPU6050);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

void mpu_read_bytes(uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(I2C_ADDR_MPU6050);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(I2C_ADDR_MPU6050, len, true);
  for (uint8_t i = 0; i < len && Wire.available(); i++) {
    buf[i] = Wire.read();
  }
}

bool mpu_init() {
  // Wake up the MPU6050 (it starts in sleep mode)
  mpu_write_byte(0x6B, 0x00);
  delay(100);

  // Configure:
  // - Accel range: ±2g (register 0x1C = 0x00)
  // - Gyro range: ±250°/s (register 0x1B = 0x00)
  // - Low-pass filter: 44Hz (register 0x1A = 0x03)
  // - Sample rate divider: 9 → ~100Hz internal (register 0x19 = 0x09)
  mpu_write_byte(0x1C, 0x00);  // ±2g
  mpu_write_byte(0x1B, 0x00);  // ±250°/s
  mpu_write_byte(0x1A, 0x03);  // DLPF 44Hz
  mpu_write_byte(0x19, 0x09);  // ~100Hz

  delay(50);

  // Verify connection
  uint8_t whoami;
  mpu_read_bytes(0x75, &whoami, 1);
  return (whoami == 0x68);
}

struct RawData {
  float accel_x, accel_y, accel_z;
  float gyro_x, gyro_y, gyro_z;
  float temp_c;
};

RawData mpu_read_raw() {
  uint8_t buf[14];
  mpu_read_bytes(0x3B, buf, 14);

  RawData r;
  r.accel_x = (int16_t)((buf[0] << 8) | buf[1]);
  r.accel_y = (int16_t)((buf[2] << 8) | buf[3]);
  r.accel_z = (int16_t)((buf[4] << 8) | buf[5]);
  r.temp_c  = (int16_t)((buf[6] << 8) | buf[7]) / 340.0 + 36.53;
  r.gyro_x  = (int16_t)((buf[8] << 8) | buf[9]);
  r.gyro_y  = (int16_t)((buf[10] << 8) | buf[11]);
  r.gyro_z  = (int16_t)((buf[12] << 8) | buf[13]);

  return r;
}

// ═══════════════════════════════════════════════════════════════
//  Calibration
// ═══════════════════════════════════════════════════════════════

void calibrate() {
  Serial.println("{\"status\":\"calibrating\",\"cycles\":" + String(CALIBRATION_CYCLES) + "}");

  float ax = 0, ay = 0, az = 0;
  float gx = 0, gy = 0, gz = 0;

  for (int i = 0; i < CALIBRATION_CYCLES; i++) {
    RawData r = mpu_read_raw();
    ax += r.accel_x; ay += r.accel_y; az += r.accel_z;
    gx += r.gyro_x;  gy += r.gyro_y;  gz += r.gyro_z;
    delay(10);
  }

  cal.accel_x_offset = ax / CALIBRATION_CYCLES;
  cal.accel_y_offset = ay / CALIBRATION_CYCLES;
  cal.accel_z_offset = az / CALIBRATION_CYCLES - GRAVITY; // Expect 1g on Z
  cal.gyro_x_offset  = gx / CALIBRATION_CYCLES;
  cal.gyro_y_offset  = gy / CALIBRATION_CYCLES;
  cal.gyro_z_offset  = gz / CALIBRATION_CYCLES;

  calibrated = true;

  StaticJsonDocument<256> doc;
  doc["status"] = "calibrated";
  doc["ax_off"] = cal.accel_x_offset;
  doc["ay_off"] = cal.accel_y_offset;
  doc["az_off"] = cal.accel_z_offset;
  doc["gx_off"] = cal.gyro_x_offset;
  doc["gy_off"] = cal.gyro_y_offset;
  doc["gz_off"] = cal.gyro_z_offset;
  serializeJson(doc, Serial);
  Serial.println();
}

// ═══════════════════════════════════════════════════════════════
//  Angle Estimation (Complementary Filter)
// ═══════════════════════════════════════════════════════════════

void update_angles(RawData &r, float dt) {
  // Remove calibration offsets
  float ax = r.accel_x - cal.accel_x_offset;
  float ay = r.accel_y - cal.accel_y_offset;
  float az = r.accel_z - cal.accel_z_offset + GRAVITY;
  float gx = (r.gyro_x - cal.gyro_x_offset) / 131.0;   // °/s
  float gy = (r.gyro_y - cal.gyro_y_offset) / 131.0;   // °/s
  float gz = (r.gyro_z - cal.gyro_z_offset) / 131.0;   // °/s

  // Accelerometer angles (noisy but stable long-term)
  float accel_roll  = atan2(ay, az) * 180.0 / PI;
  float accel_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 180.0 / PI;

  // Complementary filter: gyro for short-term, accel for long-term
  // α = 0.98 typically
  float alpha = 0.98;
  roll_angle  = alpha * (roll_angle  + gx * dt) + (1 - alpha) * accel_roll;
  pitch_angle = alpha * (pitch_angle + gy * dt) + (1 - alpha) * accel_pitch;
  yaw_rate    = gz;  // Yaw drifts without magnetometer — just report rate
}

// ═══════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(100);

  Wire.begin(21, 22);  // SDA=21, SCL=22
  Wire.setClock(400000);  // 400kHz fast-mode

  Serial.println("{\"status\":\"boot\"}");

  if (!mpu_init()) {
    Serial.println("{\"error\":\"MPU6050 not found\"}");
    while (1) { delay(1000); }
  }

  Serial.println("{\"status\":\"mpu6050_ok\"}");

  calibrate();

  start_time = millis();
  Serial.println("{\"status\":\"running\"}");
}

void loop() {
  unsigned long now = millis();
  unsigned long elapsed = now - last_sample;

  float dt = elapsed / 1000.0;
  if (dt < 1.0 / SAMPLE_RATE_HZ) {
    return;
  }

  last_sample = now;

  RawData r = mpu_read_raw();
  update_angles(r, dt);

  // Compute roll rate (derivative)
  float roll_rate = (roll_angle - prev_roll) / dt;
  prev_roll = roll_angle;

  // Output JSON
  StaticJsonDocument<200> doc;
  doc["roll"] = round(roll_angle * 100) / 100.0;
  doc["pitch"] = round(pitch_angle * 100) / 100.0;
  doc["yaw_rate"] = round(yaw_rate * 100) / 100.0;
  doc["roll_rate"] = round(roll_rate * 100) / 100.0;
  doc["temp"] = round(r.temp_c * 10) / 10.0;
  doc["t"] = now - start_time;

  serializeJson(doc, Serial);
  Serial.println();

  sample_count++;

  // Heartbeat every 600 samples (60 seconds at 10Hz)
  if (sample_count % 600 == 0) {
    StaticJsonDocument<100> hb;
    hb["heartbeat"] = true;
    hb["uptime_s"] = (now - start_time) / 1000;
    hb["samples"] = sample_count;
    serializeJson(hb, Serial);
    Serial.println();
  }
}
