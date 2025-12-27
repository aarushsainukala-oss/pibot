import socket
import board
import busio
from adafruit_pca9685 import PCA9685

# ================= NETWORK =================
HOST = "0.0.0.0"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print("Listening for joint angles...")

# ================= I2C + PCA9685 =================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)

pca.frequency = 50  # Servo frequency

# ================= SERVO CONFIG =================
SERVO_MIN = 500    # µs
SERVO_MAX = 2500   # µs

def angle_to_duty(angle):
    angle = max(0, min(180, angle))
    pulse_us = SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN)
    duty = int(pulse_us * 65535 / 20000)  # 20 ms period
    return duty

# ================= SERVO CHANNEL MAP =================
SERVO_MAP = {
    "R_SHOULDER": 0,
    "R_ELBOW": 4,
    "L_SHOULDER": 8,
    "L_ELBOW": 12
}

# ================= MAIN LOOP =================
while True:
    data, addr = sock.recvfrom(1024)
    msg = data.decode().strip()
    print("Received:", msg)

    try:
        joint, angle = msg.split(":")
        angle = float(angle.strip())

        if joint in SERVO_MAP:
            ch = SERVO_MAP[joint]
            duty = angle_to_duty(angle)
            pca.channels[ch].duty_cycle = duty

    except Exception as e:
        print("Error:", e)

