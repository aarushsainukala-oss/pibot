import socket
import pigpio
import time
HOST = "0.0.0.0"
PORT = 5005
pi = pigpio.pi()
if not pi.connected:
    print("pigpio not connected")
    exit()

s = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)

s.bind((HOST, PORT))

print("Waiting for angles")


try :
 while True:
  data , addr = s.recvfrom(1024)
  message = data.decode()
  print("Recieved: " , message)

  joint , angle = message.split(":")
  angle = float(angle)
  servo_angle = max(0, min(180 , angle))
  pulse = 500 + (servo_angle / 180.0) * (2000)
  pi.set_servo_pulsewidth(18, pulse)


except KeyboardInterrupt:
  print("Program Stopped")
  pi.stop()