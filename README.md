# pibot
<br>
Inverse Kinematics : IK2
<br>
Dot Product : dotproduct
<br>
pigpoid.service:
<br>
sudo systemctl daemon-reload
<br>
sudo systemctl enable pigpiod
<br>
check status : sudo systemctl status pigpiod
<br>
Servo motor:
<br>
`1. enable I2C in interface options using raspi-config
<br>
2. Verify using 
<br>
i2c detect -y 1
3. Install PCA9685 library :
<br>
pip3 install adafruit-circuitpython-pca9685
4. Install system GPIO package :
<br>
sudo apt install -y python3-rpi.gpio
