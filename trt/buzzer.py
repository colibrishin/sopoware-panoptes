import Jetson.GPIO as GPIO

class GPIO():
    def __init__(self, pin: int):
        self.pin = pin
        self.status = 1
        GPIO.setmode(GPIO.BOARD)
        
        self.setup()
    
    def setup(pin: int):
        GPIO.setup(pin, GPIO.OUT, inital=GPIO.HIGH)

    def toggle():
        self.status = 0 if self.status == 1 else 1
        GPIO.output(self.pin, self.status)

    