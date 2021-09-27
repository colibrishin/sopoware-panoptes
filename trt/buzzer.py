import Jetson.GPIO as JGPIO

class GPIO():
    def __init__(self, pin: int):
        self.pin = pin
        self.status = 1
        JGPIO.setmode(JGPIO.BOARD)
        
        self.setup()
    
    def setup(pin: int):
        JGPIO.setup(pin, JGPIO.OUT, inital=JGPIO.HIGH)

    def toggle():
        self.status = 0 if self.status == 1 else 1
        JGPIO.output(self.pin, self.status)

    def release():
        JGPIO.cleanup(self.pin)

    
