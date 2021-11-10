import Jetson.GPIO as JGPIO

class GPIO():
    def __init__(self, pin: int):
        self.__pin = pin
        self.__status = 1
        JGPIO.setmode(JGPIO.BOARD)
        self.setup()
    
    def setup(self):
        JGPIO.setup(self.__pin, JGPIO.OUT, initial=JGPIO.HIGH)

    def toggle(self):
        self.__status = 0 if self.__status == 1 else 1
        JGPIO.output(self.__pin, self.__status)
    
    def on(self):
        self.__status = 0
        JGPIO.output(self.__pin, self.__status)
    
    def off(self):
        self.__status = 1
        JGPIO.output(self.__pin, self.__status)

    def release(self):
        JGPIO.cleanup(self.__pin)