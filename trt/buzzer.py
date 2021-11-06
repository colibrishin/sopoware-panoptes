import Jetson.GPIO as JGPIO

class GPIO():
    def __init__(self, pin: int):
        self.__pin = pin
        self.__status = 1
        JGPIO.setmode(JGPIO.BOARD)
    
    def setup():
        JGPIO.setup(self.__pin, JGPIO.OUT, initial=JGPIO.HIGH)

    def toggle():
        self.__status = 0 if self.__status == 1 else 1
        JGPIO.output(self.__pin, self.__status)

    def release():
        JGPIO.cleanup(self.__pin)