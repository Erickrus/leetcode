import time
import threading

a = 0
class GracefullyStop:
    def __init__(self, stopping=False):
        self.stopping = stopping
        
def a_piece_of_code(gracefullyStop):
    global a 
    for i in range(20):
        a +=1
        print(a)
        time.sleep(3)
        if gracefullyStop.stopping:
            return
        
class Runnable(threading.Thread):
    def __init__(self, gracefullyStop, *args, **kargs):
        super().__init__(*args, **kargs)
        self.gracefullyStop = gracefullyStop
    def run(self):
        a_piece_of_code(self.gracefullyStop)

gs = GracefullyStop()
r = Runnable(gs)
r.start()
