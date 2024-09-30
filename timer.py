import time
class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        if self.end_time is None:
            raise ValueError("Timer has not been stopped.")

        elapsed_seconds = self.end_time - self.start_time
        hours = int(elapsed_seconds // 3600)
        elapsed_seconds %= 3600
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60

        return str(f"{hours} hours, {minutes} minutes, {seconds:.2f} seconds")