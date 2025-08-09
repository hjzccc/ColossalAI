class StepCounter:
    def __init__(self):
        self.value = 0
    
    def get(self):
        return self.value
    
    def set(self, value):
        self.value = value
    
    def increment(self, step=1):
        self.value += step
        return self.value
    
    def reset(self):
        self.value = 0