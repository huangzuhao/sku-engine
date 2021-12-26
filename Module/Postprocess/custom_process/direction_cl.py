from Module.builder import POSTPROCESSES


@POSTPROCESSES.register_module()
class Direction_cl():
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, resultdata):
        tempresult = resultdata["classifier"][0][0]
        direction = 0
        if tempresult[-1] > self.threshold:
            if tempresult[0] == 'normal':
                direction = 0
            if tempresult[0] == 'left90':
                direction = -90
            if tempresult[0] == 'right90':
                direction = 90
            if tempresult[0] == 'updown180':
                direction = 180
        return direction
