from Module.builder import POSTPROCESSES


@POSTPROCESSES.register_module()
class Fp_cl():

    def process(self, resultdata):
        # tempresult = resultdata["classifier"][0][0]
        # if tempresult == 2:
        #     tempresult = 0
        # return tempresult
        tempresult = resultdata["classifier"][0][0]
        fp_result = 0
        if tempresult[0] == 'real':
            fp_result = 0
        if tempresult[0] == 'ps':
            fp_result = 1
        if tempresult[0] == 'notsure':
            fp_result = 0

        return fp_result
