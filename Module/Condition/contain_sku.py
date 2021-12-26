from Module.builder import CONDITION


@CONDITION.register_module()
class Contain_sku():
    def __init__(self, data_src='cjfc', contain_sku=['a', 'b']):
        self.data_src = data_src
        self.contain_sku = contain_sku

    def process(self, result):
        state = False
        for sku in result[self.data_src]:
            if sku[5] in self.contain_sku:
                state = True
                break
        return state
