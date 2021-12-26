from Module.builder import build_detect, build_postprocess, build_preprocess, build_condition
import Module.Preprocess
import Module.Object_detect
import Module.Postprocess
import Module.Classifier
import Module.Condition


# import Module.Resultoutput


class Customer_pipline():

    def __init__(self, cfg, mode="normal"):
        self.name = cfg.name
        self.preprocess = None
        self.detect = None
        self.postprocess = None
        self.conditiondetect = []
        if mode == "simple":
            self.get_pipline(cfg.simple_process_pipeline)
        else:
            self.get_pipline(cfg.process_pipeline)

    def get_pipline(self, custom_pipline):
        for process in custom_pipline:
            if process['type'] == 'Preprocess':
                self.preprocess = build_preprocess(process['pipline'])
            elif process['type'] == 'Detect':
                self.detect = build_detect(process['pipline'])

            elif process['type'] == 'ConditionDetect':
                conditiontemp = {}
                for conditionprocess in process['pipline']:
                    conditiontemp["condition"] = build_condition(conditionprocess['condition'])
                    conditiontemp["pipline"] = build_detect(conditionprocess['pipline'])
                    self.conditiondetect.append(conditiontemp)

            elif process['type'] == 'Postprocess':
                self.postprocess = build_postprocess(process['pipline'])
            else:
                raise TypeError('can not find suit pipline')

    def do_inference(self, img, **kwargs):
        if self.preprocess is not None:
            for subprocess in self.preprocess:
                img, scale_factor, origal_w, origal_h = subprocess(img, **kwargs)

        result = dict()
        result["imageInfo"] = {"width": origal_w, "height": origal_h}
        if self.detect is not None:
            for subdetect in self.detect:
                result[subdetect.result_type] = subdetect.forward(img, scale_factor)

        if len(self.conditiondetect) > 0:
            for subcondtiondetect in self.conditiondetect:
                ret = subcondtiondetect['condition'].process(result)
                if ret:
                    for subcprocess in subcondtiondetect['pipline']:
                        result[subcprocess.result_type] = subcprocess.forward(img, scale_factor)

        # print("result:  ", result)
        if self.postprocess is not None:
            for subprocess in self.postprocess:
                # result = subprocess(result)
                result = subprocess.process(result)
        # print("result:  ", result)
        return result
