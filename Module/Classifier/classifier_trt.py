from Module.builder import DETECTS
from functools import partial
import queue
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
import torch
import torch.nn.functional as F
import cv2


@DETECTS.register_module()
class Classifier_trt():
    def __init__(self, modelname=None, result_type='classifier', model_version='',
                 sku_list=None):
        self.model_name_ = modelname
        self.result_type = result_type
        self.sku_list = sku_list
        self.model_version = model_version
        self.triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=False)
        try:
            self.model_metadata = self.triton_client.get_model_metadata(
                model_name=modelname, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))

        try:
            self.model_config = self.triton_client.get_model_config(
                model_name=modelname, model_version=model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))

        self.max_batch_size, self.input_name, self.output_name, self.dtype = parse_model_grpc(self.model_metadata,
                                                                                              self.model_config.config)

    def forward(self, img, scale_factor):
        repeatdata = []
        repeatdata.append(img)
        batched_image_data = np.stack(repeatdata, axis=0)
        responses = []
        sent_count = 0
        user_data = UserData()
        try:
            for inputs, outputs in requestGenerator(batched_image_data, self.input_name, self.output_name, self.dtype):
                sent_count += 1
                self.triton_client.async_infer(self.model_name_, inputs, partial(completion_callback, user_data),
                                               request_id=str(sent_count), model_version=self.model_version,
                                               outputs=outputs)
        except InferenceServerException as e:
            print("inference failed: " + str(e))

        processed_count = 0
        while processed_count < sent_count:
            (results, error) = user_data._completed_requests.get()
            processed_count += 1
            if error is not None:
                print("inference failed: " + str(error))
            responses.append(results)
        list_results = []

        for response in responses:
            list_results.append(self.gather_classier_result(response, self.output_name))
        return list_results

    def gather_classier_result(self, results, output_name):
        output_array = []
        final_result = []
        for i, name in enumerate(output_name):
            output_array.append(results.as_numpy(name)[0])
        result_list = F.softmax(torch.from_numpy(output_array[0]), dim=0).tolist()
        score = max(result_list)
        label = self.sku_list[result_list.index(score)]
        final_result.append([label, score])
        # final_result.append(result_list.index(max(result_list)))
        return final_result


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def parse_model_grpc(model_metadata, model_config):
    input_metadata = model_metadata.inputs[0]
    output_metadataname = []
    for name in model_metadata.outputs:
        output_metadataname.append(name.name)
    max_batch_size = 1
    return (max_batch_size, input_metadata.name, output_metadataname, input_metadata.datatype)


def requestGenerator(batched_image_data, input_name, output_name, dtype):
    inputs = []
    inputs.append(grpcclient.InferInput(input_name, batched_image_data.shape, dtype))
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = []
    for name in output_name:
        outputs.append(grpcclient.InferRequestedOutput(name))

    yield inputs, outputs
