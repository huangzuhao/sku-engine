import os

import sanic
import json as Json
from sanic.response import json
from sanic import Sanic, request, response
from DE_logger import DE_LOGGING_CONFIG_DEFAULTS, logger
import time
import multiprocessing
import argparse

app = Sanic('SKUDetector_service', log_config=DE_LOGGING_CONFIG_DEFAULTS)

from Core.main_workflow import customers_init, customers_inference, customers_inference_simple
from PIL import Image


@app.route('/detService/skuDet', methods=['POST'])
async def skuDet(request):
    # data = request.body.decode('utf-8')
    # data_json = Json.loads(data)
    # ID = str(data_json['headers']['tenantId'])
    result = {"status": "0", "data": 'detect fail!'}
    try:
        # base64_data = data_json['base64Data']
        base64_data = request.form.get('base64Data')
        ID = str(request.headers.get('tenantId'))
        if base64_data is not None:
            resultoutput = customers_inference_simple(ID, base64_data)
            result['status'] = "1"
            result['data'] = resultoutput
            logger.info(result)
        else:
            logger.error("get bad base64data")
            result['data'] = "get bad base64data"

    except Exception as e:
        logger.error(repr(e))
        result['status'] = "0"
        result['data'] = repr(e)
    finally:
        return json(result)


@app.route('/detService/layerDet', methods=['POST'])
async def layerDet(request):
    # data = request.body.decode('utf-8')
    # data_json = Json.loads(data)
    # ID = str(data_json['headers']['tenantId'])
    result = {"status": "0", "data": 'detect fail!'}
    try:
        # base64_data = data_json['base64Data']
        base64_data = request.form.get('base64Data')
        ID = str(request.headers.get('tenantId'))
        if base64_data is not None:
            resultoutput = customers_inference(ID, base64_data)
            result['status'] = "1"
            result['data'] = resultoutput
            logger.info(result)
        else:
            logger.error("get bad base64data")
            result['data'] = "get bad base64data"

    except Exception as e:
        logger.error(repr(e))
        result['status'] = "0"
        result['data'] = repr(e)
    finally:
        return json(result)


@app.route('/detService/sceneAndLayerDet', methods=['POST'])
async def sceneAndLayerDet(request):
    result = {"status": "0", "data": 'detect fail!'}
    try:
        # base64_data = data_json['base64Data']
        base64_data = request.form.get('base64Data')
        ID = str(request.headers.get('tenantId'))
        if base64_data is not None:

            img_direction = customers_inference("direction", base64_data)
            resultoutput = customers_inference(ID, base64_data, direction=img_direction)
            resultoutput['imageInfo']['direction'] = img_direction
            result['status'] = "1"
            result['data'] = resultoutput
            logger.info(result)
        else:
            logger.error("get bad base64data")
            result['data'] = "get bad base64data"

    except Exception as e:
        logger.error(repr(e))
        result['status'] = "0"
        result['data'] = repr(e)
    finally:
        return json(result)


@app.route('/imageprocess/signboard', methods=['POST'])
async def skuDet(request):
    data = request.body.decode('utf-8')
    data_json = Json.loads(data)
    ID = 'dtcj'
    result = {"status": "0", "data": 'detect fail!'}
    try:
        base64_data = data_json['base64Data']
        if base64_data is not None:
            resultoutput = customers_inference(ID, base64_data)
            result['status'] = "1"
            result['data'] = resultoutput
            logger.info(result)
        else:
            logger.error("get bad base64data")
            result['data'] = "get bad base64data"

    except Exception as e:
        logger.error(repr(e))
        result['status'] = "0"
        result['data'] = repr(e)
    finally:
        return json(result)


@app.route('/mestrics', methods=['POST'])
async def check_mestrics(request):
    data = request.body.decode('utf-8')
    data_json = Json.loads(data)
    ID = 'swin'
    result = {"status": "0", "data": {'statu': 'detect fail!', 'result': 0}}
    try:
        base64_data = data_json['base64Data']
        if base64_data is not None:
            resultoutput = customers_inference(ID, base64_data)
            result['status'] = "1"
            result['data']['result'] = resultoutput
            result['data']['statu'] = 'detect success!'
            logger.info(result)
        else:
            logger.error("get bad base64data")
            result['data']['statu'] = "get bad base64data"

    except Exception as e:
        logger.error(repr(e))
        result['status'] = "0"
        result['data']['result'] = 0
        result['data']['statu'] = repr(e)
    finally:
        return json(result)


def parse_args():
    parser = argparse.ArgumentParser(description='start service')
    parser.add_argument('--port', '-p', default='9190', help='port')
    argss = parser.parse_args()
    return argss


if __name__ == '__main__':
    argss = parse_args()
    port = int(argss.port)
    customers_init()
    workers = multiprocessing.cpu_count()
    app.run(host='0.0.0.0', port=port, workers=workers, access_log=False, debug=False, auto_reload=False)
