import os
from Core.build.config import Config
from Core.customer_pipline import Customer_pipline

config_path = 'Customer_config'
all_customer = dict()


def customers_init():
    for file in os.listdir(config_path):
        if file.endswith('.py'):
            cfg = Config.fromfile(os.path.join(config_path, file))
            if cfg.status:
                all_customer[cfg.name] = cfg
    # print(all_customer)


def customers_inference(customID, img, **kwargs):
    pipline = Customer_pipline(all_customer[customID])
    result = pipline.do_inference(img, **kwargs)
    return result


def customers_inference_simple(customID, img, **kwargs):
    pipline = Customer_pipline(all_customer[customID], mode="simple")
    result = pipline.do_inference(img)
    return result
