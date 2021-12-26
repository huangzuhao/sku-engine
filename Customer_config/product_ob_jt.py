name = '10016'

status = True
sku_labels = ['__ignore__', 'yl_jt_bss_1000', 'yl_jt_bss_1000X', 'yl_jt_bss_348', 'yl_jt_bss_348LLB', 'yl_jt_bss_348X',
              'yl_jt_bss_4500', 'yl_jt_bss_4500X', 'yl_jt_bss_570', 'yl_jt_bss_570LLB', 'yl_jt_bss_570X',
              'yl_jt_jt_1500', 'yl_jt_jt_1500SEP', 'yl_jt_jt_1500X', 'yl_jt_jt_360', 'yl_jt_jt_360ESS',
              'yl_jt_jt_360SEP', 'yl_jt_jt_360X', 'yl_jt_jt_4600', 'yl_jt_jt_4600X', 'yl_jt_jt_560', 'yl_jt_jt_560ESS',
              'yl_jt_jt_560SEP', 'yl_jt_jt_560X']
bottle_labels = ['daiding_101']

sku_threshold = {"Default": 0.4}
bottle_threshold = {"Default": 0.5}

layer_config = {"Func": "Fine", "Default": 0.9}
stack_config = {"Enable": False, "Func": "Raw"}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1000, height=700)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='jt', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels),
             dict(type='Object_detect_trt', modelname='bottle', threshold=bottle_threshold, with_mask=False,
                  result_type='others', sku_list=bottle_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_computelayer', stack_config=stack_config,
                  layer_config=layer_config)
         ])
]

simple_process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1000, height=700)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='jt', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
