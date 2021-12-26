name = '10091'

status = True

sku_labels = ["yl_gs_blp1250P", "yl_gs_blp330LB", "yl_gs_blp330P", "yl_gs_blp330pjP", "yl_gs_blp330qtP",
              "yl_gs_blp500LB", "yl_gs_blp500P", "yl_gs_gwqps480P", "yl_gs_mtqp330P", "yl_gs_sxpj500P"]
bottle_labels = ['daiding_101']

sku_threshold = {"Default": 0.5}
bottle_threshold = {"Default": 0.5}

layer_config = {"Func": "Fine", "Default": 0.9}
stack_config = {"Enable": False, "Func": "Raw"}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='gs', threshold=sku_threshold, with_mask=False,
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
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='gs', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
