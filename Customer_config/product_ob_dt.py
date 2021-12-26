name = 'dtcj'

status = True
sku_labels = ['signboard', 'sampler']
sku_threshold = {"Default": 0.85}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1000, height=600)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname=name, threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Dt_ob_sku')
         ])
]
