name = 'fanpai'

status = False

labels = ['real', 'ps', 'notsure']

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Classifier_resize_trt', resize_resolution=512, crop_resolution=448)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Classifier_trt', modelname=name, result_type='classifier', sku_list=labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Fp_cl')
         ])
]
