name = 'direction'

status = True
labels = ['left90', 'normal', 'right90', 'updown180']
cls_threshold = 0.9

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Classifier_resize_trt', resize_resolution=224, crop_resolution=224)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Classifier_trt', modelname='direction', result_type='classifier',
                  sku_list=labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Direction_cl', threshold=cls_threshold)
         ])
]
