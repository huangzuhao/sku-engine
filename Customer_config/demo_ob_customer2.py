name = 'layerdetect'

status = False

sku_labels = ['yl_bkl_4LBst500P', 'yl_bkl_6LBst350P', 'yl_bkl_stgtyl15g', 'yl_bkl_stgtyl15gX5', 'yl_bkl_wl',
              'yl_bkl_wsxyyy550P', 'yl_bkl_jq', 'yl_bkl_st350P', 'yl_bkl_st500P', 'yl_bkl_st900P', 'yl_md_btkw450p',
              'yl_md_qnkw600p', 'yl_md_tzkw600p', 'yl_md_xsnm450p', 'yl_md_xyjz600p', 'yl_jdl_cw600p', 'yl_jdl_lm600p',
              'yl_bkl_zsp', 'yl_jdl_nm600p', 'yl_jdl_xy600p', 'yl_bkl_nfsq_jj_xy', 'yl_bkl_nfsq_jj_nm',
              'yl_bkl_8LBst500P', 'yl_bkl_stgtyl15gX5X5']
bottle_labels = ['daiding_101']
sku_threshold = {"Default": 0.5, "Specify": {"yl_bkl_4LBst500P": 0.51, "yl_bkl_6LBst350P": 0.52}}
bottle_threshold = {"Default": 0.35}
stack_config = {"Enable": False, "Func": "Raw"}
layer_config = {"Func": "Raw", "Default": 0.9}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='bkl', threshold=sku_threshold, with_mask=False,
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
