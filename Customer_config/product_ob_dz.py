name = '10089'
# dazhong bkl
status = True

sku_labels = ['yl_bkl_4LBst500P', 'yl_bkl_6LBst350P', 'yl_bkl_stgtyl15g', 'yl_bkl_stgtyl15gX5', 'yl_bkl_wl',
              'yl_bkl_wsxyyy550P', 'yl_bkl_jq', 'yl_bkl_st350P', 'yl_bkl_st500P', 'yl_bkl_st900P', 'yl_md_btkw450p',
              'yl_md_qnkw600p', 'yl_md_tzkw600p', 'yl_md_xsnm450p', 'yl_md_xyjz600p', 'yl_jdl_cw600p', 'yl_jdl_lm600p',
              'yl_bkl_zsp', 'yl_jdl_nm600p', 'yl_jdl_xy600p', 'yl_bkl_nfsq_jj_xy', 'yl_bkl_nfsq_jj_nm',
              'yl_bkl_8LBst500P', 'yl_bkl_stgtyl15gX5X5']
bottle_labels = ['daiding_101']
# cjfc_labels = ["ignore", "layer", "cj_hj"]
cjfc_labels = ["ignore", "ty_cj_logo_ty", "cj_wsbg", "cj_hj", "cj_lsbx", "cj_tg", "cj_mit", "cj_clgj", "cj_gx",
               "cj_clj", "layer"]
sku_threshold = {"Default": 0.5}
bottle_threshold = {"Default": 0.5}
cjfc_threshold = {"Default": 0.43}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='dz', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels),
             dict(type='Object_detect_trt', modelname='bottle', threshold=bottle_threshold, with_mask=False,
                  result_type='others', sku_list=bottle_labels),
             dict(type='Object_detect_trt', modelname='cjfc', threshold=cjfc_threshold, with_mask=True,
                  result_type='cjfc', sku_list=cjfc_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Dz_ob_modellayer')
         ])
]

simple_process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='dz', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Dz_ob_sku')
         ])
]
