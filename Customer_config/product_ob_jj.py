name = '10092'

status = True

sku_labels = ["__ignore_", "jl_jj_52_sef9n_480", "jl_jj_38_5ncsz_455L", "jl_jj_33_5ncsz_455H", "jl_jj_35_dsxz_150",
              "jl_jj_53_jp_5.1L", "jl_jj_38_sz_38d_5.1L", "jl_jj_29.5_sz_5.1L", "jl_jj_29.5_kbctj_610",
              "jl_jj_29.5_kbdj_500", "jl_jj_29.5_xijiu_305", "jl_jj_29.5_xijing_250", "jl_jj_43_jm40_500",
              "jl_jj_29.5_bj_156", "jl_jj_42_jm30+_500", "jl_jj_30.5_jm30_500", "jl_jj_29.5_ctjp_610",
              "jl_jj_29.5_dj_500", "jl_jj_38_5ncsz_455L_fake", "jl_jj_33_5ncsz_455H_fake", "jl_jj_29.5_dj_500_fake",
              "jl_jj_52_sef9n_480_fake", "jl_jj_42_jm30+_500_fake", "jl_jj_5.1L_fake", "jl_jj_29.5_xijiu_305_fake",
              "jl_jj_35_dsxz_150_fake", "jl_jj_30.5_jm30_500_fake", "jl_jj_29.5_ctjp_610_fake",
              "jl_jj_29.5_bj_156_fake"]
bottle_labels = ["daiding_101"]
cjfc_labels = ["ignore", "layer", "cj_hj"]

sku_threshold = {"Default": 0.45}
bottle_threshold = {"Default": 0.5}
cjfc_threshold = {"Default": 0.43}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='jj', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels),
             dict(type='Object_detect_trt', modelname='bottle', threshold=bottle_threshold, with_mask=False,
                  result_type='others', sku_list=bottle_labels),
             dict(type='Object_detect_trt', modelname='jjcjfc', threshold=cjfc_threshold, with_mask=True,
                  result_type='cjfc', sku_list=cjfc_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_modellayer')
         ])
]

simple_process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='jj', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
