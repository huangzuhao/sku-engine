name = '8001'

status = True
sku_labels = ['jl_jsy_gy_42dygyp1', 'jl_jsy_gy_dkp', 'jl_jsy_gy_rygyp', 'jl_jsy_jsy_dc15p', 'jl_jsy_jsy_jznp',
              'jl_jsy_jsy_lznp', 'jl_jsy_gy_rygyp2', 'jl_jsy_gy_rygyp1', 'jl_jsy_gy_skp1', 'jl_jsy_gy_dkp1',
              'jl_jsy_gy_42dygyp', 'jl_jsy_jsy_dqhp', 'jl_jsy_jsy_xqhp', 'jl_jsy_gy_42dk4', 'jl_jsy_jsy_dc15p1',
              'jl_jsy_jsy_dc10p1', 'jl_jsy_jsy_dc5p', 'jl_jsy_gy_dkp3', 'jl_jsy_gy_dygyp', 'jl_jsy_jsy_dc12p',
              'jl_jsy_jsy_dc6p', 'jl_jsy_jsy_ldqp', 'jl_jsy_gy_skp', 'jl_jsy_jsy_dc10p', 'jl_jsy_gy_k5p',
              'jl_jsy_gy_k3p', 'jl_jsy_jsy_dc5p2', 'jl_jsy_gy_sjv3p', 'jl_jsy_gy_dkp2', 'jl_jsy_gy_sjv6p',
              'jl_jsy_jsy_dc5p1', 'jl_jsy_jsy_dc18p', 'yl_tdyh_tdyh_cc', 'yl_tdyh_tdyh_ccX', 'yl_tdyh_tdyh_lv300X',
              'yl_tdyh_tdyh_pgbX', 'yl_tdyh_tdyh_pgc', 'yl_tdyh_tdyh_pgcG', 'yl_tdyh_tdyh_pgcLLB',
              'yl_tdyh_tdyh_pgcbjz', 'yl_tdyh_tdyh_pgcbjzX', 'yl_tdyh_tdyh_cc_fake', 'yl_tdyh_tdyh_lv300X_fake',
              'yl_tdyh_tdyh_pgbX_fake', 'yl_tdyh_tdyh_pgcG_fake', 'yl_tdyh_tdyh_pgcLLB_fake', 'yl_tdyh_tdyh_pgc_fake',
              'yl_tdyh_tdyh_pgcbjz_fake', 'yl_tdyh_tdyh_pgcbjzX_fake', 'yl_tdyh_tdyh_ccX_fake', 'nf_ysl_hb_1', 'fake']

bottle_labels = ['daiding_101']
sku_threshold = {"Default": 0.6}
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
             dict(type='Object_detect_trt', modelname='jsy', threshold=sku_threshold, with_mask=False,
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
             dict(type='Object_detect_trt', modelname='jsy', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
