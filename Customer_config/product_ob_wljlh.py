name = '10060'

status = True
sku_labels = ['ignore', 'yl_wlj_cnj_rht_28g', 'yl_wlj_cnj_rht_56g', 'yl_wlj_lblc250', 'yl_wlj_lc250_X16',
              'yl_wlj_lc250_X24', 'yl_wlj_lc250', 'yl_wlj_rht_28g', 'yl_wlj_rht_56g', 'yl_wlj_rhtH', 'yl_wlj_cnj_rhtH',
              'yl_wlj_szcp500', 'yl_wlj_xgj500', 'yl_wlj_zcz', 'fake']
sku_threshold = {"Default": 0.5}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1000, height=700)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='wlj', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
