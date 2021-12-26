name = '10068'

status = True

sku_labels = ["ignore", "yl_yy_lght_tjlzxrl240D16", "yl_yy_lght_tjlzD15", "yl_yy_lght_dtjpx240P",
              "yl_yy_lght_dtjpx240X10", "yl_yy_lght_dtjpx240X12", "yl_yy_lght_dtjpx_tsz240X12",
              "yl_yy_lght_dtjpx240X20", "yl_yy_lght_jyx240P", "yl_yy_lght_jyx240LB", "yl_yy_lght_jyx240X20",
              "yl_yy_lght_jxx240P", "yl_yy_lght_jxx240X20", "yl_yy_lght_tjlzx15", "yl_yy_lght_Ctjlz240P",
              "yl_yy_lght_Ctjlz240X24", "yl_yy_lght_wt240P", "yl_yy_lght_wt240D20", "yl_yy_lght_wt240X20",
              "yl_yy_lght_zhys180P", "yl_yy_lght_zhys180X20", "yl_yy_lght_Cwt240P", "yl_yy_lght_Cwt480X12",
              "yl_yy_lght_dtzhys240P", "yl_yy_lght_dtzhys240X15", "yl_yy_lght_zhxs240P", "yl_yy_lght_zhxs240X30",
              "yl_yy_lght_jdx240P", "yl_yy_lght_jdx240D12", "yl_yy_lght_jdx240D12X2", "yl_yy_lght_jdx240X12",
              "yl_yy_lght_jdx240D20", "yl_yy_lght_jdx240X20", "yl_yy_lght_yzwx240P", "yl_yy_lght_yzwx240D20",
              "yl_yy_lght_yzwx240X20", "yl_yy_lght_yzzy240P", "yl_yy_lght_jpx240P", "yl_yy_lght_jpx270X12",
              "yl_yy_lght_tjlz270P", "yl_yy_lght_tjlz270X12", "yl_yy_lght_fxz1000P", "yl_yy_lght_fxz1000X6",
              "yl_yy_lght_hsl310P", "yl_yy_lght_hsl310D15", "yl_yy_lght_hsl310X15", "yl_yy_lght_tjlzxrl240P",
              "yl_yy_lght_tjlzxrl240X16", "yl_yy_lght_tjlzxrl240X20", "yl_yy_lght_mc+240P", "yl_yy_lght_mc+240D20",
              "yl_yy_lght_mc+240X20", "yl_yy_lght_mc+240X10", "yl_yy_lght_mc+1000P", "yl_yy_lght_mc+1000D6",
              "yl_yy_lght_mc+250P", "yl_yy_lght_mc+250X16", "yl_yy_lght_mc+250X6", "yl_yy_lght_mc+250X6X4",
              "yl_yy_lght_mc+CX", "yl_ll_rll240X24", "yl_ll_rll240P", "yl_ll_yw240X20", "yl_ll_yw240P",
              "yl_ll_dgll310X15", "yl_ll_dgll310P", "yl_ll_ll480P", "yl_ll_ll480X15", "yl_ll_wtll240P",
              "yl_ll_wtll240X24", "yl_ys_yz245P", "yl_ys_yz245X25", "yl_ys_llyz245X24", "yl_ys_llyz245LB",
              "yl_ys_llyz245P", "yl_ys_llyz1000X12", "yl_ys_llyz1000P", "yl_gf_szsx350X15", "yl_gf_szsx350P",
              "yl_gf_szsx1250P", "yl_gf_szsx1250X6", "yl_hn_wssfw250P", "yl_hn_wssgn250P", "yl_hn_anj250P",
              "yl_hn_wssgn250LB", "yl_hn_wssfw250LB", "yl_hn_wssfw250X24", "yl_hn_wssgn250X24", "yl_hn_anj250X24",
              "yl_yy_lght_stdK", "yl_yy_lght_stdC", "yl_yy_lght_stdYS", "yl_yy_lght_mc+250", "yl_hn_250P",
              "yl_yy_lght_sidejpjyjx240X20", "fake_fxz", "fake_hnstd", "yl_yy_clm", "yl_yy_lght_tjlz240X10",
              "yl_yy_lght_stdGK", "yl_yy_lght_2430_240P", "yl_yy_lght_2430_240X15", "yl_yy_lght_2430_240D15",
              "yl_yy_lght_yzyx_280P", "yl_yy_lght_yzyx_280X12", "yl_yy_lght_yzyx_280D12", "yl_yy_lght_nyg_240X20",
              "yl_yy_lght_zwn250P", "yl_yy_lght_zwn250X10", "yl_yy_lght_zwn250D10", "yl_yy_lght_jpx250P",
              "yl_yy_lght_jpx250X12", "yl_yy_lght_jpx250D12"]
bottle_labels = ['daiding_101']
boxes_labels = ['ignore', 'daidingXz', 'yyLogo']


sku_threshold = {"Default": 0.45}
bottle_threshold = {"Default": 0.5}
boxes_threshold = {"Default": 0.25}


layer_config = {"Func": "Fine", "Default": 0.9}
stack_config = {"Enable": False, "Func": "Raw"}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1000, height=700)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='yy', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels),
             dict(type='Object_detect_trt', modelname='bottle', threshold=bottle_threshold, with_mask=False,
                  result_type='others', sku_list=bottle_labels),
             dict(type='Object_detect_trt', modelname='yybox', threshold=boxes_threshold, with_mask=False,
                  result_type='boxes', sku_list=boxes_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Yy_ob_computelayer', stack_config=stack_config,
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
             dict(type='Object_detect_trt', modelname='yy', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Yy_ob_sku')
         ])
]
