name = '10072'

status = True

sku_labels = ["ignore", "yl_wlj_wlj_500P", "yl_jdb_310XTJG", "yl_jdb_1500HP_X6", "yl_jdb_250HH_X12_LH__12x250",
              "yl_jdb_310JG", "yl_jdb_250HH_X16_LH", "yl_jdb_310XTJG_X20", "yl_jdb_250XTHH_X24_CY", "yl_wlj_wlj_310SG",
              "yl_wlj_wlj_310G_L6", "yl_jdb_250HH", "yl_jdb_330MINIP_X24", "yl_jdb_250HG_LD20", "yl_jdb_310JG_X15",
              "yl_jdb_1250HP_X6__6x1250", "yl_jdb_250HH_X16", "yl_jdb_310HG_L6", "yl_wlj_wlj_500P_1",
              "yl_jdb_250HH_X20_LH",
              "yl_jdb_310JG_X20", "yl_jdb_1250HP", "yl_jdb_250XTHH_L6", "yl_jdb_1500HP", "yl_jdb_1000HH_X10",
              "yl_jdb_250HH_L6", "yl_jdb_1500HP_X6__6x1500", "yl_jdb_250XTHH_X20_LH", "yl_jdb_250XTHH_X20_LH__20x250",
              "yl_jdb_310JG_X12", "yl_jdb_250HH_LD20", "yl_jdb_310JG_X24", "yl_jdb_310HG_X15",
              "yl_jdb_1000HH_X10_10X1000",
              "yl_jdb_250XTHH_X16_LH__16x250", "yl_jdb_250XTHH_X18", "yl_jdb_330MINIP", "yl_jdb_250HH_X20_LH__20x250",
              "yl_jdb_310XTHG_X12", "yl_jdb_310HG_LD20", "yl_jdb_250XTHH_X24", "yl_wlj_wlj_1500",
              "yl_jdb_250XTHH_X16_LH",
              "yl_jdb_250HH_X16_LH__20x250", "yl_jdb_310HG", "yl_jdb_250HH_X12_LH", "yl_wlj_wlj_310G",
              "yl_jdb_250HH_X24",
              "yl_jdb_1250HP_X6", "yl_jdb_250XTHH", "yl_hqz_hqz_550", "yl_jdb_1000HH", "yl_jdb_550HP_X15",
              "yl_jdb_310HG_X24", "yl_jdb_310XTHG", "yl_jdb_250XTHH_X16", "yl_jdb_310HG_X24_CY2", "yl_jdb_310XTJG_X24",
              "yl_jdb_310HG_X20", "yl_jdb_550HP", "yl_jdb_310HG_X12", "yl_jdb_310XTJG_X10",
              "yl_jdb_250HH_X16_LH__16x250"]

bottle_labels = ['daiding_101']

sku_threshold = {"Default": 0.45}
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
             dict(type='Object_detect_trt', modelname='jdb', threshold=sku_threshold, with_mask=False,
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
             dict(type='Object_detect_trt', modelname='jdb', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
