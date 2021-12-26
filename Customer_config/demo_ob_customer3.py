name = 'conditiondetect'
status = False

sku_labels = ['__ignore__', 'a_dl_101', 'a_dl_101_fake', 'a_dl_102', 'a_dl_102_fake', 'a_dn_101', 'a_dp_101',
              'a_dp_102', 'a_dp_103', 'a_dp_104', 'a_gwnl_101', 'a_hn_101', 'a_hn_101_dz', 'a_hn_101_fake',
              'a_hn_101_llb', 'a_hn_101bj', 'a_hn_102', 'a_hn_102_fake', 'a_hn_201', 'a_hn_201_fake', 'a_hn_202',
              'a_hn_202_fake', 'a_hn_203', 'a_hn_203_fake', 'a_hn_301', 'a_hn_301_fake', 'a_hn_301hh', 'a_hn_302',
              'a_hn_302_fake', 'a_hn_302hh', 'a_hn_303', 'a_hn_304', 'a_hn_305', 'a_hn_306', 'a_hn_307',
              'a_hn_307_fake', 'a_hn_308', 'a_hn_308_fake', 'a_hn_309', 'a_hn_309_fake', 'a_hn_401', 'a_hn_401_BX',
              'a_hn_401_fake', 'a_hn_402', 'a_hn_403', 'a_hn_404', 'a_hn_405', 'a_hn_501', 'a_hn_501_fake', 'a_hn_502',
              'a_hn_503', 'a_hn_601', 'a_hn_602', 'a_hn_603', 'a_hn_anj', 'a_hn_anjbj', 'a_zw_101', 'a_zw_102',
              'yl_cola_bjx330G', 'yl_cola_mtjy330G', 'yl_cola_mtld330G', 'yl_cola_mttx330G', 'yl_cola_mtwt330G',
              'yl_cola_mtxw330G', 'yl_cola_mtyw330G', 'yl_cola_yw330G', 'yl_dp_dpxz_24', 'yl_fs_gzfs330_pet_24',
              'yl_fs_gzfs500_pet_24', 'yl_fs_gzfs500_pet_24_hg', 'yl_fs_qnsj330_pet_12', 'yl_fs_qtpt330_pet_12',
              'yl_fs_yhbt330_pet_12', 'yl_hn_gbs_bptz200_6', 'yl_hn_gbs_cz200_6', 'yl_hn_gbs_cz330_15',
              'yl_hn_gbs_hjl200_fh_6', 'yl_hn_gbs_hptz330_15', 'yl_hn_gbs_lz200_6', 'yl_hn_gbs_nht200_fh_6',
              'yl_hn_gbs_tpg200_fh_6', 'yl_hn_gbs_yt330_fh_15', 'yl_hn_qh_24', 'yl_hn_wtkk330_12', 'yl_hn_yw_25dz_24',
              'yl_hn_yw_25ptz_24', 'yl_hn_yw_25zgnxz_24', 'yl_hn_yw_cxz_2019sm_24', 'yl_hn_yw_ptjk_24',
              'yl_hn_yw_ptz_24', 'yl_hn_zm310_dz_24', 'yl_hn_zm310_gz_24', 'yl_hn_zm310_gz_hlxl_24',
              'yl_hn_zm400_jp_hs_15', 'yl_hn_zm400_jp_jbzhs_15', 'yl_hn_zm400_jp_jbzls_15', 'yl_hn_zm400_jp_ls_15',
              'yl_hn_zm_hP500_fake', 'yl_hn_zm_lP500_fake', 'yl_pesi_mtwt330G', 'yl_pesi_mtyw330G', 'yl_pesi_yw330G',
              'yl_pesi_yw330G_1', 'yl_xh_ycty_330G', 'yl_xh_ycty_500G']
bottle_labels = ['daiding_101']
cjfc_labels = ["ignore", "ty_cj_logo_ty", "cj_wsbg", "cj_hj", "cj_lsbx", "cj_tg", "cj_mit", "cj_clgj", "cj_gx",
               "cj_clj", "layer"]
condition_sku = ["cj_gx"]
gx_sku = ["gx1", "gx2", "gx3"]

sku_threshold = {"Default": 0.7,
                 "Specify": {'a_hn_401': 0.4, 'a_hn_301': 0.3, 'a_hn_302': 0.3, 'a_hn_303': 0.3, 'a_hn_304': 0.3,
                             'a_hn_305': 0.3, 'a_hn_306': 0.3, 'a_hn_307': 0.3, 'a_hn_308': 0.3, 'a_hn_309': 0.3}}
bottle_threshold = {"Default": 0.5}
cjfc_threshold = {"Default": 0.43}
gx_threshold = {"Default": 0.43}

process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='hn', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels),
             dict(type='Object_detect_trt', modelname='bottle', threshold=bottle_threshold, with_mask=False,
                  result_type='others', sku_list=bottle_labels),
             dict(type='Object_detect_trt', modelname='cjfc', threshold=cjfc_threshold, with_mask=True,
                  result_type='cjfc', sku_list=cjfc_labels)
         ]),
    dict(type='ConditionDetect',
         pipline=[
             dict(
                 condition=dict(type='Contain_sku', data_src='cjfc', contain_sku=condition_sku),
                 pipline=[dict(type='Object_detect_trt', modelname='tygx', threshold=gx_threshold, with_mask=False,
                               result_type='boxes', sku_list=gx_sku)])
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='Hn_ob_modellayer')
         ])
]

simple_process_pipeline = [
    dict(type='Preprocess',
         pipline=[
             dict(type='Resize_trt', width=1333, height=800)
         ]),
    dict(type='Detect',
         pipline=[
             dict(type='Object_detect_trt', modelname='hn', threshold=sku_threshold, with_mask=False,
                  result_type='sku', sku_list=sku_labels)
         ]),
    dict(type='Postprocess',
         pipline=[
             dict(type='General_ob_sku')
         ])
]
