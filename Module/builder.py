from Core.build.registry import Registry, build_from_cfg

PREPROCESSES = Registry('preprocess')
POSTPROCESSES = Registry('postprocess')
DETECTS = Registry('detect')
CONDITION = Registry('condition')
# BBOX_DETECTS = Registry('bbox_detect')
# MASK_DETECTS = Registry('mask_detect')
# CLASSIFIERS = Registry('classifier')
STITCHINGS = Registry('stitching')


def build(cfg, registry):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry) for cfg_ in cfg
        ]
        return modules
    else:
        return build_from_cfg(cfg, registry)


def build_preprocess(cfg):
    return build(cfg, PREPROCESSES)


def build_postprocess(cfg):
    return build(cfg, POSTPROCESSES)


def build_detect(cfg):
    return build(cfg, DETECTS)


# def build_bbox_detect(cfg):
#     return build(cfg, BBOX_DETECTS)
#
#
# def build_mask_detect(cfg):
#     return build(cfg, MASK_DETECTS)
#
#
# def build_classifier(cfg):
#     return build(cfg, CLASSIFIERS)


def build_condition(cfg):
    return build(cfg, CONDITION)


def build_stitching(cfg):
    return build(cfg, STITCHINGS)
