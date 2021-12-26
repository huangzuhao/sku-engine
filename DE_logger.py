import logging
import os
from logging import handlers

#
# class MyLogging(logging.Logger):
#     def __init__(self,name,level = logging.INFO,file = None):
#         """ :param name: 日志名字 :param level: 级别 :param file: 日志文件名称 """
#         # 继承logging模块中的Logger类，因为里面实现了各种各样的方法，很全面，但是初始化很简单
#         # 所以我们需要继承后把初始化再优化下，变成自己想要的。
#         super().__init__(name,level)
#
#         #设置日志格式
#         fmt = "%(asctime)s %(name)s %(levelname)s %(filename)s--%(lineno)dline :%(message)s"
#         formatter = logging.Formatter(fmt)
#
#         # 文件输出渠道
#         if file:
#             handle2 = logging.FileHandler(file,encoding="utf-8")
#             handle2.setFormatter(formatter)
#             self.addHandler(handle2)
#         # 控制台渠道
#         else:
#             handle1 = logging.StreamHandler()
#             handle1.setFormatter(formatter)
#             self.addHandler(handle1)
# # 因为一个项目的日志都是写入到一个日志文件的，所以可以把name，file这两个参数写死，直接实例化
# # 后期每个模块调用就不用实例化，导入可以直接使用
# logger = MyLogging("mylog",file="my_log.log")


import sys

DE_LOGGING_CONFIG_DEFAULTS = dict(
    version=1,
    disable_existing_loggers=False,
    loggers={
        "DE.root": {"level": "INFO", "handlers": ["file"]},
    },
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "generic",
            "filename": "Logs/log.txt",
            "when": "D",
            "backupCount": 30
        },

    },
    formatters={
        "generic": {
            # "format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
            "format": "%(asctime)s [%(process)d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            "datefmt": "[%Y-%m-%d %H:%M:%S %z]",
            "class": "logging.Formatter",
        },
    },
)

logger = logging.getLogger("DE.root")

# class Logger(object):
#     level_relations = {
#         'debug': logging.DEBUG,
#         'info': logging.INFO,
#         'warning': logging.WARNING,
#         'error': logging.ERROR,
#         'crit': logging.CRITICAL
#     }  # 日志级别关系映射
#
#     def __init__(self, filename='log', level='info', when='D', backCount=30,
#                  fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
#         isExists = os.path.exists('logs')
#         if not isExists:
#             os.makedirs('logs')
#         filename = os.path.join('logs', filename)
#         self.logger = logging.getLogger(filename)
#         format_str = logging.Formatter(fmt)  # 设置日志格式
#         self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
#         sh = logging.StreamHandler()  # 往屏幕上输出
#         sh.setFormatter(format_str)  # 设置屏幕上显示的格式
#         th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
#                                                encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
#         # 实例化TimedRotatingFileHandler
#         # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
#         # S 秒
#         # M 分
#         # H 小时、
#         # D 天、
#         # W 每星期（interval==0时代表星期一）
#         # midnight 每天凌晨
#         th.setFormatter(format_str)  # 设置文件里写入的格式
#         self.logger.addHandler(sh)  # 把对象加到logger里
#         self.logger.addHandler(th)
