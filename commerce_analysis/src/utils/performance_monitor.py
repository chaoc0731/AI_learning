#!/usr/bin/env python3
"""
性能监控工具
"""

import time
import logging
import psutil
import os
from functools import wraps

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器"""

    @staticmethod
    def timeit(func):
        """执行时间装饰器"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

            logger.info(f"开始执行: {func.__name__}")
            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            elapsed_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.info(f"完成执行: {func.__name__}, 耗时: {elapsed_time:.2f}秒, 内存使用: {memory_used:.2f}MB")
            return result

        return wrapper

    @staticmethod
    def check_memory_usage():
        """检查内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }