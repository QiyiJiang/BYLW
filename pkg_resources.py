"""
轻量级 pkg_resources 替代实现，仅满足 milvus_lite 对版本检测的需求。

避免在 uv 管理的精简环境中强依赖 setuptools。
"""


class DistributionNotFound(Exception):
    """占位异常，兼容 milvus_lite 的导入。"""


def get_distribution(name: str):
    """返回一个带有 version 属性的简单对象."""

    class _Dist:
        # 对当前用途而言，具体版本号无关紧要
        version = "0.0.0"

    return _Dist()

