"""从 YAML 加载配置，并将 ${VAR} 占位符替换为 .env 中的环境变量。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

# 在首次加载配置前加载 .env（若已安装 python-dotenv）
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

# 匹配 ${VAR_NAME} 或 ${VAR_NAME:-default}
_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _try_number(s: str) -> Any:
    """若字符串为数字则转为 int 或 float，否则返回原字符串。"""
    if not s:
        return s
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _substitute_env(value: Any) -> Any:
    """递归地将字典/列表/字符串中的 ${VAR} 或 ${VAR:-default} 替换为环境变量。"""
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)
        result = _ENV_PATTERN.sub(replacer, value)
        # 若整段是单个占位符替换结果且为数字，转为 int/float
        if re.fullmatch(r"[\d.]+", result):
            return _try_number(result)
        return result
    return value


def load_yaml_with_env(
    path: Path,
    *,
    load_dotenv_file: bool = True,
    dotenv_path: Path | None = None,
) -> Dict[str, Any]:
    """
    加载 YAML 文件，并将其中所有 ${VAR} 或 ${VAR:-default} 替换为环境变量。

    - 若 load_dotenv_file 为 True 且已安装 python-dotenv，会先加载 .env。
    - dotenv_path 指定 .env 路径；默认在 path 所在目录及上级目录查找 .env。
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if load_dotenv_file and _HAS_DOTENV:
        if dotenv_path is not None:
            load_dotenv(dotenv_path)
        else:
            # 优先从项目根目录（含 .env 的目录）加载
            for parent in [path.parent] + list(path.parents):
                env_file = parent / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break
            else:
                load_dotenv()  # 默认当前工作目录

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    return _substitute_env(raw)
