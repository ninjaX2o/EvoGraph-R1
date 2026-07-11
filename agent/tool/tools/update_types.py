"""
统一的更新类型枚举定义
避免多个文件中重复定义UpdateType导致类型不匹配
"""

from enum import Enum

class UpdateType(Enum):
    """更新类型枚举"""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
