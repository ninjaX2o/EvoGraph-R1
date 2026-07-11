# 延迟导入GraphR1，避免循环导入问题
def _get_graphr1_class():
    """延迟获取GraphR1类"""
    try:
        from .graphr1 import GraphR1
        return GraphR1
    except ImportError as e:
        print(f"❌ Failed to import GraphR1: {e}")
        return None

def _get_query_param_class():
    """延迟获取QueryParam类"""
    try:
        from .base import QueryParam
        return QueryParam
    except ImportError as e:
        print(f"❌ Failed to import QueryParam: {e}")
        return None

# 创建延迟导入的包装器
class _GraphR1Wrapper:
    def __new__(cls, *args, **kwargs):
        GraphR1Class = _get_graphr1_class()
        if GraphR1Class is None:
            raise ImportError("GraphR1 class not available")
        return GraphR1Class(*args, **kwargs)

class _QueryParamWrapper:
    def __new__(cls, *args, **kwargs):
        QueryParamClass = _get_query_param_class()
        if QueryParamClass is None:
            raise ImportError("QueryParam class not available")
        return QueryParamClass(*args, **kwargs)

# 导出包装器
GraphR1 = _GraphR1Wrapper
QueryParam = _QueryParamWrapper
