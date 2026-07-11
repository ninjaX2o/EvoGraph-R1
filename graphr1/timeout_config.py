
# 增强的重试配置
MAX_RETRIES = 6
RETRY_DELAYS = [5, 10, 20, 30, 60, 120]  # 递增延迟
TIMEOUT_CONFIG = {
    'default': 120,
    'entity_extraction': 180,
    'hyperedge_extraction': 180,
    'llm_call': 120
}
