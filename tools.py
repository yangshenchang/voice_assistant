"""内置轻量工具 —— 本地直连、零网络、即时应答。

这些是"明确、直接、简单"的函数: 时间查询、天气查询、米家 IoT 控制等。
数量不多, 每个都必须能在一轮 LLM 往返内给出结果(或明确告知"异步受理")。

约定
----
* 同步工具: 返回结果字符串 → 上层会带着 ``role:"tool"`` 结果再问模型一次,
  由模型组织成口语回答后播报。
* 异步工具(如 delegate): 返回 :data:`ASYNC_TOOL` 标记 → 本轮立即结束,
  结果以后台播报的形式送达(见 ``pipeline.VoiceAssistant.announce``)。
"""

from datetime import datetime

#: 工具返回值: 表示"已受理, 结果稍后异步送达", 上层据此结束本轮 LLM 循环
ASYNC_TOOL = object()

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


async def get_current_time() -> str:
    """当前本地日期时间(供模型组织成口语回答)。"""
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {_WEEKDAYS[now.weekday()]}"


GET_CURRENT_TIME_SPEC = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "查询当前日期和时间。用户问现在几点、今天几号、今天星期几时必须调用。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


DELEGATE_SPEC = {
    "type": "function",
    "function": {
        "name": "delegate_to_dsh",
        "description": (
            "把复杂任务交给后台 agent 异步处理: 报告生成、逻辑分析、时政查询、"
            "金融行情等时效性数据或多步推理任务。调用后不会立即有结果, 系统会在"
            "稍后播报。任务描述必须完整自包含(执行端看不到当前对话)。简单问答不要调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "完整自包含的任务描述"},
            },
            "required": ["task"],
        },
    },
}
