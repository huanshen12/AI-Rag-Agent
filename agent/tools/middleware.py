from typing import Callable
from langchain.agents.middleware import ModelRequest, wrap_tool_call,before_model,dynamic_prompt
from langchain.agents import AgentState
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from langgraph.runtime import Runtime
from utils.prompt_loader import load_system_prompt,load_report_prompt

@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,         
    handler: Callable[[ToolCallRequest],ToolMessage | Command]
):     #工具执行的监控
    logger.info(f"调用工具: {request.tool_call['name']}")
    logger.info(f"工具参数: {request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"工具: {request.tool_call['name']} 调用成功")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"工具执行出错: {str(e)}")
        raise e
    

@before_model
def log_before_model(
    state:  AgentState,     #记录状态
    runtime: Runtime        #记录上下文信息
): #模型调用前的日志记录
    logger.info(f"[log_before_model] 即将调用模型，有{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model][type={type(state['messages'][-1]).__name__}] 消息内容: {state['messages'][-1].content.strip()}")
    return None

@dynamic_prompt
def report_prompt_switch(request: ModelRequest):   #动态切换提示词
    is_report = request.runtime.context.get("report",False)
    if is_report:
        logger.debug(f"[report_prompt_switch] 即将切换提示词")
    
        return load_report_prompt()
    return load_system_prompt()