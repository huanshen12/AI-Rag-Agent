from agent.tools.agent_tools import (rag_summarize,get_current_month,get_user_id,
get_user_location,get_weather,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from utils.path_tool import get_abs_path
from chat_history.file_chat_history import get_history
import os
from model.factory import chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from utils.logger_handler import logger


class react_agent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            tools=[rag_summarize,get_current_month,get_user_id,
get_user_location,get_weather,fetch_external_data,fill_context_for_report],
            system_prompt = None,
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )

    def excute_stream(self,query:str,session_id:str):

        chat_history = get_history(session_id)
        input_messages = chat_history.messages + [HumanMessage(content=query)]

        input_dict ={
            "messages":input_messages
        }
        full_response = ""
        latest_message = ""
        message_tosave = None
        try:
            for chunk in self.agent.stream(input_dict,stream_mode="values",context = {"report":False}):
                latest_message = chunk["messages"][-1]
                if latest_message.type == "ai" and latest_message.content:
                    full_response = latest_message.content
                    message_tosave = latest_message
                    yield latest_message.content.strip() + "\n"    
        except Exception as e:
            yield f"发生错误，请稍后重试{str(e)}"

        if full_response and message_tosave:
            logger.info("正在保存对话...")
            chat_history.add_messages([HumanMessage(content=query),message_tosave])
