from agent.tools.agent_tools import (rag_summarize,get_current_month,get_user_id,
get_user_location,get_weather,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

from model.factory import chat_model
from langchain.agents import create_agent


class react_agent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            tools=[rag_summarize,get_current_month,get_user_id,
get_user_location,get_weather,fetch_external_data,fill_context_for_report],
            system_prompt = None,
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )

    def excute_stream(self,query:str):
        input_dict ={
            "messages":[
                {"role":"user", "content":query}
            ]
        }
        for chunk in self.agent.stream(input_dict,stream_mode="values",context = {"report":False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"    

