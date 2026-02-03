from langchain_core.tools import tool
from rag.rag_service import rag_service
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger



rag = rag_service()
user_ids = ["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"]

month_arr = ["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09","2025-10","2025-11","2025-12"]
external_data = {}


@tool
def rag_summarize(query:str):
    """
    【核心工具】当用户询问关于扫地机器人的安装、设置、故障代码、参数对比或说明书内容时，必须调用此工具。它包含最新的官方技术文档，比你自带的知识更准确。
    """
    return rag.rag_sumarize(query)

@tool
def get_weather(city:str) -> str:
    """
    用于获取天气信息，以消息字符串的形式返回
    """
    return f"城市{city}的天气为晴天，温度为26摄氏度，空气湿度为60%，南风一级，AQI21，最近六小时降雨概率极低"

@tool
def get_user_location():
    """
    用于获取用户当前城市位置，以消息字符串的形式返回
    """
    return f"用户当前城市位置为{random.choice(['北京','上海','广州','深圳'])}"

@tool
def get_user_id() -> str:
    """
    用于获取用户ID，以消息字符串的形式返回
    """
    return f"用户ID为{random.choice(user_ids)}"

@tool
def get_current_month() -> str:
    """
    用于获取当前月份，以消息字符串的形式返回
    """
    return f"当前月份为{random.choice(month_arr)}"

def generate_external_data():
    """
    {
        "user_id":{
            "month": {"特征":""},
            "month": {"特征":""},
        }
    }
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not external_data_path:
            raise KeyError(f"{external_data_path}不存在")

        with open(external_data_path,"r",encoding="utf-8") as f:
            lines = f.readlines()[1:]
            for line in lines:
                arr = line.strip().split(",")

                user_id = arr[0].replace('"',"")
                feature = arr[1].replace('"',"")
                xiaolv = arr[2].replace('"',"")
                cost = arr[3].replace('"',"")
                comparison = arr[4].replace('"',"")
                month = arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][month] = {"特征":feature,"效率":xiaolv,"耗材":cost,"对比":comparison}
                


@tool
def fetch_external_data(user_id:str,month:str) -> str:
    """
    用于获取用户在指定月份的外部数据，以纯字符串的形式返回，若未检索到则返回空字符串
    """
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"用户{user_id}在{month}未检索到外部数据")
        return ""

@tool
def fill_context_for_report():
    """
    无参数，无返回值，调用后中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息
    """
    return "fill_context_for_report已调用"