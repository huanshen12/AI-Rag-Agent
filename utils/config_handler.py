import yaml

from utils.path_tool import get_abs_path

def load_rag_config(config_file:str = "config/rag.yml"):
    with open(get_abs_path(config_file), "r", encoding="utf-8") as file:
        config = yaml.load(file,Loader=yaml.FullLoader)
    return config

def load_chroma_config(config_file:str = "config/chroma.yml"):
    with open(get_abs_path(config_file), "r", encoding="utf-8") as file:
        config = yaml.load(file,Loader=yaml.FullLoader)
    return config

def load_prompts_config(config_file:str = "config/prompts.yml"):
    with open(get_abs_path(config_file), "r", encoding="utf-8") as file:
        config = yaml.load(file,Loader=yaml.FullLoader)
    return config

def load_agent_config(config_file:str = "config/agent.yml"):
    with open(get_abs_path(config_file), "r", encoding="utf-8") as file:
        config = yaml.load(file,Loader=yaml.FullLoader)
    return config

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

if __name__ == "__main__":
    print(rag_conf["chat_model"])

