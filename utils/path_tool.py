import os

def get_project_root():
    current_path = os.path.abspath(__file__)
    current_dir  = os.path.dirname(current_path)
    project_dir  = os.path.dirname(current_dir)
    return  project_dir


def get_abs_path(relateive_file:str):
    file_abs_path = get_project_root()
    # print(file_abs_path)
    return os.path.join(file_abs_path,relateive_file)

# if __name__ == "__main__":
#     result = get_abs_path("config/config.txt")
#     print(result)