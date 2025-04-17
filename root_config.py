import os


class RootConfig:
    # global variables
    tempModelCatch = []
    
    # Absolute path of the project.
    root_path = os.path.dirname(
        os.path.abspath(__file__))
    print("--------------root_path-------------\n",root_path)
    if not root_path.endswith("/"):
        root_path = root_path + "/"

    CUDA_VISIBLE_DEVICES = "1,2"
    HTTP_PROXY = ""
    HTTPS_PROXY = ""
    OPENAI_API_KEY = ""
    SERPER_API_KEY = ""
