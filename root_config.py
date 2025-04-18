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

    # ------------------------edit-----------------------
    CUDA_VISIBLE_DEVICES = "1,2"
    HTTP_PROXY = ""
    HTTPS_PROXY = ""
    OPENAI_API_KEY = ""
    DEEPSEEK_API_KEY = ""
    SERPER_API_KEY = ""
    
    
    # -----------------------------------------------
    LLMAM2_7B_CHAT_HF_MODEL_PATH = "/netcache/huggingface/Llama-2-7b-chat-hf"
    if not os.path.exists(LLMAM2_7B_CHAT_HF_MODEL_PATH):
        LLMAM2_7B_CHAT_HF_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
        
    LLMAM2_13B_CHAT_HF_MODEL_PATH = "/netcache/huggingface/Llama-2-13b-chat-hf"
    if not os.path.exists(LLMAM2_13B_CHAT_HF_MODEL_PATH):
        LLMAM2_13B_CHAT_HF_MODEL_PATH = "meta-llama/Llama-2-13b-chat-hf"
        
    BAICHUAN2_7B_CHAT_MODEL_PATH = "/netcache/huggingface/Baichuan2-7B-Chat"
    if not os.path.exists(BAICHUAN2_7B_CHAT_MODEL_PATH):
        BAICHUAN2_7B_CHAT_MODEL_PATH = "baichuan-inc/Baichuan2-7B-Chat"
        
    BAICHUAN2_13B_CHAT_MODEL_PATH = "/netcache/huggingface/Baichuan2-13B-Chat"
    if not os.path.exists(BAICHUAN2_13B_CHAT_MODEL_PATH):
        BAICHUAN2_13B_CHAT_MODEL_PATH = "baichuan-inc/Baichuan2-13B-Chat"
    
