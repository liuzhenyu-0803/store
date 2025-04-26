import os
from llama_cpp import Llama

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "gemma-3-4B-it-QAT-Q4_0.gguf")

local_model = Llama(
                model_path=model_path, 
                n_gpu_layers=-1,          # -1表示所有层都使用GPU（需要Vulkan支持）
                n_ctx=4096,               # 缩小上下文窗口以提高性能
                verbose=True
            )

response = local_model.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "hi"
		}
	]
)

print(response)