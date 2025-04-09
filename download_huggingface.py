"""
코드는 실행은 아래 링크에서 구체적인 내용 확인
https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/

필요한 라이브러리 설치
pip install llama-index llama-index-llms-huggingface llama-index-llms-huggingface-api
pip install transformers
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings


user_input="How do drones identify vehicles?"

#similarity_top_k
k=3
#temperature
temp=0.1
#num_output
mt=1024

model_name = "EleutherAI/gpt-neo-1.3B"  # 사용할 공개 LLM 모델 이름
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    model=model,
    tokenizer=tokenizer,
    device_map="cpu",  # GPU가 있다면 auto로 설정, CPU만 사용 시 "cpu"
    model_kwargs={"temperature": temp, "max_new_tokens": mt}, # 쿼리 엔진의 파라미터와 유사하게 설정
)
