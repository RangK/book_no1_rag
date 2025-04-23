import transformers
import torch    
import os
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import datasets as hunggingfaceDatasets
import pandas as pd


def time_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
        
    return wrapper

@time_func
def create_pipeline_with_huggingface(access_key):
    # Use HF_TOKEN to load model
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_key)

    # LLM 에 Query를 날리고 Text를 만들어주는 Pipeline 구성 
    # Query -> Tokenization -> Transformer Inference -> Text
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return pipeline

@time_func
def load_dataset():
    dataset = hunggingfaceDatasets.load_dataset("sciq", split="train")

    return dataset


@time_func
def create_dataframe(dataset):
    colmns_to_drop = ["distractor3", "distractor2", "distractor1"]

    df = pd.DataFrame(dataset)
    df.drop(columns=colmns_to_drop, inplace=True)

    # 정답 + 지문 (정답에 대한 설명)
    df["completion"] = df["correct_answer"] + " because " + df["support"]

    df.dropna(subset=["completion"], inplace=True)

    return df