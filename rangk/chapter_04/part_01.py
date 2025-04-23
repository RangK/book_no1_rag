# 4장. 드론 기술을 위한 다중 모달 모듈형 RAG
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core import VectorStoreIndex
import os
import openai
import deeplake
import json
import pandas as pd
import numpy as np
import time
import textwrap


# openAI 

# os.makedirs('./etc', exist_ok=True)
# with open('./etc/resolv.conf', 'w') as file:
#    file.write("nameserver 8.8.8.8")
   

dataset_path_llm = "hub://rangkim/drone_v2"
ds_llm = deeplake.load(dataset_path_llm)


data_llm = {}

# Iterate through the tensors in the dataset
for tensor_name in ds_llm.tensors:
    tensor_data = ds_llm[tensor_name].numpy()

    # Check if the tensor is multi-dimensional
    if tensor_data.ndim > 1:
        # Flatten multi-dimensional tensors
        data_llm[tensor_name] = [np.array(e).flatten().tolist() for e in tensor_data]
    else:
        # Convert 1D tensors directly to lists and decode text
        if tensor_name == "text":
            data_llm[tensor_name] = [t.tobytes().decode('utf-8') if t else "" for t in tensor_data]
        else:
            data_llm[tensor_name] = tensor_data.tolist()

# Create a Pandas DataFrame from the dictionary
df_llm = pd.DataFrame(data_llm)

# Ensure 'text' column is of type string
df_llm['text'] = df_llm['text'].astype(str)
# Create documents with IDs
documents_llm = [Document(text=row['text'], doc_id=str(row['id'])) for _, row in df_llm.iterrows()]
vector_store_index_llm = VectorStoreIndex.from_documents(documents_llm)
vector_query_engine_llm = vector_store_index_llm.as_query_engine(similarity_top_k=2, temperature=0.1, num_output=1024)

user_input="How do drones identify a truck?"

start_time = time.time()
llm_response = vector_query_engine_llm.query(user_input)
# Stop the timer
end_time = time.time()
# Calculate and print the execution time
elapsed_time = end_time - start_time
print(f"Query execution time: {elapsed_time:.4f} seconds")
print(textwrap.fill(str(llm_response), 100))