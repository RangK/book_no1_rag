from datasets import load_dataset
import pandas as pd
import os 
import time
import chromadb
import spacy
import numpy as np
from rangk.chapter_08.llm_tools import *

huggingface_access_key = ""
os.environ["HF_TOKEN"] = huggingface_access_key

pipeline = create_pipeline_with_huggingface(huggingface_access_key)
dataset = load_dataset()
## filters only data where the support and correct_answer key values are not null
filtered_dataset = dataset.filter(lambda x: x["support"] is not None and x["correct_answer"] is not None)
df = create_dataframe(filtered_dataset)

### 8.5 크로마 컬렉션에 데이터 임베딩 및 업서트
## chromadb에서 사용하는 embedding model의 기본 값은 아래와 같다.
## embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
client = chromadb.PersistentClient(path="/Users/rangkim/.cache/chroma")
collection_name = "sciq_supports6"
collections = client.list_collections()

# collection을 가져와서, 이름이 collection_name과 같은 이름을 가진 collection이 있는지 확인한다.
# any = True가 하나라도 반환되면 True
collection_exists = any(collection.name == collection_name for collection in collections)

# use fstring
print(f"Collection exists: {collection_exists}")
collection = client.get_or_create_collection(name=collection_name)
results = collection.get()

print("#### Collection Data ####")
for result in results:
    print(result)
print("#################")
    

# 8.5.1 모델 선택
ldf = len(df)
nb = ldf 

"""
print(df["completion"][:nb])
Desc : 데이터 프레임의 데이터를 리스트 형식으로 printout하면, index번호가 따라 붙는다.
    0        mesophilic organisms because Mesophiles grow b...
    1        coriolis effect because Without Coriolis Effec...
    2        exothermic because Summary Changes of state ar...
    3        alpha decay because All radioactive decay is d...
    4        smoke and ash because Example 3.5 Calculating ...
                                ...
    11674    peptides because Protein A large part of prote...
    11675    rate of decay because The rate of decay of a r...
    11676    biomes because Terrestrial ecosystems, also kn...
    11677    supersonic because The modern day formulation ...
    11678    organ because An organ is a structure composed...
    
    Name: completion, Length: 11679, dtype: object
    nb = 11679
"""

# iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# 책에서 사용한 코드는 astype(str).tolist()구분을 사용하지만, 불필요 (Series.tolist()은 이미 str)
completion_list = df["completion"].iloc[:nb].tolist()

# 아래 코드가 실행되면, collection에서 사용하는 embedding 모델을 사용하여 completion_list를 임베딩한다.
# 만약, 모델을 처음 호출하면 download가 이루어진다.
# /Users/rangkim/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 79.3M/79.3M [00:12<00:00, 6.81MiB/s]

# chromadb has max batch size == 5461
# add new documents which id not exists to collection 

# 데이터 보관 소요 시간 : 848초 mac pro m1 기준

s_time = time.time()
max_batch_size = 1000
for i in range(0, nb, max_batch_size):
    documents = completion_list[i:i+max_batch_size]
    ids = [str(i) for i in range(i, i+len(documents))]
    already_exists = collection.get(ids)
    had_ids = set(already_exists['ids'])
    
    new_ids = [id for id in ids if id not in had_ids]
    new_documents = [doc for id, doc in zip(ids, documents) if id not in had_ids]
    
    if new_ids:
        print(f"Adding {len(new_ids)} new documents")
        collection.add(
            ids=new_ids,
            documents=new_documents,
            metadatas=[{"type": "completion"} for _ in range(i, i+len(documents))],
        )
    
    e_time = time.time()
    print(f"Batch {i//5461+1} completed in {e_time - s_time:.4f} seconds")
        
    
## 8.5.3 임베딩 표시
# 정상적으로 embedding 되어 저장된 데이터만 가져온다.
result = collection.get(include=['embeddings'])

# 23:08:59  

query_number = nb
query_texts = df["question"][:query_number].astype(str).tolist()
results = collection.query(query_texts=query_texts, n_results=1)
print("#### Query Results ####")
for result in results:
    print(result)


# 이 코드를 실행 하기 전에 en_core_web_sm를 설치해주세요.
# !python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")  

def simple_text_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    if np.linalg.norm(doc1.vector) == 0 or np.linalg.norm(doc2.vector) == 0:
        return 0.0
    
    # cosine similarity
    return (np.dot(doc1.vector, doc2.vector) / (np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector)))


nbpd = 100
acc_counter = 0
display_counter = 0

for i, q in enumerate(df['question'][:nb]):
    original_completion = df['completion'][i]
    retrieved_document = results['documents'][i][0]
    similarity_score = simple_text_similarity(original_completion, retrieved_document)
    if similarity_score > 0.7:
        acc_counter += 1
    display_counter += 1
    if display_counter <= nbpd or display_counter >= nb - nbpd:
        print(i, " ", f"Question: {q}")
        print(f"Retrieved document: {retrieved_document}")
        print(f"Original completion: {original_completion}")
        print(f"Similarity Score: {similarity_score:.2f}")
        print()  # Blank line for better readability between entries

if nb > 0:
    acc = acc_counter / nb
    print(f"Number of documents: {nb:.2f}")
    print(f"Overall similarity score: {acc:.2f}")
