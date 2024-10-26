import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from langchain.chains import RetrievalQA
from zhipuai import ZhipuAI
import re
from elasticsearch import Elasticsearch, exceptions
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import numpy as np

load_dotenv(find_dotenv()) 
client = OpenAI()

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


class MedicineInfoStandardizer:# 药物信息标准化器
    global field_list
    field_list=["药物名","类别", "鉴别", "贮藏", "指纹图谱", "功能主治", "规格", #文档中会出现的小标题
                             "含量测定",  "性味与归经", "浸岀物", 
                             "规定", "制法",  "检査", "用法与用量",
                             "用途",  "触藏", "正丁醇提取物", "特征图谱","禁忌", 
                              "效价测定", "正丁醇浸出物", 
                             "注意事项", "功能与主治", "制剂",
                             "性状","挥发油","处方", 
                             "适应症"]
    def __init__(self, llm):
        """
        初始化方法，存储语言模型实例。

        :param llm: 语言模型实例，用于处理文本的标准化。
        """
        self.field_list = field_list
        self.client = client
        self.llm = llm
    # 药物信息字段列表
    
    def bzh(self,input_data):# 从问题中提取字段列表对应的信息，标准化后输出。
        """
        从问题中提取字段列表对应的信息，如果没有对应信息则为空。

        :param question: 输入的问题。
        :param field_list: 字段列表。
        :return: 一个字典，键为字段名，值为从问题中提取出的对应信息或空字符串。
        """
        text='''问题一般是药品相关的问题，所以字段可能会有近义语句，
        比如制法即是制作方法,有重量的是处方的一部分，“用*制作而成”*也一般是处方，处方中通常只有药材名例如“板蓝根，罂粟壳”，无其他说明
        只有提到的字段才可以出现'''
        all_fields_str = ", ".join(field_list)
        extract_template = f"""你是个语意理解大师，你需要充分理解问题中的内容含义，他的问题提到了哪些信息，他的问题通常答案指向一种药物的名称，所以问题中提到的药物有克数的一般为处方中的内容。你需要把问题中的信息分类到字段中提到的内容中
        问题：{input_data}
        字段：{all_fields_str}
        
        用中文回复以及中文字符，回复时参考以下格式，比如 处方：板蓝根1500g,大青叶2250g。将涉及的字段与信息全部输出，顺序为字段名，提到的字段，同时，未提到的字段不需要输出字段名：信息
        额外信息：{text}
        未涉及的字段一定不要提到。一定不要出现“字段：None”的类似句子通常来说问题中只有2-3个字段内容，确保你不会输出超过3个字段，字段间换行输出
        """
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extract_template}]
        )
        
        answer = response.choices[0].message.content.strip() if response.choices else "无法提供答案"
        return answer
    
    def standardize_information(self, input_string):#将输入信息标准化
        """
        使用大模型处理输入信息并进行标准化。

        :param input_string: 输入的药物信息字符串。
        :return: 标准化后的字段信息。
        """
        
        extract_template = f"""从以下药物信息中总结字段，回答这句话需要使用到哪些字段，以及该药品的药品名，不用赘述其他：
        {input_string}
        字段列表：{', '.join(self.field_list)}
        
        请返回以下格式的结果：
        提到的药品名：药品名
        标准化输出：
        字段名
        字段名
        例如：
        提到的药品名：八角茴香
        标准化输出：
        功能主治
        性状
        """
        
        # 调用大模型生成标准化输出
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extract_template}]
        )
        standardized_output = response.choices[0].message.content.strip() if response.choices else "无输出"
        # 提取返回内容
        final_output = standardized_output
        return final_output

def classify_pharmacy_query(input_data):# 检查用户输入的问题是否与药品或药学相关。
        """
        判断用户输入的问题是否与药品或药学相关。

        :param input_data: 用户输入的问题字符串。
        :return: "good"（药学相关问题）或 "bad"（非药学相关问题）。
        """
        classify_template = f"""你是药学专家，请判断以下问题是否与药品或药学相关：
        {input_data}

        如果这是一个与药品或药学相关的问题（如提问药品的功能、用途、副作用等），返回 "good"；
        如果不是药学相关的问题，返回 "bad"。
        只能返回“good”或者“bad”
        """
        
        response =  client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classify_template}]
        )
        
        query_type = response.choices[0].message.content.strip() if response.choices else "未知类型"
        print(query_type)
        
        # 确保只返回 "good" 或 "bad"
        if query_type not in ["good", "bad"]:
            return "未知类型"
        
        return query_type

def extract_subsections(content):# 提取小标题和内容

    pattern = re.compile(r'(?:【|t)(.+?)(?:】)')
    matches = pattern.finditer(content)
    
    subsections = {}
    last_position = 0
    last_title = None
    
    for match in matches:
        title = match.group().strip('【】t]')
        if last_title:  
            subsections[last_title] = content[last_position:match.start()].strip()
        
        last_title = title
        last_position = match.end()

    if last_title:
        subsections[last_title] = content[last_position:].strip()

    return subsections

def retrieve_data_from_es(index_name):# 从Elasticsearch中检索数据
    res = es.search(index=index_name, body={"query": {"match_all": {}}, "size": 10000})
    return res['hits']['hits']

def load_faiss_index(embedding_file_path):# 加载FAISS索引

    global index, ids, id_to_content
    data = np.load(embedding_file_path)
    embeddings = data['embeddings']
    ids = data['ids']
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

def process_and_vectorize(index_name,embedding_file_path):# 处理并向量化Elasticsearch中的数据(存储过程中的）

    """
    处理并向量化Elasticsearch中的数据。如果本地已经存在FAISS索引，则直接加载。
    """
    global index, ids, id_to_content,db_path
    db_path = embedding_file_path
    # 如果已存在嵌入文件，加载它
    if os.path.exists(embedding_file_path):
        print("Loading existing FAISS index from disk...")
        load_faiss_index(embedding_file_path) 
        return
    
    # 否则，处理并向量化新数据
    print("No existing FAISS index found. Processing and vectorizing data...")
    entries = retrieve_data_from_es(index_name)
    subsections_list = []
    ids = []
    id_to_content = {}
    texts = []  # 存储小标题和对应文本的列表

    # 遍历所有文档条目
    for entry in entries:
        doc_id = entry['_id']  # 文档ID
        content = entry['_source']['content']  # 文档内容
        subsections = extract_subsections(content)  # 提取文档的小节
        
        # 将每个文档ID和全文存储到id_to_content中
        id_to_content[doc_id] = content
        print(f"Document ID: {doc_id} - Content Length: {len(content)}")  # 打印文档ID和内容长度

        # 为每个小节存储对应的ID和文本
        for title, text in subsections.items():
            subsections_list.append((doc_id, title, text))  # 文档ID, 标题, 内容
            ids.append(doc_id)  # 存储文档ID
            texts.append((title, text))  # 存储小标题和对应文本

    # 向量化小节内容
    embeddings = model.encode([text for _, _, text in subsections_list], convert_to_numpy=True)

    # 创建FAISS索引
    dimension = embeddings.shape[1]  # 获取嵌入维度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))  # 将向量添加到FAISS索引中

    # 将嵌入和ID、小标题保存到磁盘
    np.savez_compressed(embedding_file_path, embeddings=embeddings, ids=ids, texts=texts)
    print("Data processed and saved to disk.")

def retrieve_vector_and_text(input_data, embedding_file_path, top_k=1):#将接收的数据向量化并在本地数据库中进行向量检索，同时返回检索到的文本。
    """
    将输入文字向量化并在本地数据库中进行向量检索，同时返回检索到的文本。

    :param input_data: 用户输入的文本。
    :param embedding_file_path: 嵌入文件的路径。
    :param top_k: 检索的向量数量。
    :return: 检索到的文档ID、小标题及其对应的文本。
    """
    # 检查嵌入文件是否存在
    print(f"Embedding file path: {embedding_file_path}")

    if not os.path.exists(embedding_file_path):
        raise FileNotFoundError(f"Embedding file not found at: {embedding_file_path}")
    query_embedding = model.encode([input_data], convert_to_numpy=True)
    #加载 FAISS 索引和向量
    data = np.load(embedding_file_path, allow_pickle=True)
    embeddings = data['embeddings']
    ids = data['ids']
    texts = data['texts']  # 这里加载文本信息
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    # 检索最相关的向量
    D,I = index.search(query_embedding.astype(np.float32), top_k)
    # 获取检索到的向量和对应的文档ID及文本
    retrieved_ids = ids[I[0]].tolist()
    data = np.load(embedding_file_path, allow_pickle=True)
    texts = data['texts']  # 这里加载文本信息
    #print("这里是retrieved_ids的打印")
    retrieved_texts = [texts[i] for i in I[0]]  # 获取对应的文本内容
    #print("这里是retrieved_texts的打印")

    # 组合结果
    results = [(retrieved_ids[i], retrieved_texts[i][0], retrieved_texts[i][1]) for i in range(top_k)]
    #print("这里是results的打印")
    
    return results

    
    # Combine retrieved content and generated answer for display

    return retrieved_content, generated_answer

def connect_elasticsearch():# 连接Elasticsearch
    hosts = [
        {'host': '192.168.110.28', 'port': 9200, 'scheme': 'https'},
        {'host': '192.168.1.230', 'port': 9200, 'scheme': 'https'},
        {'host': '127.0.0.1', 'port': 9200, 'scheme': 'https'}
    ]
    es = None
    for host in hosts:
        try:
            es = Elasticsearch(
                [host],
                basic_auth=('elastic', '7ztvwEMjr0H+_R4Vec*R'),
                verify_certs=False  # 在开发时禁用 SSL 验证，生产环境中请谨慎使用
            )
            if es.ping():
                print(f'成功连接到 Elasticsearch: {host["host"]}')
                return es
            else:
                print(f'无法连接到 Elasticsearch: {host["host"]}')
        except exceptions.ConnectionError as e:
            print(f"连接错误：{e} - 尝试下一个主机")

    print('所有主机连接失败')
    return None
es = connect_elasticsearch()# 初始化Elasticsearch
def extract_drug_info(text):# 将标准化后的信息切分

    # 匹配多个药品名和标准化输出
    drug_pattern = re.compile(r'提到的药品名：(.+?)\s+标准化输出：\s*(.+?)(?=(提到的药品名：|$))', re.DOTALL)
    matches = drug_pattern.findall(text)
    
    drugs = []
    standard_outputs = []
    
    for match in matches:
        # 提取药品名，可能包含多个药品，以逗号或其他标点分隔
        drug_names = [name.strip() for name in re.split(r'[、,，]', match[0]) if name.strip()]
        # 提取并分割标准化输出的每一行
        outputs = [line.strip() for line in match[1].strip().split('\n') if line.strip()]
        
        for drug_name in drug_names:
            drugs.append(drug_name)
            standard_outputs.append(outputs)

    return drugs, standard_outputs

def verify_data_in_elasticsearch(es, index_name, doc_id, sub_titles):# 验证数据是否在Elasticsearch中，并且返回结果

    output = []  # 存储输出结果

    try:
        #    检查索引是否存在
        response = es.get(index=index_name, id=doc_id)
        content = response['_source']['content']
        
        #    提取所有小标题及其内容
        subsections = extract_subsections(content)
        
        found_content = []
        
        #    检查每个小标题是否存在于提取的内容中
        for sub_title in sub_titles:
            if sub_title in subsections and subsections[sub_title]:
                found_content.append(f"小标题 '{sub_title}' 的内容:\n{subsections[sub_title]}")
        
        #    输出结果
        if found_content:
            output.append("\n".join(found_content))
        else:
            #    没有找到任何小标题，输出完整内容
            output.append(f"未找到任何小标题，输出完整内容:\n{content}")
        
    except exceptions.NotFoundError:
        output.append(f"文档 ID '{doc_id}' 或索引 '{index_name}' 不存在。")
    except exceptions.TransportError as e:
        output.append(f"查询错误：{e}")

    return "\n".join(output)  # 以换行符连接输出结果

