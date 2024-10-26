import time
import os
import docx
import re
from elasticsearch import Elasticsearch, exceptions
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, exceptions
from openai import OpenAI
from embed import MedicineInfoStandardizer,classify_pharmacy_query, connect_elasticsearch, extract_drug_info, process_and_vectorize,verify_data_in_elasticsearch, retrieve_vector_and_text
import os

client = OpenAI()#· 调用openai的api

history = []  # 问答记忆列表

config = {}# 配置信息全局变量

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')# 模型选择

es = connect_elasticsearch()# 连接es

# web中配置tab页的更新函数
def update_config(es_host, es_port, es_user, es_pass, es_index, vector_db):# 配置页面的更新函数, 用于更新配置信息应用到全局
    """
    更新配置信息

    参数:
    es_host (str): Elasticsearch 主机地址
    es_port (str): Elasticsearch 端口号
    es_user (str): Elasticsearch 用户名
    es_pass (str): Elasticsearch 密码
    es_index (str): Elasticsearch 索引名称
    vector_db (str): 向量数据库路径

    返回:
    str: 配置更新成功的消息
    """
    global config, vector_db_path,es_indx
    config = {
        'es_host': es_host,
        'es_port': es_port,
        'es_user': es_user,
        'es_pass': es_pass,
        'es_index': es_index
    }
    es_indx=es_index
    vector_db_path = vector_db  # 更新全局变量
    return "配置已更新!"

class UploadDoc:# 上传文档类
    
    def __init__(self, file_input):#初始化类的实例
        """
        初始化类的实例。

        参数:
        file_input (str): 上传文件的路径。

        属性:
        file_input (str): 上传文件的路径。
        es_host (str): Elasticsearch 主机地址。
        es_port (str): Elasticsearch 端口号。
        es_user (str): Elasticsearch 用户名。
        es_pass (str): Elasticsearch 密码。

        异常:
        ValueError: 如果配置未设置，则抛出异常。
        """
        self.file_input = file_input  # Path to the uploaded file
        try:
            self.es_host = config['es_host']
            self.es_port = config['es_port']
            self.es_user = config['es_user']
            self.es_pass = config['es_pass']
            self.es_index = config['es_index']
        except KeyError as e:
            raise ValueError(f"配置未设置: {e}")


    def clean_filename(self,filename):#从文件名中提取中文字符并去除末尾的空格
        """
        从文件名中提取中文字符并去除末尾的空格。

        参数:
        filename (str): 文件名。

        返回:
        str: 提取后的中文字符串。
        """
        return ''.join(re.findall(r'[\u4e00-\u9fff]+', filename)).rstrip()

    def extract_titles_and_content(self, doc_obj):#从 Word 文档对象中提取标题和内容，并将其存储在一个字典中。
        
        content_dict = {}
        temp_doc = []

        for paragraph in doc_obj.paragraphs:
            if not paragraph.runs:
                continue
                
            font_size = paragraph.runs[0].font.size
            
            if font_size is not None:
                print(f"Font size: {font_size.pt}, Type: {type(font_size.pt)}")
                
                # 确保 font_size.pt 是数字
                if isinstance(font_size.pt, (int, float)):
                    if font_size.pt == 12:
                        if temp_doc:
                            title = self.clean_filename(temp_doc[0])
                            if title:
                                content_dict[title] = temp_doc
                            temp_doc = []
                else:
                    print(f"Unexpected font size type: {type(font_size.pt)}")
            
            # 添加调试信息，检查段落文本
            print(f"Adding paragraph text: {paragraph.text}")

            # 将段落文本添加到临时文档
            temp_doc.append(paragraph.text)

        # 处理最后一个段落
        if temp_doc:
            title = self.clean_filename(temp_doc[0])
            if title:
                content_dict[title] = temp_doc

        return content_dict

    def connect_elasticsearch(self):#连接到es
        print(f"Connecting to Elasticsearch at {self.es_host}:{self.es_port} with user {self.es_user}")
        
        es = Elasticsearch(
            [{'host': self.es_host, 'port': 9200, 'scheme': 'https'}],
            basic_auth=(self.es_user, self.es_pass),
            verify_certs=False
        )
        
        if es.ping():
            print('成功连接到 Elasticsearch')
        else:
            print('无法连接到 Elasticsearch')
            
        return es


    def store_in_elasticsearch(self, content_dict):#将内容存入es
        print(f"Content dict to store: {content_dict}")
        for title, content in content_dict.items():
            try:
                es.index(index= index_cname, id=title, body={'content': '\n'.join(content)})
                print(f"已存储: {title}到{index_cname}")
            except exceptions.ConnectionError as e:
                print(f"连接错误：{e}")
            except exceptions.TransportError as e:
                print(f"存储错误：{e}")

    def split_and_index_doc(self):#将文档分割成篇章，然后调用存储到es函数存入
        if not os.path.exists(self.file_input):
            print(f"文件 {self.file_input} 不存在。")
            return

        try:
            doc_obj = docx.Document(self.file_input)
            try:
                # 假设这是调用 extract_titles_and_content 的地方
                content_dict = self.extract_titles_and_content(doc_obj)
            except Exception as e:
                print(f"Error during content extraction: {e}")
            
            self.store_in_elasticsearch(content_dict)
            print(f"已将 {len(content_dict)} 篇章存入 Elasticsearch")
        except Exception as e:
            print(f"处理文件时出错：{e}")

    

    def upload_doc(self, index_name, vector_db_path): #提交
        self.split_and_index_doc()
        vector_db_path = f"{vector_db_path}/{index_name}.npz"
        process_and_vectorize(index_name, vector_db_path)

def import_new_documents(uploaded_file, index_name, vector_db_path):# 上传文档
    global index_cname
    index_cname = index_name
    if uploaded_file is not None: 
        file_input = uploaded_file.name  # 获取上传文件的路径

        # 检查配置是否已设置
        if not config or 'es_host' not in config:
            return "请先更新配置!"

        uploader = UploadDoc(file_input=file_input)
        uploader.upload_doc(index_name,vector_db_path)
        
        return "文档上传成功"  
    else:
        return "没有上传文件"  

def LLM_QA(llm_q):# 调用llm进行qa环节
    print(f"LLM_QA 输入: {llm_q}")  # 调试信息
    qa_template = f"""你是一个药典问答机器人，请回答以下问题：
    {llm_q}
    
    请给出具体的回答。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": qa_template}],
            stream=False  # Set to False for normal output
        )

        answer = response.choices[0].message.content.strip() if response.choices else "无法提供答案"
        print(f"完整回答: {answer}")  # 调试信息
        return answer

    except Exception as e:
        print(f"API调用失败: {e}")  # 捕获并打印异常
        return "无法提供答案"

def slow_echo(message, history):# 问答环节主代码部分
    input_data = message# 输入的问题传入
    standardizer = MedicineInfoStandardizer(client)# 初始化信息标准化器
    history.append((message, ""))# 历史记录中加入用户输入的问题
    # 判断用户输入是否与药学相关
    query_type = classify_pharmacy_query(input_data)
    
    if query_type == "good":# 是药学相关的问题
        
        standardized_result = standardizer.standardize_information(input_data)# 标准化用户输入的问题

        if es:#使用es进行检索
            input_text = standardized_result
            print("这里是es步骤的产物：---------------------------------------")
            print(f"标准化结果: {input_text}")  # 打印标准化结果
            doc_id, sub_title = extract_drug_info(input_text)
            print(f"提取的药品ID: {doc_id}, 子标题: {sub_title}")  # 打印提取的ID和子标题

            combined_results = []
            for id, outputs in zip(doc_id, sub_title):
                print(f"当前ID: {id}, 输出: {outputs}")  # 打印当前处理的ID和对应的子标题
                for sub in outputs:
                    print(f"检索子标题: {sub}")  # 打印当前检索的子标题
                    result = verify_data_in_elasticsearch(es, es_indx, id, sub)
                    print(f"检索结果: {result}")  # 打印每次检索的结果
                    combined_results.append(result)

            unique_content = list(set(combined_results))
            final_output = "\n\n".join(unique_content) + "\n"
            print(f"es的检索结果: {final_output}")  # 打印最终检索结果
            print("----------------------------------------------------------")
        else:
            final_output = ""

        # 使用 向量化 检索
        llm_q = standardizer.bzh(input_data)
        output_lines = []
        for line in llm_q.splitlines():
            results = retrieve_vector_and_text(line.strip(), vector_db_path, top_k=1)
            for doc_id, title, text in results:
                output_lines.append(f"Document ID: {doc_id}, Title: {title}, Text: {text}")

        output_result = "\n".join(output_lines)# 向量化检索的结果
        final_output += output_result + "\n"# 向量化检索的结果加入到es的检索结果中

        context = "\n".join([f"用户: {msg}\n助手: {resp}" for msg, resp in history]) 
        final_query = f"这里是上下文：{context}\n\n这是你要回答的问题{input_data}请使用提供的数据信息进行回答:'{final_output}'.不要瞎编，涉及的敏感词请替换成同义词。可以润色内容，使其贴切或易懂,只回答我提的问题，问题没问的哪怕给了信息，也不用回答,可以结合用户和助手的对话内容，不要重复回答。"

        print(f"最终查询: {final_query}")  # 调试信息
        opening_statement = "小助手正在借助数据库思考，请稍后..."
        for i in range(len(opening_statement)):
            time.sleep(0.01)  
            yield opening_statement[:i + 1]# 逐步返回小助手的思考过程
        final_answer = LLM_QA(final_query)# 调用llm进行qa环节

    elif query_type == "bad":# 非药学相关的问题
        # 直接将用户的输入传到 LLM_QA
        context = "\n".join([f"用户: {msg}\n助手: {resp}" for msg, resp in history]) 
        final_query = f"这里是上下文：{context}\n\n这是你要回答的问题{input_data}.不要瞎编，可以润色内容，使其贴切或易懂,只回答我提的问题，问题没问的哪怕给了信息，也不用回答,可以结合用户和助手的对话内容，不要重复回答。"
        final_answer = LLM_QA(final_query)

    else:
        final_answer = "无法判断输入的问题类型。"

    # Add a newline for separation
    yield "\n"

    # Yield final answer word-by-word
    for i in range(len(final_answer)):
        time.sleep(0.03)  # Adjust the delay as needed
        yield final_answer[:i + 1]
    history[-1] = (message, final_answer.strip())

with gr.Blocks() as demo:# web页面效果主代码
    
    with gr.Tab("药典问答"):#问答页面
        # 创建 Chatbot 实例并设置高度
        gr.Markdown("**第一次使用请先去配置自己的信息并保存哦**")
        chatbot = gr.Chatbot(height=600)  # 设置高度为600像素
        qa_interface = gr.ChatInterface(
            fn=slow_echo,
            chatbot=chatbot  # 将自定义的 Chatbot 传递给 ChatInterface
        )

    
    with gr.Tab("文档导入"):# 文档导入页面
        folder_input = gr.Textbox(label="保存向量数据库的文件夹路径", value=r"C:/Users/abc10/桌面/workllm/ragproject")


        upload_interface = gr.Interface(
            fn=import_new_documents,
            inputs=[
                gr.File(label="上传新的药典文档"),
                gr.Textbox(label="数据库和索引名命名", placeholder="请输入"),
                folder_input,
            ],
            outputs=gr.Textbox(label="结果"),
            title="文档导入",
            description="上传新的药典文档以更新数据库。"
        )
    
    with gr.Tab("配置"):# 配置页面
        es_host_input = gr.Textbox(label="ES主机地址", value='192.168.110.28')
        es_port_input = gr.Textbox(label="ES服务端口", value='9200')
        es_user_input = gr.Textbox(label="ES用户名", value='elastic')
        es_pass_input = gr.Textbox(label="ES密码", type="password", value='7ztvwEMjr0H+_R4Vec*R')
        
        es_index_input = gr.Textbox(label="ES索引名（上传的会和向量数据库同名）", value='zhyd')  # 新增索引名输入框
        vector_db_path_input = gr.Textbox(label="向量数据库位置", value='C:/Users/abc10/桌面/workllm/ragproject/embeddings2.npz')  # 新增向量数据库位置输入框

        config_submit = gr.Button("保存配置")
        config_message = gr.Textbox(label="状态", interactive=False)

        config_submit.click(
            fn=update_config,
            inputs=[es_host_input, es_port_input, es_user_input, es_pass_input, es_index_input, vector_db_path_input],
            outputs=config_message
        )


# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",share=True)
