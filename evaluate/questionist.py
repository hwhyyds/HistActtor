from transformers import AutoTokenizer, AutoModel
import torch
import re
import time
import os
import sys
import json
import jieba
import openai
import requests
import httpx
from dashscope import Generation
import pandas as pd
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import qianfan
from prompt import evaluate_prompts

# 加载模型
def get_sushi_answer(json_data, name="local"):
    '''
    通过name来指定对应的苏轼模型
    '''
    if name == "base":
        url = "http://localhost:5001"
        response = requests.post(url, json=json_data)
        if response.status_code == 200:
            sushi_answer = json_data.copy()
            sushi_answer.append({"role": "assistant", "content": response.json()[-1]["content"]})
            # sushi_answer = response.json()

    if name == "local":
        url = "http://localhost:5000"
        response = requests.post(url, json=json_data)
        if response.status_code == 200:
            sushi_answer = json_data.copy()
            sushi_answer.append({"role": "assistant", "content": response.json()[-1]["content"]})

    elif name == "local_rlhf":
        url = "http://localhost:5001"
        response = requests.post(url, json=json_data)
        if response.status_code == 200:
            sushi_answer = json_data.copy()
            sushi_answer.append({"role": "assistant", "content": response.json()[-1]["content"]})


    # 通义大模型
    elif name == "tongyi":
        response = Generation.call(
            api_key=os.getenv("ALI_API_KEY"),
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=json_data,
            max_tokens=512,
            result_format="message"
        )
        if response.status_code == 200:
            sushi_answer = json_data.copy()
            sushi_answer.append({"role": "assistant", "content": response.output.choices[0].message.content})

    # 百川NPC大模型
    elif name == "baichuanNPC":
        url = "https://api.baichuan-ai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('BAICHUAN_API_KEY')}"
        }
        body = {
            "model": "Baichuan-NPC-Turbo",
            "character_profile": {
                "character_id": 32252
            },
            "messages": json_data,
            "max_tokens": 512
        }
        response = requests.post(url, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            sushi_answer = json_data.copy()
            sushi_answer.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})

    # 百川大模型
    elif name == "baichuan":
        url = "https://api.baichuan-ai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('BAICHUAN_API_KEY')}"
        }
        body = {
            "model": "Baichuan4",
            "messages": json_data,
            "max_tokens": 512
        }
        response = requests.post(url, headers=headers, data=json.dumps(body))
        print(response.text)
        if response.status_code == 200:
            sushi_answer = json_data.copy()
            sushi_answer.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})

    # 零一万物
    elif name == "lingyi":
        API_BASE = "https://api.lingyiwanwu.com/v1"
        client = OpenAI(
            api_key=os.getenv("LINGYI_API_KEY"),
            base_url=API_BASE
        )
        completion = client.chat.completions.create(
            model="yi-lightning",
            messages=json_data,
            max_tokens=512
        )
        sushi_answer = json_data.copy()
        sushi_answer.append({"role": "assistant", "content": completion.choices[0].message.content})

    # 星火大模型
    elif name == "xinghuo":
        url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('XINGHUO_APP_ID')}"
        }
        data = {
            "model": "lite",
            "messages": json_data,
            "max_tokens": 512
        }
        response = requests.post(url, headers=headers, data=json.dumps(data)).json()
        content = response["choices"][0]["message"]["content"]
        sushi_answer = json_data.copy()
        sushi_answer.append({"role": "assistant", "content": content})

    # kimi模型
    elif name == "kimi":
        client = OpenAI(
            api_key=os.getenv("KIMI_API_KEY"),
            base_url="https://api.moonshot.cn/v1",
        )
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=json_data,
            max_tokens=512
        )
        sushi_answer = json_data.copy()
        sushi_answer.append({"role": "assistant", "content": response.choices[0].message.content})

    # 腾讯混元
    elif name == "hunyuan":
        sushi_answer = []
        new_chat = [
            json_data[0],
            json_data[-1]
        ]
        client = OpenAI(
            api_key=os.environ.get("HUNYUAN_API_KEY"),  # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
        )
        resp = client.chat.completions.create(
            model="hunyuan-turbo",
            messages=new_chat,
            max_tokens=512,
        )
        sushi_answer = json_data.copy()
        sushi_answer.append({"role": "assistant", "content": resp.choices[0].message.content})

    # 豆包
    elif name == "doubao":

        model = os.getenv("DOUBAO_MODEL")
        api_key = os.getenv("DOUBAO_API_KEY")

        client = Ark(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        # print(json_data)
        resp = client.chat.completions.create(
            model=model,
            messages=json_data,
            # max_tokens=512
        )
        sushi_answer = json_data.copy()
        sushi_answer.append({"role": "assistant", "content": resp.choices[0].message.content})

    else:
        raise ValueError(f"The method {name} doesn't be defined")

    return sushi_answer

def extract_first_number(s):
    match = re.search(r'\d+', s)  # \d+ 表示匹配一个或多个数字
    if match:
        return match.group()  # 返回匹配的第一个数字
    return None


def MBTI_evaluation(prompt, questions, name="local"):
    path = f"./output/mbti/{name}.xlsx"
    if os.path.exists(path):
        print(f"name")
        return
    messages = [{"role": "system", "content": prompt}]
    for question in questions:
        if question == "你喜欢尝试未经检验的新方法。" and "baichuan" in name:
            question = "你喜欢尝试没有被检验的新方法"  # 由于baichuan的敏感词，题目改为这个
        messages.append({"role": "user", "content": question})
        messages = get_sushi_answer(json_data=messages, name=name)
        time.sleep(5)
        if name == "kimi":  # kimi的每分钟请求速率限制
            time.sleep(60)
    pd.DataFrame(messages, index=None).to_excel(path, index=False)
    print(f"MBTI files: {path} has created")
    return messages

def QA_evaluation(prompt, questions, name="local"):
    path = f"./output/qa/{name}.xlsx"
    if os.path.exists(path):
        print(f"name")
        return
    messages = [{"role": "system", "content": prompt}]
    for question in questions:
        messages.append({"role": "user", "content": question})
        messages = get_sushi_answer(json_data=messages, name=name)
        time.sleep(5)
        if name == "kimi":  # kimi的每分钟请求速率限制
            time.sleep(60)
    pd.DataFrame(messages, index=None).to_excel(path, index=False)
    print(f"MBTI files: {path} has created")
    return messages

def main(actor):
    names = ["tongyi", "loccal", "local_rlhf", "base", "doubao", "hunyuan", "kimi", "lingyi"]
    mbti_prompt = evaluate_prompts[actor]["mbti_prompt"]
    qa_prompt = evaluate_prompts[actor]["qa_prompt"]

    qa_questions = pd.read_excel("qa.xlsx")["question"]
    mbti_questions = pd.read_excel("mbti.xlsx")["question"]

    for name in names:
        MBTI_evaluation(mbti_prompt, mbti_questions, name)
        QA_evaluation(qa_prompt, qa_questions, name)

if __name__ == "__main__":
    main(actor="sushi")
    main(actor="socrate")


