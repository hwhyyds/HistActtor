import json

from fetch_gpt_response import FullScene, get_gpt_answer
import random
import base64

generate_scene_prompt = {
    "en": {
        "socrates": [
            "Imagine a scene in which 10 people from different walks of life talk to the protagonist Socrates. This scene can be modern or ancient. It can be assumed that Socrates travels to modern times or someone travels to ancient times.\nThe theme of these scenes can be modern new things, or it can be about Socrates’s life and personal experience, personal information exchange and communication, and it can also invite Socrates to express some views aimed at different people.\nDescription of the scene should be concise, focus on the background, do not go into details. Try to be innovative and diverse. Don’t omit.",
            "Imagine the themes of conversations between ten people from different industries and the protagonist Socrates. These conversations are about Socrates's life, personal experiences, and personal information. They can involve asking Socrates for his opinions on others, various things, and different concepts and ideas. These people could be those Socrates knew and had intersections with, or they could be modern people who traveled back in time to have a dialogue with Socrates.\nDescription of the scene should be concise, focus on the background, do not go into details. Try to be innovative and diverse. Don’t omit."
            "Imagine 10 different people talking to the main character Socrates. Please choose from the following topics:\n1. The works of Socrates;\n2. Evaluation of various works in history;\n3. Evaluation of famous people in history;\n4. Something related to Socrates.\nDescription of the scene should be concise, focus on the background, do not go into details. Try to be innovative and diverse. Don’t omit."
        ]
    },
    "ch": {
        "sushi": [
            "想象10个不同行业的人和主角苏轼对话的场景，这个场景可以是现代也可以是古代，可以假设为苏轼穿越到现代或某人穿越到古代。 这些场景的主题可以是现代的新事物，也可以是关于苏轼的生平和个人经历，个人信息的交流沟通，还可以是邀请苏轼发表一些针对于不同的人的观点。 场景描述要简洁，注重背景，不讲细节。尝试创新和多样化。不要省略。",
            "想象10个不同行业的人和主角苏轼对话的主题，这些对话的主题是关于苏轼的生平和个人经历，个人信息的，可以是询问苏轼对于其他人的看法，各事物的看法和各种不同的观念，思想等看法。 这些人可以是苏轼认识的，有交集的人，也可以是现代人穿越到苏轼年代与苏轼对话的人。主题描述要简洁，注重背景，不讲细节。尝试创新和多样化。不要省略。",
            "想象10个不同的人和主角苏轼对话的主题，这些对话谈论的主题请在以下几个主题中选择：1.苏轼的作品；2.对于历史上各种作品的评价；3.对于历史上的名人的评价；4.与苏轼有一定关联的东西。 主题描述要简洁，注重背景，不讲细节。尝试创新和多样化。不要省略。"
        ]
    }
}

def main(times: int = 100, language: str = "en", person_name: str = "socrates"):
    if person_name == "socrates":
        with open("pdf_knowledge/Socrates/Socrates.pdf", "rb") as f:
            data = f.read()
    else:
        with open("pdf_knowledge/Sushi/Sushi.pdf", "rb") as f:
            data = f.read()
    base64_string = base64.b64encode(data).decode("utf-8")

    if language=="en":
        user_prompt = "Please generate ten scenarios as required"
    else:
        user_prompt = "请安装要求生成相应的十个场景"
    Scenarios = []
    for i in range(times):
        prompt = random.choice(generate_scene_prompt[language][person_name])
        input_message = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {
                    "type": "input_file",
                    "text": f"data:application/pdf;base64,{base64_string}",
                },
                {
                    "type": "input_text",
                    "text": user_prompt,
                }, ]}
        ]
        result = get_gpt_answer(
            message_schema=FullScene,
            input=input_message
        )
        Scenarios.extend(result["SceneList"])
    with open(f"Scenario/{person_name}_scenario.json", "w", encoding="utf-8") as f:
        json.dump(Scenarios, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    main()
