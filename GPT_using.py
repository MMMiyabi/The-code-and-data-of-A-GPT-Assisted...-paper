import openai
import jsonlines
from demonstrations import choose_demonstrations
from rules import material_rules, structure_rules, method_rules, RL_rules, thickness_rules, EAB_rules
openai.api_key = ""

def construct_prompt(demonstration):

    input, output = [], []
    for k, v in demonstration.items():
        dict = v
        input.append(dict['text'])
        # output.append(str({"h": dict['h']['name'], "t": dict['t']['name'], "relation": dict['relation']}))
        output.append(dict['dict'])
    prompt = "You are provided with a text paragraph that contains various technical entities. " \
             "Your task is to identify and extract these entities accurately and categorize them into the predefined template. " \
             "The entity categories include 'Material', 'Structure', 'Method', 'Thickness', 'EAB', and 'RL'. " \
             "'Material' refers to terms that denote chemical elements or compounds, sometimes accompanied by descriptors of their microstructures. " \
             "'Structure' encompasses macroscopic structural descriptions such as 'one-dimensional', 'multi-layered', 'spherical shells', as well as specific structural terms like 'MOF' and 'aerogel'. " \
             "'Method' indicates the fabrication or synthesis techniques used for creating materials. " \
             "'Thickness' is a numerical value related to dimensions, typically presented in millimeters (mm). " \
             "'EAB' stands for the effective absorption bandwidth, which is a numerical range often expressed in gigahertz (GHz). Lastly, " \
             "'RL' denotes reflection loss, represented by a numerical value in decibels (dB). " \
             "Your goal is to parse the paragraph, identify these entities, and place them into the correct slots in the template provided. Note that not all entity types will always appear in a paragraph. Below are three examples."
    ip1, op1, ip2, op2, ip3, op3 = input[0], output[0], input[1], output[1], input[2], output[2]
    return prompt, ip1, str(op1), ip2, str(op2), ip3, str(op3)


# sentences = ["Transition metal dichalcogenide MoS2 is considered as a type of dielectric loss dominated electromagnetic wave absorbing material owing to the high specific surface area , layered structure , and lightweight ."]
def GPT_predict(data_path, tag, results_path, database_path):

    datas = []
    with open(data_path, "r", encoding="utf-8") as file:
        if tag == False:
            for line in file:
                sentence = line.strip()  # 去除首尾的空白字符
                datas.append([sentence])
        else:
            for line in file:
                data = eval(line)
                datas.append([data['text'], data['dict']])

    results, gold, pred = [], [], []
    for d in datas:
        sentence = d[0]
        demonstrations = choose_demonstrations(sentence, database_path)
        prompt, ip1, op1, ip2, op2, ip3, op3 = construct_prompt(demonstrations)
        completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ip1},
            {"role": "assistant", "content": op1},
            {"role": "user", "content": ip2},
            {"role": "assistant", "content": op2},
            {"role": "user", "content": ip3},
            {"role": "assistant", "content": op3},
            {"role": "user", "content": sentence},
        ],
        temperature=0,
        )
        answer = completion.choices[0].message.content
        try:
            print(sentence)
            print(answer)
            result = {}
            result['text'] = sentence
            result['dict'] = answer
            results.append(result)
        except Exception as e:
            print("eval() 函数发生错误:", str(e))
            continue
    if tag == True:
        count = 0
        for i in range(len(gold)):
            if gold[i] == pred[i]:
                count += 1
        print(count/len(gold))
    with jsonlines.open(results_path, 'w') as writer:
        writer.write_all(results)

def GPT_process(in_path, out_path):
    datas = []
    f = jsonlines.open(in_path)
    for d in f:
        dict = d
        dict['dict'] = eval(dict['dict'])
        data = {}
        data['text'] = dict['text']
        data['dict'] = {}
        data['dict']['Material'], data['dict']['Structure'], data['dict']['Method'], data['dict']['RL'], data['dict']['Thickness'], data['dict']['EAB'] = \
            material_rules(dict['dict']['Material']), structure_rules(dict['dict']['Structure']), method_rules(dict['dict']['Method']), \
            RL_rules(dict['dict']['RL']), thickness_rules(dict['dict']['Thickness']), EAB_rules(dict['dict']['EAB'])
        datas.append(data)
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(datas)
