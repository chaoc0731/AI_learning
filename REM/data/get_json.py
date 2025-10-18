import json

# 要保存的数据


data1 = {
  "dataset_info": {
    "name": "医疗关系抽取数据集",
    "description": "包含50条医疗文本的关系抽取标注数据",
    "size": 50,
    "relation_types": ["治疗", "引起", "预防", "诊断", "副作用", "无关系"],
    "entity_types": ["疾病", "症状", "药物", "检查", "并发症"],
    "created_date": "2024-01-15"
     },
  "data": [
    {
      "text": "患者服用阿司匹林后头痛症状明显缓解",
      "entities": [
        {"text": "阿司匹林", "type": "药物", "start": 3, "end": 7},
        {"text": "头痛", "type": "症状", "start": 9, "end": 11}
      ],
      "relation": "治疗"
    },
    {
      "text": "糖尿病可能引起视网膜病变",
      "entities": [
        {"text": "糖尿病", "type": "疾病", "start": 0, "end": 3},
        {"text": "视网膜病变", "type": "并发症", "start": 6, "end": 11}
      ],
      "relation": "引起"
    },
    {
      "text": "高血压患者需要定期服用硝苯地平控制血压",
      "entities": [
        {"text": "高血压", "type": "疾病", "start": 0, "end": 3},
        {"text": "硝苯地平", "type": "药物", "start": 11, "end": 15}
      ],
      "relation": "治疗"
    },
    {
      "text": "CT检查可以帮助诊断肺癌",
      "entities": [
        {"text": "CT检查", "type": "检查", "start": 0, "end": 4},
        {"text": "肺癌", "type": "疾病", "start": 9, "end": 11}
      ],
      "relation": "诊断"
    },
    {
      "text": "长期吸烟会导致慢性支气管炎",
      "entities": [
        {"text": "吸烟", "type": "症状", "start": 3, "end": 5},
        {"text": "慢性支气管炎", "type": "疾病", "start": 8, "end": 14}
      ],
      "relation": "引起"
    },
    {
      "text": "接种流感疫苗可以有效预防流感",
      "entities": [
        {"text": "流感疫苗", "type": "药物", "start": 2, "end": 6},
        {"text": "流感", "type": "疾病", "start": 12, "end": 14}
      ],
      "relation": "预防"
    },
    {
      "text": "青霉素可能引起过敏反应",
      "entities": [
        {"text": "青霉素", "type": "药物", "start": 0, "end": 3},
        {"text": "过敏反应", "type": "症状", "start": 7, "end": 11}
      ],
      "relation": "副作用"
    },
    {
      "text": "MRI扫描用于检测脑部肿瘤",
      "entities": [
        {"text": "MRI扫描", "type": "检查", "start": 0, "end": 6},
        {"text": "脑部肿瘤", "type": "疾病", "start": 10, "end": 14}
      ],
      "relation": "诊断"
    },
    {
      "text": "阿托伐他汀用于降低胆固醇",
      "entities": [
        {"text": "阿托伐他汀", "type": "药物", "start": 0, "end": 5},
        {"text": "胆固醇", "type": "症状", "start": 9, "end": 12}
      ],
      "relation": "治疗"
    },
    {
      "text": "肥胖容易引发二型糖尿病",
      "entities": [
        {"text": "肥胖", "type": "症状", "start": 0, "end": 2},
        {"text": "二型糖尿病", "type": "疾病", "start": 6, "end": 11}
      ],
      "relation": "引起"
    },
    {
      "text": "胃镜检查发现胃溃疡",
      "entities": [
        {"text": "胃镜检查", "type": "检查", "start": 0, "end": 4},
        {"text": "胃溃疡", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "胰岛素治疗糖尿病效果显著",
      "entities": [
        {"text": "胰岛素", "type": "药物", "start": 0, "end": 3},
        {"text": "糖尿病", "type": "疾病", "start": 4, "end": 7}
      ],
      "relation": "治疗"
    },
    {
      "text": "缺乏运动可能导致心血管疾病",
      "entities": [
        {"text": "缺乏运动", "type": "症状", "start": 0, "end": 4},
        {"text": "心血管疾病", "type": "疾病", "start": 8, "end": 13}
      ],
      "relation": "引起"
    },
    {
      "text": "心电图检查心律不齐",
      "entities": [
        {"text": "心电图", "type": "检查", "start": 0, "end": 3},
        {"text": "心律不齐", "type": "症状", "start": 5, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "维生素C补充剂预防感冒",
      "entities": [
        {"text": "维生素C补充剂", "type": "药物", "start": 0, "end": 7},
        {"text": "感冒", "type": "疾病", "start": 9, "end": 11}
      ],
      "relation": "预防"
    },
    {
      "text": "化疗药物会引起恶心呕吐",
      "entities": [
        {"text": "化疗药物", "type": "药物", "start": 0, "end": 4},
        {"text": "恶心呕吐", "type": "症状", "start": 7, "end": 11}
      ],
      "relation": "副作用"
    },
    {
      "text": "B超检查显示肝脏正常",
      "entities": [
        {"text": "B超检查", "type": "检查", "start": 0, "end": 4},
        {"text": "肝脏", "type": "症状", "start": 7, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": " Metformin控制血糖水平",
      "entities": [
        {"text": "Metformin", "type": "药物", "start": 1, "end": 10},
        {"text": "血糖", "type": "症状", "start": 12, "end": 14}
      ],
      "relation": "治疗"
    },
    {
      "text": "高盐饮食导致高血压",
      "entities": [
        {"text": "高盐饮食", "type": "症状", "start": 0, "end": 4},
        {"text": "高血压", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "引起"
    },
    {
      "text": "X光片诊断骨折",
      "entities": [
        {"text": "X光片", "type": "检查", "start": 0, "end": 3},
        {"text": "骨折", "type": "疾病", "start": 6, "end": 8}
      ],
      "relation": "诊断"
    },
    {
      "text": "接种HPV疫苗预防宫颈癌",
      "entities": [
        {"text": "HPV疫苗", "type": "药物", "start": 2, "end": 8},
        {"text": "宫颈癌", "type": "疾病", "start": 10, "end": 13}
      ],
      "relation": "预防"
    },
    {
      "text": "抗生素使用可能导致肠道菌群失调",
      "entities": [
        {"text": "抗生素", "type": "药物", "start": 0, "end": 3},
        {"text": "肠道菌群失调", "type": "症状", "start": 9, "end": 15}
      ],
      "relation": "副作用"
    },
    {
      "text": "PET-CT检测癌症转移",
      "entities": [
        {"text": "PET-CT", "type": "检查", "start": 0, "end": 6},
        {"text": "癌症", "type": "疾病", "start": 8, "end": 10}
      ],
      "relation": "诊断"
    },
    {
      "text": "降压药控制血压稳定",
      "entities": [
        {"text": "降压药", "type": "药物", "start": 0, "end": 3},
        {"text": "血压", "type": "症状", "start": 5, "end": 7}
      ],
      "relation": "治疗"
    },
    {
      "text": "长期饮酒会引起肝硬化",
      "entities": [
        {"text": "长期饮酒", "type": "症状", "start": 0, "end": 4},
        {"text": "肝硬化", "type": "疾病", "start": 8, "end": 11}
      ],
      "relation": "引起"
    },
    {
      "text": "超声检查发现肾结石",
      "entities": [
        {"text": "超声检查", "type": "检查", "start": 0, "end": 4},
        {"text": "肾结石", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "阿司匹林预防心肌梗死",
      "entities": [
        {"text": "阿司匹林", "type": "药物", "start": 0, "end": 4},
        {"text": "心肌梗死", "type": "疾病", "start": 6, "end": 10}
      ],
      "relation": "预防"
    },
    {
      "text": "化疗可能导致脱发",
      "entities": [
        {"text": "化疗", "type": "治疗", "start": 0, "end": 2},
        {"text": "脱发", "type": "症状", "start": 6, "end": 8}
      ],
      "relation": "副作用"
    },
    {
      "text": "血常规检查贫血",
      "entities": [
        {"text": "血常规", "type": "检查", "start": 0, "end": 3},
        {"text": "贫血", "type": "症状", "start": 5, "end": 7}
      ],
      "relation": "诊断"
    },
    {
      "text": "胰岛素依赖型糖尿病",
      "entities": [
        {"text": "胰岛素", "type": "药物", "start": 0, "end": 3},
        {"text": "糖尿病", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "治疗"
    },
    {
      "text": "空气污染引发呼吸道疾病",
      "entities": [
        {"text": "空气污染", "type": "症状", "start": 0, "end": 4},
        {"text": "呼吸道疾病", "type": "疾病", "start": 6, "end": 11}
      ],
      "relation": "引起"
    },
    {
      "text": "胃镜诊断胃炎",
      "entities": [
        {"text": "胃镜", "type": "检查", "start": 0, "end": 2},
        {"text": "胃炎", "type": "疾病", "start": 5, "end": 7}
      ],
      "relation": "诊断"
    },
    {
      "text": "疫苗接种预防传染病",
      "entities": [
        {"text": "疫苗", "type": "药物", "start": 0, "end": 2},
        {"text": "传染病", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "预防"
    },
    {
      "text": "长期服药可能导致肝损伤",
      "entities": [
        {"text": "长期服药", "type": "症状", "start": 0, "end": 4},
        {"text": "肝损伤", "type": "症状", "start": 8, "end": 11}
      ],
      "relation": "副作用"
    },
    {
      "text": "CT扫描诊断肺炎",
      "entities": [
        {"text": "CT扫描", "type": "检查", "start": 0, "end": 4},
        {"text": "肺炎", "type": "疾病", "start": 7, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "降压药物治疗高血压",
      "entities": [
        {"text": "降压药物", "type": "药物", "start": 0, "end": 4},
        {"text": "高血压", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "治疗"
    },
    {
      "text": "高脂饮食导致动脉硬化",
      "entities": [
        {"text": "高脂饮食", "type": "症状", "start": 0, "end": 4},
        {"text": "动脉硬化", "type": "疾病", "start": 6, "end": 10}
      ],
      "relation": "引起"
    },
    {
      "text": "B超诊断胆囊结石",
      "entities": [
        {"text": "B超", "type": "检查", "start": 0, "end": 2},
        {"text": "胆囊结石", "type": "疾病", "start": 5, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "维生素D预防骨质疏松",
      "entities": [
        {"text": "维生素D", "type": "药物", "start": 0, "end": 5},
        {"text": "骨质疏松", "type": "疾病", "start": 7, "end": 11}
      ],
      "relation": "预防"
    },
    {
      "text": "放疗引起皮肤灼伤",
      "entities": [
        {"text": "放疗", "type": "治疗", "start": 0, "end": 2},
        {"text": "皮肤灼伤", "type": "症状", "start": 5, "end": 9}
      ],
      "relation": "副作用"
    },
    {
      "text": "心电图诊断冠心病",
      "entities": [
        {"text": "心电图", "type": "检查", "start": 0, "end": 3},
        {"text": "冠心病", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "胰岛素注射控制糖尿病",
      "entities": [
        {"text": "胰岛素", "type": "药物", "start": 0, "end": 3},
        {"text": "糖尿病", "type": "疾病", "start": 7, "end": 10}
      ],
      "relation": "治疗"
    },
    {
      "text": "吸烟引起慢性阻塞性肺病",
      "entities": [
        {"text": "吸烟", "type": "症状", "start": 0, "end": 2},
        {"text": "慢性阻塞性肺病", "type": "疾病", "start": 5, "end": 12}
      ],
      "relation": "引起"
    },
    {
      "text": "MRI诊断脑梗死",
      "entities": [
        {"text": "MRI", "type": "检查", "start": 0, "end": 3},
        {"text": "脑梗死", "type": "疾病", "start": 6, "end": 9}
      ],
      "relation": "诊断"
    },
    {
      "text": "流感疫苗预防季节性流感",
      "entities": [
        {"text": "流感疫苗", "type": "药物", "start": 0, "end": 4},
        {"text": "季节性流感", "type": "疾病", "start": 6, "end": 11}
      ],
      "relation": "预防"
    },
    {
      "text": "抗癌药物导致白细胞减少",
      "entities": [
        {"text": "抗癌药物", "type": "药物", "start": 0, "end": 4},
        {"text": "白细胞减少", "type": "症状", "start": 6, "end": 11}
      ],
      "relation": "副作用"
    },
    {
      "text": "X光诊断肺结核",
      "entities": [
        {"text": "X光", "type": "检查", "start": 0, "end": 2},
        {"text": "肺结核", "type": "疾病", "start": 5, "end": 8}
      ],
      "relation": "诊断"
    },
    {
      "text": "降压药维持血压正常",
      "entities": [
        {"text": "降压药", "type": "药物", "start": 0, "end": 3},
        {"text": "血压", "type": "症状", "start": 5, "end": 7}
      ],
      "relation": "治疗"
    },
    {
      "text": "糖尿病引起肾病",
      "entities": [
        {"text": "糖尿病", "type": "疾病", "start": 0, "end": 3},
        {"text": "肾病", "type": "并发症", "start": 6, "end": 8}
      ],
      "relation": "引起"
    }
            ]
        }

data2 = {
  "治疗": 0,
  "引起": 1,
  "预防": 2,
  "诊断": 3,
  "副作用": 4,
  "无关系": 5
}


# 写入JSON文件
with open('raw_data.json', 'w', encoding='utf-8') as f:
    json.dump(data1, f, ensure_ascii=False, indent=4)

with open('label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(data2, f, ensure_ascii=False, indent=4)

print("JSON文件创建成功！")