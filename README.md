## 目录


- [快速上手](#getting-started)
- [更多教程](#further-tutorials)
- [引用](#citation)



## 快速上手


### 第一步：安装

#### 环境要求
在安装之前，请确保您的系统满足以下要求：

- **Python**：版本 3.10 至 3.12（含）
- **CUDA**：版本 12.4 至 12.8（含）

#### 1. 从源码安装
```bash
git https://github.com/Javkonline/AMoPO.git
cd AMoPO/
```
##### 2. 创建虚拟环境

###### 使用 Conda
```bash
conda create -n amopo python=3.10
conda activate amopo
```

##### 3. 安装软件包

```bash
pip install -r requeriments.txt
```

### 第二步：准备数据集和模型

目前已经支持已经扩充后的HelpSteer2数据集，如需要数据集，整个数据集的格式应该仿照HelpSteer2数据集。
```bash
{
    "prompt": "\n\nQuestion: \nIf I want to pick one signature appetizer to represent each Asian country, what do you suggest? and why\uff1f\n\nSingapore \nMalaysia\nIndonesia \nCambodia \nThailand\nThe Philippines \nVietnam\nGuam\nHawaii \nKorea \nJapan \nTaiwan \nMacau \nHong Kong \nChina\nIndia \nTurkey",
    "chosen": {
        "response": "Here are some suggestions for a signature appetizer from each Asian country:\n\nSingapore: Chili Crab Buns - a delicious combination of sweet and savory flavors, with a spicy kick from the chili crab sauce.\n\nMalaysia: Satay - grilled skewers of meat (usually chicken or beef) served with a delicious peanut sauce.\n\nIndonesia: Gado-Gado - a delicious salad made with boiled vegetables, served with a peanut sauce dressing.\n\nCambodia: Amok - a traditional Khmer curry, served in a coconut shell and made with fish, lemongrass, and other herbs and spices.\n\nThailand: Tom Yum Goong - a spicy and sour soup made with shrimp, lemongrass, and other herbs and spices.\n\nThe Philippines: Adobo - a popular Filipino dish made with chicken or pork, cooked in a vinegar-based sauce with garlic, onions, and other spices.\n\nVietnam: Nem Nuong - grilled pork meatballs served with lettuce, herbs, and rice paper rolls.\n\nGuam: Chamorro Spring Rolls - a unique fusion dish that combines traditional Chamorro ingredients like pork belly and green beans with spring roll ingredients.\n\nHawaii: Kalua Pig Sliders - a delicious take on the traditional Hawaiian dish of kalua pig, served on mini buns with a touch of sweetness from the pineapple salsa.\n\nKorea: Kimchi - a traditional Korean side dish made with fermented cabbage and other vegetables, seasoned with spices and served as an appetizer.\n\nJapan: Edamame - boiled soybeans, seasoned with salt and served as a popular snack.\n\nTaiwan: Beef Noodle Soup - a delicious soup made with tender beef, noodles, and a flavorful broth.\n\nMacau: Portuguese Chicken - a unique dish that combines Portuguese and Chinese flavors, made with chicken, garlic, and soy sauce.\n\nHong Kong: Dim Sum - a variety of small dishes, including steamed buns, dumplings, and other savory treats.\n\nChina: Spring Rolls - a popular Chinese appetizer made with a variety of fillings, including vegetables and meat, and served with a dipping sauce.\n\nIndia: Chaat - a variety of delicious street food snacks, including gol gappe, aloo tikki, and pani puri.\n\nTurkey: Hummus - a popular Middle Eastern appetizer made with chickpeas, tahini, and other ingredients, served with pita bread or vegetables.",
        "helpfulness": "4",
        "correctness": "4",
        "coherence": "4",
        "complexity": "1",
        "verbosity": "2",
        "instruction_following": "3"
    },
    "rejected": {
        "response": "Here are some signature appetizers that could represent each Asian country:\n\nSingapore - Chili Crab: A sweet and spicy crab dish that is a popular street food in Singapore.\n\nMalaysia - Satay: Grilled meat skewers served with a peanut sauce, a popular street food found throughout Malaysia.\n\nIndonesia - Gado-Gado: A vegetarian dish of boiled vegetables and tofu served with a peanut sauce, a popular street food in Indonesia.\n\nCambodia - Amok: A creamy fish curry dish steamed in a banana leaf, a traditional Cambodian dish.\n\nThailand - Tom Yum Goong: A spicy and sour soup with shrimp, mushrooms, and lemongrass, a popular Thai appetizer.\n\nThe Philippines - Adobo: A vinegar and soy-based dish with chicken or pork, a popular Filipino dish.\n\nVietnam - Spring Rolls: Fresh spring rolls with rice paper, filled with vegetables, herbs, and shrimp, a popular Vietnamese appetizer.\n\nGuam - Spam Musubi: A Hawaiian dish of Spam (a popular canned meat) and rice wrapped in seaweed, a popular dish in Guam with strong Hawaiian influences.\n\nHawaii - Poke: A raw fish salad with soy sauce, sesame oil, and other seasonings, a popular Hawaiian dish.\n\nKorea - Kimchi: A fermented cabbage dish with a spicy and sour flavor, a staple in Korean cuisine.\n\nJapan - Sushi: Raw fish and rice wrapped in seaweed, a popular Japanese dish.\n\nTaiwan - Beef Noodle Soup: A noodle soup with thin slices of beef and a flavorful broth, a popular dish in Taiwan.\n\nMacau - Portuguese Egg Tarts: A sweet pastry with a creamy egg filling, a popular dessert in Macau with strong Portuguese influences.\n\nHong Kong - Dim Sum: A variety of small dishes, such as steamed buns, dumplings, and rice rolls, a popular tradition in Hong Kong.\n\nChina - Kung Pao Chicken: A spicy chicken dish with peanuts and chili peppers, a popular dish in Sichuan province.\n\nIndia - Samosas: A fried pastry filled with potatoes and spices, a popular snack in India.\n\nTurkey - Hummus: A dip made from chickpeas, tahini, and lemon juice, a popular Middle Eastern dish.",
        "helpfulness": "3",
        "correctness": "3",
        "coherence": "4",
        "complexity": "1",
        "verbosity": "2",
        "instruction_following": "3"
    }
}
```

如需更改对应的System Prompt，可以参考pairwise_multi_object[src/llamafactory/data/processors/pairwise_multi_object.py]

### 第三步：运行 AMoPO 流程
```bash
bash AMoPO/run_amopo/run_bash_accelerate.sh
```

## 引用
```bibtex
@inproceedings{liu-etal-2025-amopo,
    title = "{AM}o{PO}: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models",
    author = "Liu, Qi  and
      Ruan, Jingqing  and
      Li, Hao  and
      Zhao, Haodong  and
      Wang, Desheng  and
      Chen, Jiansong  and
      Guanglu, Wan  and
      Cai, Xunliang  and
      Zheng, Zhi  and
      Xu, Tong",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.462/",
    doi = "10.18653/v1/2025.findings-acl.462"
}
```
