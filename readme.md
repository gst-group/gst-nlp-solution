项目目录介绍
.
├── readme.md                   #readme
├── requirements.txt            #pip install安装依赖项
└── src                         #源代码目录
    ├── __init__.py
    ├── api                     #web api源代码路径
    │   ├── __init__.py
    │   ├── app.py              #web api源代码入口
    │   └── temp
    │       └── __init__.py
    ├── assets                  #资源文件目录
    │   ├── stop_word_my.txt
    │   └── userdict_insurance.txt
    ├── data                    #数据集
    ├── main.py
    ├── models                  #模型算法
    │   ├── __init__.py
    │   └── random_forest_classifier.py
    ├── output                  #结果输出
    │   ├── xxx_84b76797-59dd-4f6a-9dcd-69fcaeb782a1.csv
    │   └── xxx_c4fce464-f0c1-4a9a-994a-4ec609759ace.csv
    ├── tests                   #单元测试
    │   ├── __init__.py
    │   ├── test_xxx.py
    └── utils.py                #自定义工具类

Enjoy it!