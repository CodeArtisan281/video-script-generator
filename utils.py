from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper

def generate_script(subject,video_length,api_key,base_url,creativity):

    # 用ChatPromptTemplate的from_messages函数创建一个标题提示模板
    title_template = ChatPromptTemplate.from_messages(
        [
            ('human','请为{subject}这个主题想一个吸引人的标题，并且使用简体中文展现出来')
        ]
    )

    # 用ChatPromptTemplate的from_messages函数创建一个正文提示模板
    script_template = ChatPromptTemplate.from_messages(
        [
            ('human',
             """你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
             视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
             要求开头抓住限球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
             整体内容的表达方式要尽量轻松有趣，吸引年轻人。脚本的语言请用简体中文。
             脚本内容可以结合以下维基百科搜索出的信息，但仅作为参考，只结合相关的即可，对不相关的进行忽略：
             ```{wikipedia_search}```""")
        ]
    )

    # 使用'gpt-3.5-turbo'模型并设置好参数
    model = ChatOpenAI(
        model = 'gpt-3.5-turbo',
        api_key = api_key,
        base_url = base_url,
        temperature = creativity,
        frequency_penalty = 1.1
    )

    # 调用维基百科的API进行搜索
    search = WikipediaAPIWrapper(lang = 'zh')
    search_result = search.run(subject)

    """使用管道操作用invoke方法用AI进行生成"""

    # 用AI生成标题
    title_chain = title_template | model
    title = title_chain.invoke({'subject':subject}).content

    # 用AI生成正文
    script_chain = script_template | model
    script = script_chain.invoke({'title':title,
                                  'duration':video_length,
                                  'wikipedia_search':search_result}).content

    # 返回AI生成的结果和维基百科的搜索结果
    return title,script,search_result

# # 进行一个简单的测试
# print(generate_script(subject='ChatGPT',
#                       video_length=1,
#                       api_key = '请输入自己的API',
#                       base_url='请输入API的接口网址',
#                       creativity=0.7))

"""测试通过，AI后端已搭建完成"""