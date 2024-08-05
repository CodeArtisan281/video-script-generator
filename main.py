import streamlit as st
from utils import generate_script

# 给网站写一个标题
st.header('🎬 短视频脚本生成器')
st.write('##### ps：🫎🏄‍♂️🐠使用要交版权费😏')

# 创建一个侧边栏
with st.sidebar:
    # 侧边栏中要让用户输入自己的api秘钥，并才用密码的形式展示
    api_key = st.text_input('请输入OpenAI秘钥：',type='password')
    # 侧边栏中要让用户输入api接口网址
    base_url = st.text_input('请输入您的API_Key的接口网址')
    # 给用户提供一个获取OpenAI官方的API官方网址
    st.markdown('[获取OpenAI API官方秘钥（需要科学上网）](https://platform.openai.com/account/api-keys)')
    # 给用户提供一个获取中转API网址，使国内用户也能够通过中转API使用到GPT
    st.markdown('[获取中转API秘钥（国内可用，推荐这个）](https://api.aigc369.com/)')

# 创建一个视频主题的输入框
subject = st.text_input('💡 请输入视频的主题')

# 创建一个视频时长的数字输入框
video_length = st.number_input('⏱️ 请输入视频的大致时长（单位：分钟）',min_value=0.1,step=0.1)

# 创建一个创造性的参数滑块
creativity = st.slider('✨ 请输入视频脚本的创造力（数字小说明更严谨，数字大说明更多样)',
                       min_value=0.0,
                       max_value=1.0,
                       step=0.1,
                       value=0.7)

# 创建一个生成脚本的按钮
submit = st.button('生成脚本')

# 用户按按钮前若没输入API秘钥提示用户输入API秘钥才能生成脚本
if submit and not api_key:
    st.info('请输入您的OpenAI API秘钥')    # 提示用户输入API秘钥
    st.stop()   # 用stop函数网页将不再运行下面的代码

# 用户按按钮前若没输入API接口网址提示用户输入API接口网址才能生成脚本
if submit and not base_url:
    st.info('请输入您的OpenAI API接口网址')  # 提示用户输入API接口网址
    st.stop()   # 用stop函数网页将不再运行下面的代码

# 用户按按钮前若没输入视频主题提示用户输入视频主题才能生成脚本
if submit and not subject:
    st.info('请输入视频的主题')     # 提示用户输入视频主题
    st.stop()   # 用stop函数网页将不再运行下面的代码

# 最后当用户全都输入好按下按钮时
if submit:
    # 创建一个加载中的小提示
    with st.spinner('AI正在思考中，可以先去敲敲🫎🏄‍♂️🐠的脑瓜子🤪'):
        # 将加载中较慢的代码放进来
        title,script,search_result = generate_script(subject,video_length,api_key,base_url,creativity)

    # 若代码生成成功后给一个小提示
    st.success('视频脚本已生成！')

    # 放入生成第一列内容的标题
    st.subheader('🔥 标题：')

    # 将AI生成的标题展示上去
    st.write(title)

    # 放入生成第二列内容的标题
    st.subheader('📝 视频脚本：')

    # 将AI生成的标题展示上去
    st.write(script)

    # 放入一个搜索隐藏下拉框
    with st.expander('维基百科搜索结果 👀'):
        st.info(search_result)  # 展示维基百科的搜索结果