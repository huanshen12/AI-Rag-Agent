import streamlit as st
from agent.react_agent import react_agent
import time

st.title("智能扫地机器人客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state.agent = react_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("请输入您的问题")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    response = []
    def generate(generator,file):
        for chunk in generator:
            file.append(chunk)
            for a in chunk:
                time.sleep(0.01)
                yield a
    with st.spinner("思考中..."):
        res  = st.session_state.agent.excute_stream(prompt)
        st.chat_message("assistant").write_stream(generate(res,response))
    st.session_state.messages.append({"role":"assistant","content":response[-1]})
    st.rerun()
    

