import streamlit as st
from agent.react_agent import react_agent
import time

st.title("智能扫地机器人客服")
st.divider()

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_001"
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
    def generate(generator):
        for chunk in generator:
            for a in chunk:
                time.sleep(0.01)
                yield a
    with st.spinner("思考中..."):
        res  = st.session_state.agent.excute_stream(prompt,st.session_state.session_id)
        result = st.chat_message("assistant").write_stream(generate(res))
    st.session_state.messages.append({"role":"assistant","content":result})
    st.rerun()
    

