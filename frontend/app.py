import streamlit as st
import requests

st.title("Psychsviser - Psychology Advisor")

user_input = st.text_input("Describe your situation")

if st.button("Ask Psychsviser"):

    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"user_input": user_input}
            )

            data = response.json()

            st.subheader("Psychsviser Advice")
            st.write(data.get("response", "No response received"))

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")