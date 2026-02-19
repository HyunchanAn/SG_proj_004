import streamlit as st

class User:
    def __init__(self, email):
        self.email = email

class AuthManager:
    def __init__(self):
        # Mock user database
        self.users = {
            "test@sg.com": "password"
        }

    def login(self, email, password):
        if email in self.users and self.users[email] == password:
            return User(email)
        return None

    def logout(self):
        if "user" in st.session_state:
            del st.session_state["user"]

def render_login_ui(auth_manager):
    st.title("Login")
    with st.form("login_form"):
        email = st.text_input("Email", value="test@sg.com")
        password = st.text_input("Password", type="password", value="password")
        submit = st.form_submit_button("Log In")

        if submit:
            user = auth_manager.login(email, password)
            if user:
                st.session_state["user"] = user
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid email or password")
