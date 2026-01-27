import streamlit as st
from supabase import create_client, Client
import time

class AuthManager:
    def __init__(self):
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Supabase configuration error: {e}")
            self.supabase = None

    def login(self, email, password):
        try:
            response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return response.user
        except Exception as e:
            st.error(f"Login failed: {e}")
            return None

    def signup(self, email, password):
        try:
            response = self.supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            return response.user
        except Exception as e:
            st.error(f"Signup failed: {e}")
            return None

    def logout(self):
        try:
            self.supabase.auth.sign_out()
        except Exception as e:
            pass # Ignore logout errors

def render_login_ui(auth_manager):
    st.title("SG-RADAR Login")
    
    check_secrets()

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if not email or not password:
                    st.warning("Please enter both email and password")
                else:
                    user = auth_manager.login(email, password)
                    if user:
                        st.session_state["user"] = user
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()

    with tab2:
        with st.form("signup_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            signup_submit = st.form_submit_button("Sign Up")
            
            if signup_submit:
                 if not new_email or not new_password:
                    st.warning("Please enter both email and password")
                 else:
                    user = auth_manager.signup(new_email, new_password)
                    if user:
                        st.success("Signup successful! Please check your email to confirm.")

def check_secrets():
    if "supabase" not in st.secrets:
        st.error("Missing [supabase] section in .streamlit/secrets.toml")
        st.stop()
    if st.secrets["supabase"]["url"] == "INSERT_YOUR_SUPABASE_URL_HERE":
        st.warning("Please configure your Supabase URL and Key in .streamlit/secrets.toml")
        st.stop()
