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
            user = response.user
            
            if user:
                # Check for 'approved' metadata
                # Default to False if key doesn't exist to be safe, or True if you want open by default 
                # (Requirements said: wait for approval, so strict default check)
                is_approved = user.user_metadata.get("approved", False)
                
                if not is_approved:
                    st.error("로그인 실패: 관리자 승인 대기 중입니다. 승인 후 이용 가능합니다.")
                    self.logout() # Force logout session cleanup
                    return None
                    
            return user
        except Exception as e:
            error_msg = str(e)
            if "invalid_grant" in error_msg or "Email not confirmed" in error_msg:
                 # Supabase often returns 'invalid_grant' for banned users or wrong passwords.
                 pass

            # Let's wrap the error display.
            if "Invalid login credentials" in error_msg:
                st.error("로그인 실패: 이메일 또는 비밀번호를 확인해주세요.")
            else:
                # If we want to force the specific message the user asked for:
                st.error("로그인 실패: 해당 계정은 사용 중지되었습니다. 세계화학공업으로 연락 부탁드립니다.")
            
            return None

    def signup(self, email, password):
        try:
            # Add metadata approved=False by default
            response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "approved": False
                    }
                }
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
                        st.success("로그인되었습니다.")
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
                        st.success("회원가입 요청이 완료되었습니다! 관리자 승인 후 로그인하실 수 있습니다.")

def check_secrets():
    if "supabase" not in st.secrets:
        st.error("Missing [supabase] section in .streamlit/secrets.toml")
        st.stop()
    if st.secrets["supabase"]["url"] == "INSERT_YOUR_SUPABASE_URL_HERE":
        st.warning("Please configure your Supabase URL and Key in .streamlit/secrets.toml")
        st.stop()
