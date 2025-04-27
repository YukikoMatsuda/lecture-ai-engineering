# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="日本語GPT-2チャットボット", layout="wide")

# --- 初期化処理 ---
metrics.initialize_nltk()
database.init_db()
data.ensure_initial_data()

# --- LLMモデルのロード（キャッシュを利用） ---
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        return None

pipe = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🤖 Japanese GPT-2 Chatbot with Feedback")
st.write("日本語GPT-2モデルを使用したチャットボットです。質問を入力して送信してください。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
if 'page' not in st.session_state:
    st.session_state.page = "チャット"

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page),
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector)
)

# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        st.subheader("質問を入力してください")
        user_input = st.text_area("質問を入力してください", key="input_text")
        if st.button("質問を送信"):
            with st.spinner("考え中..."):
                try:
                    output = pipe(user_input, max_length=100, do_sample=True)[0]["generated_text"]
                    st.success(f"回答: {output}")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッター ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")
