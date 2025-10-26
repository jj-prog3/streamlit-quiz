import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

# --- 페이지 설정 ---
st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 QuizGPT: AI 퀴즈 생성기")

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input(
        "OpenAI API Key를 입력하세요",
        type="password",
        help="API 키는 OpenAI 웹사이트에서 발급받을 수 있습니다."
    )

    st.markdown("---")
    st.markdown(
        "❤️ [GitHub Repository](https://github.com/jj-prog3/streamlit-quiz)"
    )

if not api_key:
    st.info("먼저 OpenAI API Key를 입력하여 시작하세요.")
    st.stop()

# --- 2. 함수 호출을 위한 Pydantic 모델 정의 ---
# LLM이 생성할 JSON 출력의 구조를 명확하게 정의합니다.

class Answer(BaseModel):
    """퀴즈 질문에 대한 답변 선택지를 나타내는 모델"""
    answer_text: str = Field(description="선택지의 텍스트 내용")
    is_correct: bool = Field(description="이 선택지가 정답인지 여부")

class Question(BaseModel):
    """하나의 퀴즈 질문을 나타내는 모델"""
    question_text: str = Field(description="사용자에게 보여질 질문의 텍스트")
    answers: List[Answer] = Field(description="질문에 대한 4개의 선택지 목록")

class Quiz(BaseModel):
    """전체 퀴즈 데이터 구조를 나타내는 모델"""
    questions: List[Question] = Field(description="생성된 퀴즈 질문들의 목록")

# --- 3. 세션 상태 초기화 ---
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "score" not in st.session_state:
    st.session_state.score = 0

# --- 4. 문서 처리 및 퀴즈 생성 함수 ---
# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 파일 분할 및 로드 함수 (캐싱하여 중복 작업 방지)
@st.cache_data(show_spinner="파일을 처리 중입니다...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
    loader = UnstructuredFileLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)

# 위키피디아 검색 함수 (캐싱)
@st.cache_data(show_spinner="Wikipedia를 검색 중입니다...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(top_k_results=3, lang="ko")
    return retriever.get_relevant_documents(topic)

# 퀴즈 생성 함수 (캐싱)
@st.cache_data(show_spinner="퀴즈를 생성 중입니다...")
def generate_quiz(_docs, topic, num_questions, difficulty):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 주어진 컨텍스트를 기반으로 퀴즈를 만드는 선생님입니다.
        사용자가 지정한 난이도와 문제 수에 맞춰 퀴즈를 생성해야 합니다.
        각 질문에는 반드시 4개의 선택지가 있어야 하며, 그 중 하나만 정답이어야 합니다.
        """),
        ("human", """
        아래 컨텍스트를 바탕으로 '{num_questions}'개의 문제를 '{difficulty}' 난이도로 만들어 주세요.
        
        컨텍스트:
        {context}
        """)
    ])
    
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o", openai_api_key=api_key)
    # Pydantic 모델을 사용하여 LLM의 출력을 JSON으로 강제
    structured_llm = llm.with_structured_output(Quiz)
    chain = {"context": format_docs} | prompt | structured_llm
    
    return chain.invoke({"context": _docs, "num_questions": num_questions, "difficulty": difficulty})

# --- 5. UI ---
# 퀴즈 생성 UI
with st.container(border=True):
    st.subheader("어떤 주제로 퀴즈를 만들어 볼까요?")
    
    # 퀴즈 소스 선택 (파일 또는 위키피디아)
    source_type = st.radio("퀴즈 소스 선택:", ("파일 업로드", "Wikipedia 검색"), horizontal=True)
    
    docs = None
    topic_name = ""
    
    if source_type == "파일 업로드":
        uploaded_file = st.file_uploader("docx, txt, pdf 파일을 업로드하세요.", type=["pdf", "txt", "docx"])
        if uploaded_file:
            docs = split_file(uploaded_file)
            topic_name = uploaded_file.name
    else:
        wiki_topic = st.text_input("Wikipedia에서 검색할 주제를 입력하세요.")
        if wiki_topic:
            docs = search_wikipedia(wiki_topic)
            topic_name = wiki_topic
            
    # 퀴즈 설정 (문제 수, 난이도)
    num_questions = st.slider("문제 수", min_value=3, max_value=10, value=5)
    difficulty = st.radio("난이도", ("쉬움", "어려움"), horizontal=True)

    # 퀴즈 생성 버튼
    if st.button("퀴즈 생성하기", type="primary", use_container_width=True, disabled=not docs):
        # 상태 초기화
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.score = 0
        st.session_state.quiz_data = generate_quiz(docs, topic_name, num_questions, difficulty)

# 퀴즈가 생성되었을 때만 퀴즈 풀이 UI 표시
if st.session_state.quiz_data:
    st.markdown("---")
    st.header("📝 퀴즈를 풀어보세요")

    # 퀴즈 제출 폼
    with st.form("quiz_form"):
        for i, question in enumerate(st.session_state.quiz_data.questions):
            st.subheader(f"문제 {i+1}: {question.question_text}")
            
            # 라디오 버튼으로 선택지 표시
            options = [ans.answer_text for ans in question.answers]
            user_answer = st.radio(
                "답을 선택하세요:", options, key=f"q_{i}", 
                index=None # 재응시 시 선택 초기화를 위해
            )
            st.session_state.user_answers[i] = user_answer
            st.markdown("---")
            
        submitted = st.form_submit_button("결과 확인하기")

        if submitted:
            st.session_state.quiz_submitted = True
            
            # 채점 로직
            score = 0
            for i, question_data in enumerate(st.session_state.quiz_data.questions):
                correct_answer = next(ans.answer_text for ans in question_data.answers if ans.is_correct)
                if st.session_state.user_answers.get(i) == correct_answer:
                    score += 1
            st.session_state.score = (score / len(st.session_state.quiz_data.questions)) * 100

# 채점 결과 표시
if st.session_state.quiz_submitted:
    st.header(f"📈 채점 결과: {st.session_state.score:.0f}점")

    if st.session_state.score == 100:
        st.success("🎉 축하합니다! 만점입니다!")
        st.balloons()
    else:
        st.warning(f"아쉽네요. 만점까지 {100 - st.session_state.score:.0f}점 남았습니다.")
        if st.button("다시 풀어보기"):
            # 재응시를 위해 상태 초기화 후 새로고침
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.rerun()

    # 정답 및 해설 보기
    with st.expander("🔍 정답 및 해설 보기", expanded=True):
        for i, question_data in enumerate(st.session_state.quiz_data.questions):
            st.subheader(f"문제 {i+1}: {question_data.question_text}")
            
            user_ans = st.session_state.user_answers.get(i, "답변 안 함")
            correct_ans = next(ans.answer_text for ans in question_data.answers if ans.is_correct)

            if user_ans == correct_ans:
                st.markdown(f"**- 당신의 답:** <span style='color:green;'>{user_ans} (정답)</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**- 당신의 답:** <span style='color:red;'>{user_ans} (오답)</span>", unsafe_allow_html=True)
                st.markdown(f"**- 정답:** <span style='color:green;'>{correct_ans}</span>", unsafe_allow_html=True)
            st.markdown("---")