from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, OPENROUTER_API_KEY

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=0,
    api_key=OPENROUTER_API_KEY,
)
