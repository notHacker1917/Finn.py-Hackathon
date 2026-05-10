from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("STACKIT_MODEL_SERVING_MODEL"),
    base_url=os.getenv("STACKIT_MODEL_SERVING_BASE_URL"),
    api_key=os.getenv("STACKIT_MODEL_SERVING_AUTH_TOKEN"),
    max_retries=3,
    temperature=0.7,
)

resp = llm.invoke("Hallo!")
print(resp.content)   # print the assistant’s text
