import asyncio

from openai import OpenAI, OpenAIError
from config import settings

MODEL = "llama-3.3-70b-versatile"
client = None


def get_client():
    global client
    if client is not None:
        return client
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    try:
        client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
    except OpenAIError as e:
        raise RuntimeError(f"Could not initialize AI client: {e}")
    return client


async def call_llm(prompt: str) -> str:
    def _call():
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    return await asyncio.to_thread(_call)
async def generate_pandas_code(question: str, columns: list[str]):

    prompt = f"""
You are the AI engine of DataMind AI.

You convert natural language into executable pandas code.

Dataset columns:
{columns}

User Question:
{question}

IMPORTANT RULES

1. Return ONLY executable pandas code.
2. Variable name is df.
3. Do NOT use markdown.
4. Do NOT explain anything.
5. Never return ```python.
6. Never import libraries.
7. Return either:
   - a dataframe
   - a pandas series
   - a scalar value
8. Always use existing column names only.
9. Use case-insensitive matching whenever possible.

Examples

Question:
Who has the highest salary?

Answer:
df.nlargest(1, "salary")[["name","department","salary"]]

----------------------------

Question:
Top 5 employees by salary

Answer:
df.nlargest(5,"salary")[["name","salary"]]

----------------------------

Question:
Average salary

Answer:
df["salary"].mean()

----------------------------

Question:
Department with highest average salary

Answer:
df.groupby("department")["salary"].mean().sort_values(ascending=False)

----------------------------

Question:
How many employees are in each department?

Answer:
df.groupby("department").size().reset_index(name="Employees")

----------------------------

Question:
Employees with more than 5 years experience

Answer:
df[df["experience"]>5]

----------------------------

Question:
IT employees

Answer:
df[df["department"].str.lower()=="it"]

----------------------------

Question:
Highest production in last 5 years

Answer:
df[df["year"]>=df["year"].max()-4] \
.groupby("product")["production"] \
.sum() \
.sort_values(ascending=False)

----------------------------

Question:
Revenue by state

Answer:
df.groupby("state")["revenue"].sum().sort_values(ascending=False)

----------------------------

Question:
Monthly sales trend

Answer:
df.groupby("month")["sales"].sum()

Return ONLY pandas code.
"""

    return (await call_llm(prompt)).strip()