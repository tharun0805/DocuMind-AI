import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.config import get_google_api_key
from tools.dataframe_tool import load_dataframe, get_dataframe_info
from loguru import logger


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0
    )


def run_dataframe_agent(question: str, file_path: str) -> str:
    logger.info(f"Running DataFrame agent for: {question}")

    df = load_dataframe(file_path)
    df_info = get_dataframe_info(df)

    prompt = PromptTemplate(
        input_variables=["question", "df_info"],
        template="""
        You are a data analysis expert.
        You have access to a DataFrame with the following structure:

        {df_info}

        Write a single line of Python pandas code to answer this question:
        {question}

        Rules:
        - The DataFrame variable is called 'df'
        - Write only the code, nothing else
        - No explanations, no markdown, no backticks
        - The code must return a value or print a result
        - Example: df['column'].sum() or df[df['col'] > 100]['name'].tolist()
        """
    )

    for attempt in range(3):
        try:
            chain = prompt | get_llm()
            result = chain.invoke({
                "question": question,
                "df_info": df_info
            })

            code = result.content.strip()
            logger.info(f"Generated pandas code: {code}")

            answer = eval(code, {"df": df, "pd": pd})
            logger.info("DataFrame computation successful")
            return f"Result: {answer}"

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"DataFrame agent error: {str(e)}")
                return f"I could not compute the answer. Error: {str(e)}"

    return "I could not compute the answer at this time. Please try again."