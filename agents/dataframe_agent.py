import time
import pandas as pd
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from tools.dataframe_tool import load_dataframe, get_dataframe_info
from loguru import logger


def run_dataframe_agent(question: str, file_path: str) -> str:
    logger.info(f"Running DataFrame agent for: {question}")

    df = load_dataframe(file_path)
    df_info = get_dataframe_info(df)

    prompt = PromptTemplate(
        input_variables=["question", "df_info"],
        template="""
        You are a data analysis expert.
        DataFrame structure:
        {df_info}

        Write a single line of Python pandas code to answer:
        {question}

        Rules:
        - DataFrame variable is called df
        - Write only the code nothing else
        - No explanations no markdown no backticks
        """
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0)
            chain = prompt | llm
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
                time.sleep(60 * (attempt + 1))
            else:
                logger.error(f"DataFrame error: {e}")
                return f"Could not compute. Error: {str(e)}"

    return "Could not compute. Please try again."