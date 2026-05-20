import time
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from loguru import logger


def query_multiple_documents(
    question: str,
    documents: list[dict]
) -> str:
    logger.info(f"Querying {len(documents)} documents simultaneously")

    doc_contexts = ""
    for i, doc in enumerate(documents, 1):
        doc_contexts += f"\n{'='*50}\n"
        doc_contexts += f"DOCUMENT {i}: {doc['name']}\n"
        doc_contexts += f"{'='*50}\n"
        doc_contexts += doc["text"][:2500]
        doc_contexts += "\n"

    prompt = PromptTemplate(
        input_variables=["question", "doc_contexts", "num_docs"],
        template="""
You are DocuMind AI analyzing {num_docs} documents simultaneously.

{doc_contexts}

Question: {question}

Instructions:
- Analyze ALL documents carefully
- Compare and contrast information across documents when relevant
- Clearly cite which document (Document 1, Document 2, etc.) contains each piece of information
- Synthesize insights that span multiple documents
- If asking for comparison, explicitly compare each document
- Give a comprehensive answer covering all relevant documents
- End with: "Key Takeaway: [one clear cross-document insight]"

Answer:"""
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.3)
            chain = prompt | llm
            result = chain.invoke({
                "question": question,
                "doc_contexts": doc_contexts,
                "num_docs": len(documents)
            })
            logger.info("Multi-document query complete")
            return result.content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                raise e

    return "Could not query documents. Please try again."


def compare_documents(documents: list[dict]) -> str:
    logger.info(f"Comparing {len(documents)} documents")

    if len(documents) < 2:
        return "Need at least 2 documents to compare."

    doc_summaries = ""
    for i, doc in enumerate(documents, 1):
        doc_summaries += f"\nDocument {i} - {doc['name']}:\n{doc['text'][:1500]}\n"

    prompt = PromptTemplate(
        input_variables=["doc_summaries", "num_docs"],
        template="""
You are comparing {num_docs} documents. Provide a detailed comparison.

{doc_summaries}

Create a comprehensive comparison covering:
1. Main topics and themes of each document
2. Key similarities between documents
3. Key differences between documents
4. Unique insights from each document
5. Which document is most relevant for what purpose

Format as a clear structured analysis.
End with: "Summary: [one paragraph synthesizing all documents]"

Comparison:"""
    )

    for attempt in range(3):
        try:
            llm = get_shared_llm(temperature=0.3)
            chain = prompt | llm
            result = chain.invoke({
                "doc_summaries": doc_summaries,
                "num_docs": len(documents)
            })
            return result.content.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                raise e

    return "Could not compare documents. Please try again."
