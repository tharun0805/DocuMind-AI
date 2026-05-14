from loguru import logger


class SessionMemory:
    def __init__(self):
        self.history = []
        logger.info("Session memory initialized")

    def add_human_message(self, message: str) -> None:
        self.history.append({
            "role": "human",
            "content": message
        })
        logger.info(f"Human message added to memory: {message[:50]}")

    def add_ai_message(self, message: str) -> None:
        self.history.append({
            "role": "assistant",
            "content": message
        })
        logger.info(f"AI message added to memory: {message[:50]}")

    def get_history(self) -> list[dict]:
        return self.history

    def get_history_as_text(self) -> str:
        if not self.history:
            return ""

        history_text = ""
        for message in self.history:
            if message["role"] == "human":
                history_text += f"Human: {message['content']}\n"
            else:
                history_text += f"Assistant: {message['content']}\n"

        return history_text.strip()

    def clear(self) -> None:
        self.history = []
        logger.info("Session memory cleared")

    def is_empty(self) -> bool:
        return len(self.history) == 0