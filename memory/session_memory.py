from loguru import logger


class SessionMemory:
    def __init__(self):
        self.history = []
        logger.debug("Session memory initialized")

    def add_human_message(self, message: str) -> None:
        self.history.append({"role": "human", "content": message})
        logger.debug(f"Memory: human msg ({len(message)} chars)")

    def add_ai_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})
        logger.debug(f"Memory: ai msg ({len(message)} chars)")

    def get_history(self) -> list[dict]:
        return self.history

    def get_history_as_text(self) -> str:
        if not self.history:
            return ""
        lines = []
        for msg in self.history[-10:]:
            role = "Human" if msg["role"] == "human" else "Assistant"
            content = msg["content"][:400]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history = []
        logger.info("Session memory cleared")

    def is_empty(self) -> bool:
        return len(self.history) == 0