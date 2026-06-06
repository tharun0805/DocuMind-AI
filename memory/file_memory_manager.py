from memory.session_memory import SessionMemory
from loguru import logger


class FileMemoryManager:
    def __init__(self):
        self.memories: dict[str, SessionMemory] = {}
        logger.debug("File memory manager initialized")

    def get_memory(self, file_name: str) -> SessionMemory:
        if file_name not in self.memories:
            self.memories[file_name] = SessionMemory()
            logger.debug(f"Created memory for: {file_name}")
        return self.memories[file_name]

    def clear_memory(self, file_name: str) -> None:
        if file_name in self.memories:
            self.memories[file_name].clear()

    def clear_all(self) -> None:
        self.memories = {}
        logger.info("All file memories cleared")

    def has_memory(self, file_name: str) -> bool:
        return (
            file_name in self.memories
            and not self.memories[file_name].is_empty()
        )