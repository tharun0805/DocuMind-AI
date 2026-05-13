from loguru import logger


def plan_route(intent: str) -> str:
    logger.info(f"Planning route for intent: {intent}")

    if intent == "computational":
        route = "dataframe"
    else:
        route = "retrieval"

    logger.info(f"Route decided: {route}")
    return route