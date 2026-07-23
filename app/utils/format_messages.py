from langchain_core.messages import HumanMessage, AIMessage


def format_messages(messages, thread_id):
    formatted = []

    for msg in messages:

        if isinstance(msg, HumanMessage):
            formatted.append(
                {
                    "thread_id": thread_id,
                    "role": "user",
                    "content": msg.content,
                }
            )

        elif isinstance(msg, AIMessage):

            if not msg.content:
                continue

            formatted.append(
                {
                    "thread_id": thread_id,
                    "role": "assistant",
                    "content": msg.content,
                }
            )

    return formatted
