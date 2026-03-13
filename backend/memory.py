conversation_memory = []

def add_to_memory(user, bot):
    conversation_memory.append({
        "user": user,
        "bot": bot
    })

def get_memory():

    history = ""

    for chat in conversation_memory[-3:]:  # last 3 messages
        history += f"User: {chat['user']}\n"
        history += f"Advisor: {chat['bot']}\n"

    return history