SYSTEM_PROMPT_SUMMARIZATION = """Please analyze and summarize the arxiv paper into an **Korean** AI newsletter. Feel free to include some technical keywords in English itself. You can write side-by-side the original English words in parenthesis if the words are not familiar or not frequently used in Korean. Please answer in JSON format where **keys are in English**. Consider the following format and components for the summary (but don't include all the keys if not applicable):
[
    {"What's New": "..."},
    {"Technical Details": "..."},
    {"Performance Highlights": "..."},
]
"""
