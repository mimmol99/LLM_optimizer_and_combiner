from litellm import completion

class LLM_Model():

    def __init__(self, model: str):
        self.model = model
        self.messages = []

    def set_system_instruction(self, instruction: str):
        self.messages = [{"role": "system", "content": instruction} if msg["role"] == "system" else msg for msg in self.messages]

    def update_messages(self, content: str, role: str):
        self.messages.append({"content": content, "role": role})

    def generate_answer(self, prompt: str) -> str:
        self.update_messages(prompt, "user")
        text_answer = completion(model=self.model, messages=self.messages).choices[0].message.content
        self.update_messages(text_answer, "system")
        return text_answer
    
    def get_messages(self):
        return self.messages
    
    def save_messages(self, path=None):
        with open(path or f"{self.model.replace('/','_')}_messages.txt", "w") as f:
            f.writelines(f"ROLE:{msg['role']}\n CONTENT:{msg['content']}\n" for msg in self.messages)


