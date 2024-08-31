from litellm import completion

class LLM_Model():

    def __init__(self, model: str):
        self.model = model
        self.messages = []

    def set_system_instruction(self, instruction: str):
        self.messages = [msg for msg in self.messages if msg["role"] != "system"]
        self.messages.append({"role": "system", "content": instruction})

    def update_messages(self, content: str, role: str):
        self.messages.append({"content": content, "role": role})

    def generate_answer(self, prompt: str) -> str:
        self.update_messages(prompt, "user")
        response = completion(model=self.model, messages=self.messages)
        text_answer = response.choices[0].message.content
        self.update_messages(text_answer, "system")
        return text_answer
    
    def get_messages(self):
        return self.messages

