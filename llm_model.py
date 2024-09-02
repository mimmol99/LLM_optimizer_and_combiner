from litellm import completion

class LLM_Model():

    def __init__(self, model: str):
        self.model = model
        self.messages = []
        self.feedback_memory = []
        self.thought_chain = []   

    def set_system_instruction(self, instruction: str):
        """
        Set the system instruction for the chat model.
        
        Parameters:
            instruction (str): The instruction to give guidance to the model.
        """
        self.messages = [{"role": "system", "content": instruction} if msg["role"] == "system" else msg for msg in self.messages]

    def update_messages(self, content: str, role: str):
        """
        Append a new message to the chat history.
        
        Parameters:
            content (str): The content of the message.
            role (str): The role of the sender; should be 'user' or 'system'.
        """
        self.messages.append({"content": content, "role": role})

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer based on the input prompt.
        
        Parameters:
            prompt (str): The input text for which a response is generated.
        
        Returns:
            str: The generated response from the model.
        """
        self.update_messages(prompt, "user")
        text_answer = completion(model=self.model, messages=self.messages).choices[0].message.content
        self.update_messages(text_answer, "system")
        return text_answer
    
    def get_messages(self)->list:
        """
        Retrieve the chat history.
        
        Returns:
            list: A list of messages exchanged with the model.
        """
        return self.messages
    
    def save_messages(self, path=None):
        """
        Save the chat history to a text file.
        
        Parameters:
            path (str, optional): The file path where messages will be saved. 
                                  If not specified, a default path will be used.
        """
        with open(path or f"{self.model.replace('/','_')}_messages.txt", "w") as f:
            f.writelines(f"ROLE: {msg['role']}\n CONTENT: {msg['content']}\n" for msg in self.messages)

    def self_refine(self, prompt: str, N_refine: int = 1) -> str:
        """
        Parameters:
            prompt (str): The prompt to be refined.
        Returns:
            
        """
        if not isinstance(prompt, str):
            raise ValueError("Feedback must be a string.")
        answer = self.generate_answer(prompt)
        refined_prompt = f"Given this prompt: '{prompt}' and this answer: '{answer}' check the answer coeherence and correctness and then re-write it,re-write only the new improved answer:"
        for _ in range(N_refine):
            refined_answer = self.generate_answer(refined_prompt.format(prompt=prompt,answer=answer))
        return refined_answer

    def chain_of_thought(self, prompt: str)-> str:
        """
        Perform Chain-Of-Thoughts on prompt

        Parameters:
            prompt: The prompt to be processed step-by-step
        """
        cot_prompt = "Given this prompt: '{prompt}' write some steps to solve the request and perform them step-by-step:".format(prompt=prompt)
        return self.generate_answer(cot_prompt)



