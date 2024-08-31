from llm_model import LLM_Model
from typing import List

class LLM_MultiAgents():

    def __init__(self, models: List[LLM_Model]):
        self.models = models
        self.roles = {model:None for model in self.models}

    def set_role(self, models: LLM_Model | List[LLM_Model], role: str):

        if not isinstance(models,list):
            models = [models]

        for model in models:
            self.roles[model] = role

    def generate_answer(self, models: LLM_Model | List[LLM_Model], prompt: str) -> List[str]:

        if not isinstance(models,list):
            models = [models]

        answer_models = []

        for model in models:
            answer_models.append(model.generate_answer(prompt))      
        
        return answer_models
    
    def generate_different_version(self, model: LLM_Model, prompt: str, N_version = 2) -> list[str]:
        
        new_prompts = []

        for _ in range(N_version):
            new_prompts.append(model.generate_answer(f"Generate a different and more complete version of this prompt: Prompt: {prompt} Your New Prompt:"))

        return new_prompts

    def combine_answer(self, model: LLM_Model, answers: List[str]) -> str:

        prompt = f"Combine these different answer from different LLM to \
                  obtain a final optimized and combined answer: { 'Answer:'.join(answers)}\
                  Your final combined Answer:"
        
        return model.generate_answer(prompt)



    

        
