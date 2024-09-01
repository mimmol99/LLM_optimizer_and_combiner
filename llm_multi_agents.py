from concurrent.futures import ThreadPoolExecutor
from typing import List
from llm_model import LLM_Model

class LLM_MultiAgents():

    def __init__(self, models: List[LLM_Model]):
        self.models = self.filter_models(models)

    def filter_models(self, models: LLM_Model | List[LLM_Model]) -> List[LLM_Model]:
        return [model for model in (models if isinstance(models, list) else [models]) if isinstance(model, LLM_Model) or print(f"Not valid LLM_Model: {model}")]

    def set_role(self, models: LLM_Model | List[LLM_Model], instruction: str):
        for model in self.filter_models(models):
            model.set_system_instruction(instruction)

    def call(self, models: LLM_Model | List[LLM_Model], prompt: str) -> List[str]:
        models = self.filter_models(models)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda model: model.generate_answer(prompt), models))
        return results
    
    def call_parallell(self, models: List[LLM_Model], prompts: List[str]) -> List[str]:
        if len(models) != len(prompts):
            raise ValueError("The number of models and prompts must be the same.")
        models = self.filter_models(models)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda pair: pair[0].generate_answer(pair[1]), zip(models, prompts)))
        return results

    def generate_different_version(self, model: LLM_Model, prompt: str, N_version=2) -> List[str]:
        model = self.filter_models(model)[0]
        return [model.generate_answer(f"Generate a different and more complete version of this prompt: Prompt: ''' {prompt} '''. Your New Prompt:") for _ in range(N_version)]

    def combine_answer(self, model: LLM_Model, answers: List[str]) -> str:
        model = self.filter_models(model)[0]
        prompt = f"Combine these different answers from different LLMs to obtain a final optimized and combined answer: 'Answer:'{'Answer:'.join(answers)} Your final combined Answer:"
        return model.generate_answer(prompt)
    
    def save_chat(self, path=None):
        for model in self.models:
            model.save_messages(path)
