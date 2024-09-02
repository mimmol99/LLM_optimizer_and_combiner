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

    def call_model(self, model: LLM_Model, prompt: str, apply_refine: bool = False, apply_cot: bool = False) -> str:
        answer = model.generate_answer(prompt)
        if apply_refine:
            answer = model.self_refine(answer)
        if apply_cot:
            answer = model.chain_of_thought(answer)
        return answer
    
    def call_models(self, models: LLM_Model | List[LLM_Model], prompt: str, apply_refine: bool = False, apply_cot: bool = False) -> List[str]:
        models = self.filter_models(models)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda model: self.call_model(model,prompt,apply_refine = apply_refine,apply_cot = apply_cot), models))
        return results
    
    def call_prompts_in_parallell(self, models: List[LLM_Model], prompts: List[str], apply_refine: bool = False, apply_cot: bool = False) -> List[str]:
        if len(models) != len(prompts):
            raise ValueError("The number of models and prompts must be the same.")
        models = self.filter_models(models)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda pair: self.call_model(model = pair[0],prompt = pair[1],apply_refine = apply_refine,apply_cot = apply_cot), zip(models, prompts)))
        return results

    def generate_different_version(self, model: LLM_Model, prompt: str, N_version=2,apply_refine: bool = False, apply_cot: bool = False) -> List[str]:
        model = self.filter_models(model)[0]
        return [self.call_model(model,f"Generate a different and more complete version of this prompt,if there is code inside,keep the whole code in the new prompt: Prompt to be improved: ''' {prompt} '''. Your New Improved Prompt:",apply_refine,apply_cot) for _ in range(N_version)]

    def combine_answer(self, model: LLM_Model, answers: List[str],apply_refine: bool = False, apply_cot: bool = False) -> str:
        model = self.filter_models(model)[0]
        prompt = f"Combine these different answers from different LLMs to obtain a final optimized and combined answer,if there is code in one or more answer,combine the code to obtain the best code possible: 'Answer:'{'Answer:'.join(answers)} Your final combined Answer:"
        return self.call_model(model,prompt,apply_refine = apply_refine,apply_cot = apply_cot)
    
    def save_chat(self, path=None):
        for model in self.models:
            model.save_messages(path)
