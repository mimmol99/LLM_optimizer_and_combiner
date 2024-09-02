import os
import time
from dotenv import load_dotenv
import litellm
from litellm import completion
from llm_model import LLM_Model
from llm_multi_agents import LLM_MultiAgents
from pathlib import Path

load_dotenv(Path("./api_key.env"))

os.environ['LITELLM_LOG'] = 'DEBUG'
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def main():
    """Main function to run the AI model interactions."""
    openai_model_name = "gpt-4o-mini"
    claude_model_name = "claude-3-5-sonnet-20240620"
    google_model_name = "gemini/gemini-1.5-flash"

    openai_model = LLM_Model(openai_model_name)
    claude_model = LLM_Model(claude_model_name)
    google_model =  LLM_Model(google_model_name)
    
    models = [openai_model,claude_model,google_model]
    generator_model = openai_model
    answer_model = [google_model,claude_model]
    combiner_model = openai_model
    
    llm_multi_agents = LLM_MultiAgents(models)

    while True:
        # Read the original prompt from the input
        original_prompt = input("ask me something..(or insert 'exit')\n")
        if original_prompt=="exit":
            break
        original_answer = llm_multi_agents.call(generator_model, original_prompt)[0]
        # Generate optimized prompts
        optimized_prompts = llm_multi_agents.generate_different_version(generator_model, original_prompt,N_version=1)
        # Process answers from Claude and Google models
        answers = llm_multi_agents.call(answer_model, optimized_prompts[0])
        # Generate the final answer
        final_answer = llm_multi_agents.combine_answer(combiner_model, answers)
        #save all models history locally
        llm_multi_agents.save_chat()
        print(f"Optimized_and_combined_answer:\n{final_answer}")
    print("All messages have been saved locally!")
        
        


if __name__ == "__main__":
    main()
