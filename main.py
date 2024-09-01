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

    llm_multi_agents = LLM_MultiAgents([openai_model,claude_model,google_model])

    while True:
        # Read the original prompt from the input file
        original_prompt = input("ask me something..")
        original_answer = llm_multi_agents.generate_answer(openai_model,original_prompt)[0]
        # Generate optimized prompts
        optimized_prompts = llm_multi_agents.generate_different_version(openai_model, original_prompt)

        # Process answers from Claude and Google models
        answer_claude = llm_multi_agents.generate_answer(claude_model,optimized_prompts[0])[0]
        answer_google = llm_multi_agents.generate_answer(google_model,optimized_prompts[1])[0]

        # Generate the final answer
        final_answer = llm_multi_agents.combine_answer(openai_model, [answer_claude, answer_google])

        #save chat
        llm_multi_agents.save_chat()


if __name__ == "__main__":
    main()
