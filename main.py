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

# Retrieve API keys

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Define file paths
INPUT_FILE_PATH = "./input.txt"
OPT_FILE_PATH = "./opt.txt"
CLAUDE_FILE_PATH = "./answer_claude.txt"
GOOGLE_FILE_PATH = "./answer_google.txt"
ANSWER_FILE_PATH = "./answer.txt"
COMPARISON_FILE_PATH = "./comparison.txt"

    
def read_input_file(file_path):
    """Read the input prompt from a file."""
    with open(file_path, "r") as f:
        return "".join(f.readlines())

def write_to_file(file_path, content):
    """Write content to a specified file."""
    with open(file_path, "w") as f:
        f.write(content)

def generate_optimized_prompts(model, original_prompt):
    """Generate two optimized versions of the original prompt using OpenAI."""
    optimized_prompt_1 = model.generate_answer(f"Create an optimized, more detailed version of this prompt: --- {original_prompt} --- ").content
    optimized_prompt_2 = model.generate_answer(f"Create an other optimized, more detailed version of this prompt: --- {original_prompt} --- ").content
    return optimized_prompt_1, optimized_prompt_2


def main():
    """Main function to run the AI model interactions."""
    openai_model_name = "gpt-4o-mini"
    claude_model_name = "claude-3-5-sonnet-20240620"
    google_model_name = "gemini/gemini-1.5-flash"

    openai_model = LLM_Model(openai_model_name)
    claude_model = LLM_Model(claude_model_name)
    google_model =  LLM_Model(google_model_name)

    llm_multi_agents = LLM_MultiAgents([openai_model,claude_model_name,google_model])

    while True:
        # Read the original prompt from the input file
        #original_prompt = read_input_file(INPUT_FILE_PATH)

        original_prompt = input("ask me something..")

        original_answer = llm_multi_agents.generate_answer(openai_model,original_prompt)[0]

        # Generate optimized prompts
        optimized_prompts = llm_multi_agents.generate_different_version(openai_model, original_prompt)

        # Write optimized prompts to file
        write_to_file(OPT_FILE_PATH, "*****".join(optimized_prompts))

        # Process answers from Claude and Google models
        answer_claude = llm_multi_agents.generate_answer(claude_model,optimized_prompts[0])[0]
        answer_google = llm_multi_agents.generate_answer(google_model,optimized_prompts[1])[0]

        # Write answers to respective files
        write_to_file(CLAUDE_FILE_PATH, answer_claude)
        write_to_file(GOOGLE_FILE_PATH, answer_google)

        # Generate the final answer
        final_answer = llm_multi_agents.combine_answer(openai_model, [answer_claude, answer_google])

        # Write the final answer to a file
        write_to_file(ANSWER_FILE_PATH, final_answer)

        # Print the final answer
        write_to_file(COMPARISON_FILE_PATH, "\n*****\n".join([original_answer, final_answer]))

        # Wait for a specified time before the next iteration (optional)
        time.sleep(1)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()