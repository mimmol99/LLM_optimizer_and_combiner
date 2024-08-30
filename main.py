import os
import time
from dotenv import load_dotenv
from models import OpenAIModel, ClaudeModel, GoogleModel
from pathlib import Path

load_dotenv(Path("./api_key.env"))

os.getenv("OPENAI_API_KEY")

def main():
    
    input_file_path = "./input.txt"
    opt_file_path = "./opt.txt"
    claude_file_path = "./answer_claude.txt"
    google_file_path = "./answer_google.txt"
    answer_file_path = "./answer.txt"
    
    # Initialize models
    os.getenv("OPENAI_API_KEY")

    openai_model = OpenAIModel(model_name='gpt-4o-mini')
    claude_model = ClaudeModel(model_name='claude-3-sonnet-20240229')
    google_model = GoogleModel(model_name='gemini-1.5-flash')

    # Get the LLM instances
    llm_openai = openai_model.get_model()
    llm_claude = claude_model.get_model()
    llm_google = google_model.get_model()

    # Sample prompt
    with open(input_file_path,"r") as f:
        original_prompt = "".join(f.readlines())
    
    # OpenAI generates two optimized versions of the prompt
    optimized_prompt_1 = llm_openai.invoke(f"Create an optimized,more detailed version of this prompt: --- {original_prompt} --- ").content
    optimized_prompt_2 = llm_openai.invoke(f"Create an optimized,more detailed version of this prompt: --- {original_prompt} --- ").content
    
    
    with open(opt_file_path,"w") as f:
        f.write(optimized_prompt_1+ "\n *"*10+"\n" +optimized_prompt_2)

    
    
    # Claude and Gemini models process the optimized prompts
    answer_claude = llm_claude.invoke(optimized_prompt_1).content
    answer_google = llm_google.invoke(optimized_prompt_2).content

    with open(claude_file_path,"w") as f:
        f.write(answer_claude)
    with open(google_file_path,"w") as f:
        f.write(answer_google)

    
    # OpenAI processes the two answers to generate a final answer
    final_prompt = f"Given the following responses from two different AI models, combine them into a complete answer:\n1. Claude's answer: {answer_claude}\n2. Google's answer: {answer_google}"
    final_answer = llm_openai.invoke(final_prompt).content

    with open(answer_file_path,"w") as f:
        f.write(final_answer)


    print(final_answer)



if __name__ == "__main__":
    main()

