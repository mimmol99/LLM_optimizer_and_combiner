import os
import time
from dotenv import load_dotenv
import litellm
from litellm import completion
from llm_model import LLM_Model
from llm_multi_agents import LLM_MultiAgents
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import ttk
import PyPDF2
from docx import Document

load_dotenv(Path("./api_key.env"))
# Retrieve the API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class LLMApp:
    """An application that exposes a GUI for comparing different language model responses."""

    def __init__(self, root):
        """Initializes the GUI components and models used for querying the language models."""
        self.root = root
        root.title("LLM Comparison Tool")

        # Label for prompting user input
        self.prompt_label = tk.Label(root, text="Enter your question:")
        self.prompt_label.grid(row=0, column=0, padx=10, pady=10)

        # Create a frame for the prompt entry and its scrollbar
        self.prompt_frame = tk.Frame(root)
        self.prompt_frame.grid(row=0, column=1, padx=10, pady=10)

        # Create a ScrolledText widget for the prompt entry
        self.prompt_entry = scrolledtext.ScrolledText(self.prompt_frame, wrap=tk.WORD, width=80, height=5)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a button to trigger the question processing
        self.ask_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_button.grid(row=0, column=2, padx=10, pady=10)

        # Button to load documents
        self.load_docs_button = tk.Button(root, text="Load Documents", command=self.load_documents)
        self.load_docs_button.grid(row=0, column=3, padx=10, pady=10)

        # Frame to display answers
        self.result_frame = tk.Frame(root)
        self.result_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Left Frame for displaying the original answer
        self.original_answer_frame = tk.LabelFrame(self.result_frame, text="Original Answer", width=50)
        self.original_answer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_answer_text = scrolledtext.ScrolledText(self.original_answer_frame, wrap=tk.WORD, width=50, height=20)
        self.original_answer_text.pack(fill=tk.BOTH, expand=True)

        # Right Frame for displaying the final combined answer
        self.final_answer_frame = tk.LabelFrame(self.result_frame, text="Final Answer", width=50)
        self.final_answer_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.final_answer_text = scrolledtext.ScrolledText(self.final_answer_frame, wrap=tk.WORD, width=50, height=20)
        self.final_answer_text.pack(fill=tk.BOTH, expand=True)

        # Initialize language model names
        self.openai_model_name = "gpt-4o-mini"
        self.claude_model_name = "claude-3-5-sonnet-20240620"
        self.google_model_name = "gemini/gemini-1.5-flash"

        # Create model instances for interaction
        self.openai_model = LLM_Model(self.openai_model_name)
        self.claude_model = LLM_Model(self.claude_model_name)
        self.google_model = LLM_Model(self.google_model_name)

        self.models = [self.openai_model, self.claude_model, self.google_model]
        self.generator_model = self.openai_model
        self.answer_model = [self.google_model, self.claude_model]
        self.combiner_model = self.openai_model
        
        # MultiAgent interface to handle multiple models
        self.llm_multi_agents = LLM_MultiAgents(self.models)

        # Progress Bar to show status while processing responses
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

        # Exit button to close the application
        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.grid(row=3, column=0, columnspan=4, padx=10, pady=10)

    def load_documents(self):
        """Loads documents and appends their content to the prompt entry."""
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[("Text Files", "*.txt"), 
                       ("Python Files", "*.py"), 
                       ("PDF Files", "*.pdf"), 
                       ("Word Documents", "*.docx")]
        )
        
        if not file_path:
            return  # User canceled the dialog
        
        title = os.path.basename(file_path)
        content = ""

        # Read content based on file type
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
        elif file_path.endswith('.py'):
            with open(file_path, 'r') as file:
                content = file.read()
        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            content = "\n".join(paragraph.text for paragraph in doc.paragraphs)

        # Update prompt with title and content
        self.prompt_entry.insert(tk.END, f"{title}:\n{content}\n\n")

    def ask_question(self):
        """Handles the event when the user asks a question to the language models."""
        # Retrieve and validate user input
        original_prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not original_prompt:
            self.original_answer_text.delete(1.0, tk.END)
            self.final_answer_text.delete(1.0, tk.END)
            self.original_answer_text.insert(tk.END, "Please enter a question.")
            return

        # Simulate progress bar as responses are being gathered
        self.progress['value'] = 0
        self.progress['maximum'] = 100
        self.root.update_idletasks()
        
        # Fetching original answer from OpenAI model
        original_answer = self.llm_multi_agents.call_model(self.generator_model, original_prompt)
        
        self.progress['value'] = 20
        self.root.update_idletasks()
        
        # Generate optimized prompts using the OpenAI model
        optimized_prompts = self.llm_multi_agents.generate_different_version(self.generator_model, original_prompt, N_version=1)
        
        self.progress['value'] = 40
        self.root.update_idletasks()

        # Fetching answers from Google and Claude models
        answers = self.llm_multi_agents.call_models(self.answer_model, optimized_prompts[0], apply_refine=False, apply_cot=False)

        self.progress['value'] = 80
        self.root.update_idletasks()

        # Combine the answers into a final response
        final_answer = self.llm_multi_agents.combine_answer(self.combiner_model, answers)

        self.progress['value'] = 100
        self.root.update_idletasks()

        # Displaying the original and final answers in the text areas
        self.original_answer_text.delete(1.0, tk.END)
        self.original_answer_text.insert(tk.END, f"No combined Answer:\n{original_answer}\n\n")

        self.final_answer_text.delete(1.0, tk.END)
        self.final_answer_text.insert(tk.END, f"Final Combined Answer:\n{final_answer}")

def main():
    """Main function to initialize and run the application."""
    root = tk.Tk()
    app = LLMApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
