import os
import sys
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from deepagents import create_deep_agent

def main():
    # 1. Load environment variables from the .env file at the repository root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    env_path = os.path.join(repo_root, ".env")
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
        
    # Configure the GOOGLE_APPLICATION_CREDENTIALS environment variable
    # to be an absolute path relative to the repo root so Google's client libraries can find it.
    creds_rel_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_rel_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
            os.path.join(repo_root, creds_rel_path)
        )
        
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("Error: GOOGLE_CLOUD_PROJECT is not set in environment or .env file.")
        sys.exit(1)
        
    print(f"Initializing Vertex AI model (project: {project_id})...")
    
    # 2. Initialize the Vertex AI Chat model (Gemini 2.5 Flash)
    try:
        llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=project_id,
            location="us-central1"
        )
        
        # 3. Create the Deep Agent
        print("Creating Deep Agent...")
        agent = create_deep_agent(
            model=llm,
            system_prompt="You are a helpful assistant."
        )
        
        # 4. Invoke the Deep Agent
        prompt = "Explain in one sentence why deep learning is powerful."
        print(f"Invoking agent with prompt: '{prompt}'")
        
        config = {"configurable": {"thread_id": "gemini-session"}}
        result = agent.invoke({
            "messages": [{"role": "user", "content": prompt}]
        }, config=config)
        
        # 5. Extract and print the response
        last_message = result["messages"][-1]
        content = last_message.content
        
        if isinstance(content, list):
            text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
            response_text = " ".join(text_parts)
        else:
            response_text = str(content)
            
        print("\nAgent Response:")
        print(response_text)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
