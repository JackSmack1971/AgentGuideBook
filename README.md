# Ultimate Guide to AI Agent Development and Multi-Agent Systems with Agency Swarm

## Introduction

The rapid advancement in AI technologies has opened up new opportunities for automating complex tasks across various domains. At the forefront of these innovations is the concept of AI agents—intelligent entities capable of performing tasks autonomously by following instructions, retrieving relevant data, and making decisions. While individual AI agents can be powerful, combining them into an "Agency" can further enhance their capabilities, allowing for more complex workflows, reduced errors, and increased scalability.

This guide provides an in-depth look into AI agent development, with a focus on multi-agent systems using the Agency Swarm framework. We will explore everything from basic concepts to advanced techniques, including prompt engineering, API integrations, troubleshooting common issues, and deploying scalable AI solutions. By the end of this guide, you'll be equipped to build, manage, and optimize AI agents and their respective agencies.

---

## Understanding AI Agents and Multi-Agent Systems

### What are AI Agents?

AI agents are autonomous entities designed to perform specific tasks by processing data, executing actions, and making decisions based on predefined instructions and the context they operate in. Unlike traditional AI automations that follow rigid, pre-programmed workflows, AI agents possess decision-making capabilities, allowing them to adapt to new situations and handle complex tasks with greater flexibility.

### The Role of Multi-Agent Systems

A multi-agent system (MAS) is a collection of AI agents that communicate and collaborate to achieve a common goal. The key benefits of using MAS include:

1. **Reduced Hallucinations**: When agents are part of an agency, they can supervise one another and correct mistakes, reducing the likelihood of errors.
2. **Handling Complex Tasks**: By distributing responsibilities across multiple agents, a MAS can tackle more complex workflows that a single agent would struggle with.
3. **Scalability**: As your tasks and projects grow in complexity, MAS allows you to add more agents without overloading individual ones, thus maintaining efficiency and reliability.

Based on the recent updates provided for the Agency Swarm framework, here’s a GitHub Flavored Markdown (GFM) formatted summary that captures the new features, enhancements, and instructions for leveraging them:

---

# Agency Swarm Updates (5.10.24)

## 1. Streaming Responses

### Overview
The `get_completion_stream` method now allows for streaming conversations between agents. This method is similar to what is documented officially, but with an extension of the `AgencyEventHandler` class that includes additional properties like `agent_name` and `recipient_agent_name`.

### Example Implementation
```
from typing_extensions import override
from agency_swarm import AgencyEventHandler

class EventHandler(AgencyEventHandler):
    @override
    def on_text_created(self, text) -> None:
        # Get the name of the agent sending the message
        print(f"\n{self.recipient_agent_name} @ {self.agent_name} > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\n{self.recipient_agent_name} > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

    @classmethod
    def on_all_streams_end(cls):
        print("\n\nAll streams have ended.")  # Conversation is over and message is returned to the user.

# Usage
response = agency.get_completion_stream("I want you to build me a website", event_handler=EventHandler)
```
**Note:** The `on_all_streams_end` method is crucial since your event handler might be invoked multiple times by different agents.

### Is Streaming Active by Default?
- **No**: Streaming is not active by default when using `get_completion`.
- **Yes**: It is active for demo methods.

---

## 2. Assistant API V2 Integration

### New Features:
- **Temperature and Token Settings**: You can now set the `temperature` and `max_prompt_tokens` parameters in both `Agent` and `Agency` classes.
  - **Defaults**: Effective for most scenarios.
  - **Recommendation for Coding**: Set `temperature` to 0.

### Compatibility:
- The framework now supports Assistant API V2, with an effort to maintain compatibility across previous versions. Please report any issues encountered.

---

## 3. Managing Agents

### The Agent Class
Agents are essentially wrappers around Assistants in the OpenAI Assistants API. The `Agent` class offers various methods to manage the state, upload files, attach tools, and more.

#### Example: Defining an Agent in Code
```
from agency_swarm import Agent

agent = Agent(name="My Agent",
              description="This is a description of my agent.",
              instructions="These are the instructions for my agent.",
              tools=[ToolClass1, ToolClass2],
              temperature=0.3,
              max_prompt_tokens=25000)
```

### Creating an Agent Template Using CLI
You can create a structured environment for each agent using a simple CLI command.

#### Command Syntax:
```
agency-swarm create-agent-template --name "AgentName" --description "Agent Description" [--path "/path/to/directory"] [--use_txt]
```

#### Folder Structure Created:
- **`/your-specified-path/`**
  - **`agency_manifesto.md`** (created if not exists)
  - **`AgentName/`**
    - **`files/`**: Store files to be uploaded to OpenAI.
    - **`schemas/`**: Store OpenAPI schemas to be converted into tools.
    - **`tools/`**: Store Python files as tools.
    - **`AgentName.py`**: Main agent class file.
    - **`__init__.py`**: Initializes the agent folder as a Python package.
    - **`instructions.md` or `.txt`**: Instruction document for the agent.

### Importing Existing Agents
For complex use cases, pre-made agents can be imported and reused in your projects.

#### Import Command:
```
agency-swarm import-agent --name "AgentName" --destination "/path/to/directory"
```

---

## 4. Few-Shot Examples

You can now provide few-shot examples to each agent to help guide their responses.

### Example Format:
```
examples = [
    {
        "role": "user",
        "content": "Hi!",
        "attachments": [],
        "metadata": {},
    },
    {
        "role": "assistant",
        "content": "Hi! I am the CEO. I am here to help you with your tasks. Please tell me what you need help with.",
        "attachments": [],
        "metadata": {},
    }
]

agent.examples = examples
```
You can also provide these examples during the agent's initialization.

---

### Communication Flows in Agency Swarm

In the Agency Swarm framework, communication flows between agents are uniform and non-hierarchical. This means you can define communication patterns based on your specific needs, ensuring that all agents can interact seamlessly. Agents at the top level of the agency chart can communicate directly with the user, while those in nested levels can only interact with each other or their designated communication partners.

```
from agency_swarm import Agency

agency = Agency([
    ceo, dev,  # CEO and Developer can communicate with the user
    [ceo, dev],  # CEO can communicate with Developer
    [ceo, va],   # CEO can communicate with Virtual Assistant
    [dev, va]    # Developer can communicate with Virtual Assistant
])
```

### Streaming Responses

Agency Swarm allows you to stream the conversation between agents using the `get_completion_stream` method. This is particularly useful for real-time applications where you want to monitor and control agent interactions as they occur.

```
from typing_extensions import override
from agency_swarm import AgencyEventHandler

class EventHandler(AgencyEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\n{self.recipient_agent_name} @ {self.agent_name} > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

response = agency.get_completion_stream("Build a website", event_handler=EventHandler)
```

---

## Advanced Prompt Engineering Techniques

### The Importance of Prompts

In AI agent development, crafting effective prompts is crucial for guiding agents to perform their tasks correctly. Prompts must be clear, concise, and contextually relevant to ensure that the agent understands the instructions and executes the task accurately.

### Chain-of-Thought (CoT) and Retrieval Augmented Generation (RAG)

- **Chain-of-Thought (CoT)**: This technique involves breaking down complex tasks into a series of smaller, manageable steps. By guiding the agent through each step, you can reduce errors and improve the overall accuracy of the task execution.
  
- **Retrieval Augmented Generation (RAG)**: RAG combines information retrieval with text generation, enabling agents to pull relevant data from external sources before generating a response. This approach is particularly useful for tasks that require up-to-date or context-specific information.

### Example: Using CoT and RAG

```
def generate_ad_copy(product_name, product_features):
    # Chain-of-Thought: Break down the task into steps
    steps = [
        f"Describe the {product_name}.",
        f"Highlight the key features: {product_features}.",
        "Craft a catchy slogan."
    ]
    # RAG: Retrieve related data from external sources
    additional_data = retrieve_additional_data(product_name, product_features)
    
    # Generate the final ad copy
    ad_copy = ""
    for step in steps:
        ad_copy += agent.generate_response(step + additional_data)
    
    return ad_copy
```

---

## Integrating APIs with AI Agents

### SerpAPI for Real-Time Data Retrieval

The SerpAPI allows AI agents to retrieve real-time data from search engines, ensuring that the information they use is current and relevant. This is essential for tasks such as market analysis, trend monitoring, and competitive research.

```
import serpapi

def get_latest_trends(query):
    client = serpapi.GoogleSearchAPI(api_key="YOUR_API_KEY")
    results = client.search(query)
    return results
```

### You.com API for Contextual Information

The You.com API offers a suite of tools designed to ground AI outputs in the most recent and accurate information available on the web. It can be used to enhance the contextual understanding of AI agents, particularly when dealing with dynamic or time-sensitive topics.

```
import requests

def get_ai_snippets_for_query(query):
    headers = {"X-API-Key": "YOUR_API_KEY"}
    params = {"query": query}
    response = requests.get("https://api.ydc-index.io/search", params=params, headers=headers)
    return response.json()
```

### Mistral AI API for Specialized Tasks

Mistral AI API provides specialized endpoints that can be integrated into your AI agents for tasks such as advanced data processing, natural language understanding, and more. This API is useful for developing agents that require a higher level of intelligence and customization.

```
import mistral

def process_data_with_mistral(data):
    client = mistral.Client(api_key="YOUR_API_KEY")
    processed_data = client.process_data(data)
    return processed_data
```

### Pinecone Vector Database for Similarity Search

Pinecone is a vector database that allows AI agents to perform similarity searches, which is essential for tasks like recommendation systems, anomaly detection, and content matching. Integrating Pinecone with your AI agents can significantly enhance their ability to process and analyze large datasets.

```
import pinecone

def vector_search(query_vector):
    client = pinecone.Client(api_key="YOUR_API_KEY")
    index = client.Index("your-index-name")
    results = index.query(query_vector)
    return results
```

---

## Troubleshooting and Best Practices

### Common Issues and Solutions for Windows and Mac Users

- **Windows UTF-8 Encoding Issue**: Windows users may encounter installation errors related to UTF-8 encoding. To fix this, set the environment variable `PYTHONUTF8=1` before running installation commands. Alternatively, ensure the README.md file in `setup.py` is read with 'utf-8' encoding.

- **File Permission Errors on Windows**: If you encounter `PermissionError` when handling files, ensure that files are closed before attempting operations like renaming. This can be done by adding `f.close()` before `os.rename()`.

- **ChromeDriver Issues on Mac**: Mac users may experience issues with ChromeDriver unexpectedly exiting. The recommended solution is to quit Chrome before using the browsing agent, ensuring that no conflicts arise during execution.

### Best Practices for AI Agent Development

1. **Start Small**: Begin with a minimal set of agents and fine-tune them until they perform as expected. This makes debugging easier and ensures that the system remains manageable.

2. **Modular Design**: Design agents to be modular, with each agent responsible for a specific task. This reduces complexity and makes the system easier to scale.

3. **Effective Communication Flows**: Clearly define communication flows between agents to prevent confusion and ensure that tasks are executed in the correct order.

4. **Use Shared State Wisely**: When agents need to share data, use shared state variables to minimize the risk of data loss or corruption due to hallucinations or errors.

5. **Regular Updates and Testing**: Continuously update and test your agents to ensure they remain compatible with the latest APIs, libraries, and frameworks.

---

## Case Study: Building a Social Media Marketing Agency

### Step-by-Step Process

1. **Define the Agents**: Start by defining the agents needed for the agency, such as an Ad Copy Agent, Image Generator Agent, and Facebook Manager Agent.

2. **Set Up Communication Flows**: Establish a sequential communication flow where the CEO agent coordinates tasks between the Ad Copy Agent, Image Generator Agent, and Facebook Manager Agent.

3. **Create Tools and Integrations**: Develop tools for each agent, such as an `ImageGenerator` tool that uses OpenAI's DALL-E 3 to generate images, and a `FacebookManager` tool that interacts with the Facebook API to post ads.

4. **Test and Fine-Tune**: Thoroughly test each tool and agent, making necessary adjustments to prompts, communication flows, and integrations.

5. **Deploy and Monitor**: Deploy the agency using the `agency.run_demo()` command and monitor its performance through the Gradio interface or terminal.

### Example Code: Facebook

 Manager Agent

```
from agency_swarm.agents import Agent

class FacebookManager(Agent):
    def __init__(self):
        super().__init__(
            name="Facebook Manager",
            description="Manages Facebook ad campaigns.",
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools_folder="./tools",
            temperature=0.3,
            max_prompt_tokens=25000,
            examples=[]
        )

    def post_ad(self, ad_copy, image_path):
        # Posting the ad on Facebook
        response = self.run_tool("AdCreator", ad_copy=ad_copy, image_path=image_path)
        return response
```
Here's a GitHub Flavored Markdown (GFM) formatted summary and solution based on the issue described in the document you provided:

---

# Resolving the Readline Module Issue in Agency Swarm

## Issue

When working with the Agency Swarm framework, users on Windows (PC) may encounter an issue where the `readline` module is not available. This can cause errors when running the framework, particularly in the `agency.py` file.

## Solution 1: Modifying `agency.py` to Handle Windows OS

### Steps:

1. **Locate the `agency.py` File:**
   - Navigate to the installed library directory in your virtual environment:
     ```
     \venv\Lib\site-packages\agency_swarm\agency\agency.py
     ```

2. **Modify the Import Statement:**
   - Replace the existing import statement with a conditional import to skip `readline` on Windows (`nt`).
     ```
     if os.name != 'nt':
         import readline
     ```

3. **Update the `setup_autocomplete` Function:**
   - Replace the existing `setup_autocomplete` function in the same file with the following:
     ```
     def setup_autocomplete(self):
         """
         Sets up readline with the completer function.
         """
         if os.name == 'nt':
             # If running on Windows simply pass as readline is not available
             pass
         else:
             self.recipient_agents = [agent.name for agent in self.main_recipients]  # Cache recipient agents for autocomplete
             readline.set_completer(self.recipient_agent_completer)
             readline.parse_and_bind('tab: complete')
     ```

## Solution 2: Installing `pyreadline3`

If you prefer not to modify the `agency.py` file or if you need the `readline` functionality on Windows, you can install the `pyreadline3` module, which is a compatible alternative for Windows.

### Steps:

1. **Install `pyreadline3`:**
   - Run the following command in your virtual environment:
     ```
     pip install pyreadline3
     ```

2. **Verify the Installation:**
   - Ensure that the module is correctly installed by running your project again.

By following these steps, you should be able to resolve the `readline` module issue on Windows systems, ensuring that your Agency Swarm framework runs smoothly.

---

This summary provides a clear, step-by-step guide to resolving the `readline` module issue that users might encounter when working with the Agency Swarm framework on Windows.

---

# Using Open-Source Models with Agency Swarm

While OpenAI is typically recommended for use with the Agency Swarm framework, there are scenarios where you might prefer to work with open-source models. This guide provides an overview of tested and upcoming open-source projects that mimic the Assistants API, along with instructions for integrating these models with Agency Swarm.

---

## ✅ Tested Open-Source Projects

- **[Open Assistant API](https://github.com/MLT-OSS/open-assistant-api)**  
  - Stable and tested. However, there is [one known bug](https://github.com/MLT-OSS/open-assistant-api/issues/61) that needs resolution. Currently, this is the best open-source option.

## 🔜 Projects Under Development

- **[Astra Assistants API](https://github.com/datastax/astra-assistants-api)**  
  - Under active development. Some issues with tool logic are still being worked on ([issue #27](https://github.com/datastax/astra-assistants-api/issues/27)).

- **[OpenOpenAI](https://github.com/transitive-bullshit/OpenOpenAI)**  
  - Unverified but likely operational.

- **[LiteLLM](https://github.com/BerriAI/litellm/issues/2842)**  
  - An Assistants API Proxy currently in development. It could become the preferred choice once it stabilizes.

---

## Integrating Open-Source Models with Agency Swarm

To use open-source models with Agency Swarm, it’s recommended to install an earlier version of the `agency-swarm` package, as most open-source projects are not yet compatible with streaming and Assistants V2.

### Installation

1. **Install the compatible version of Agency Swarm:**

   ```
   pip install agency-swarm==0.1.7
   ```

2. **Set up the OpenAI client with your local open-source model:**

   ```
   import openai
   from agency_swarm import set_openai_client

   client = openai.OpenAI(api_key="your-api-key", base_url="http://127.0.0.1:8000/")
   set_openai_client(client)
   ```

3. **Define your agents using the open-source model:**

   ```
   from agency_swarm import Agent

   ceo = Agent(name="ceo", description="I am the CEO", model='ollama/llama3')
   ```

### Running the Agency with Gradio

For a simple Gradio interface, use the non-streaming `demo_gradio` method from the `agency-swarm-lab` repository:

```
from agency_swarm import Agency
from .demo_gradio import demo_gradio

agency = Agency([ceo])

demo_gradio(agency)
```

### Backend Integration Example

For direct backend usage, you can get completions like this:

```
agency.get_completion("I am the CEO")
```

---

## Known Limitations

- **No Function Calling Support:**  
  Most open-source models do not support function calling, which prevents agents from interacting with other agents within the agency. As a result, such models must be positioned at the end of the agency chart and cannot utilize any tools.

- **Limited Retrieval-Augmented Generation (RAG):**  
  Open-source models typically have limited RAG capabilities. You may need to develop a custom tool with a dedicated vector database.

- **No Code Interpreter Support:**  
  The Code Interpreter feature is still under development for all open-source assistants API implementations.

---

## Future Plans

We will provide updates as new open-source assistant API implementations stabilize. If you successfully integrate other open-source projects with Agency Swarm, please share your experience through an issue or pull request on the project's repository.

---

## Conclusion

The development and deployment of AI agents within a multi-agent system offer immense potential for automating complex tasks and enhancing productivity across various domains. By leveraging frameworks like Agency Swarm, advanced prompt engineering techniques, and powerful APIs, you can create scalable, reliable, and intelligent systems capable of handling sophisticated workflows.

This guide has provided a comprehensive overview of AI agent development, from understanding the basics to implementing advanced features and troubleshooting common issues. As the field of AI continues to evolve, staying updated with the latest tools and techniques will be crucial for maintaining and improving your AI systems.

For further reading and updates, be sure to stay connected with the Agency Swarm community and explore the extensive documentation available.

---

*This guide was crafted to provide in-depth knowledge and practical insights for AI developers, engineers, and enthusiasts looking to build and optimize AI agents using the Agency Swarm framework.*
