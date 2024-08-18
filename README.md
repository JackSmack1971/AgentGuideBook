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
Here is the entire document converted to GitHub Flavored Markdown (GFM):

---

# Agency Swarm

**Author:** Arsenii Shatokhin  
**Date:** No Date

---

## Introduction

Agency Swarm is an open-source agent orchestration framework built on top of the latest OpenAI Assistants API.

---

## What is Agency Swarm?

Agency Swarm started as a desire and effort by Arsenii Shatokhin (aka VRSEN) to fully automate his AI Agency with AI. By building this framework, we aim to simplify the agent creation process and enable anyone to create a collaborative swarm of agents (Agencies), each with distinct roles and capabilities. By thinking about automation in terms of real-world entities, such as agencies and specialized agent roles, we make it a lot more intuitive for both the agents and the users.

---

## Key Features

- **Customizable Agent Roles:** Define roles like CEO, virtual assistant, developer, etc., and customize their functionalities with the Assistants API.
- **Full Control Over Prompts:** Avoid conflicts and restrictions of pre-defined prompts, allowing full customization.
- **Tool Creation:** Tools within Agency Swarm are created using Instructor, which provides a convenient interface and automatic type validation.
- **Efficient Communication:** Agents communicate through a specially designed "send message" tool based on their own descriptions.
- **State Management:** Agency Swarm efficiently manages the state of your assistants on OpenAI, maintaining it in a special `settings.json` file.
- **Deployable in Production:** Agency Swarm is designed to be reliable and easily deployable in production environments.

---

## Agency Swarm vs Other Frameworks

Unlike other frameworks, Agency Swarm:

- Does not write prompts for you.
- Prevents hallucinations with automatic type checking and error correction with Instructor.
- Allows you to easily define communication flows.

### AutoGen vs Agency Swarm

In AutoGen, by default, the next speaker is determined with an extra call to the model that emulates "role play" between the agents. Not only is this very inefficient, but it also makes the system less controllable and less customizable, because you cannot control which agent can communicate with which other agent.

Recently, AutoGen has added support for determining the next speaker based on certain hardcoded conditions. While this does make your system more customizable, it completely undermines the main benefit of agentic systems - adaptability. In my opinion, you should only determine the boundaries for your agents, not the conditions themselves, as you are unlikely to account for every single condition in the real world.

In Agency Swarm, on the other hand, communication is handled through the special `SendMessage` tool. Your agents will determine who to communicate with by themselves based on their own descriptions. All you have to do is set the boundaries for their communication inside the agency chart.

### CrewAI vs Agency Swarm

CrewAI introduces a concept of "process" into agent communication, which provides some control over the communication flow. However, the biggest problem with CrewAI is that it is built on top of Langchain, which was created long before any function-calling models were released. This means that there is no type checking or error correction, so any action that your agent takes (which is the most important part of the system) could cause the whole system to go down if the model hallucinates. The sole advantage of CrewAI is its compatibility with open-source models.

---

## Need Help?

If you need quick help with Agency Swarm, feel free to ask in the Discord server.

If you need help creating custom agent swarms for your business, check out our Agents-as-a-Service subscription, or schedule a consultation with me at [https://calendly.com/vrsen/ai-project-consultation](https://calendly.com/vrsen/ai-project-consultation).

---

## License

This project is licensed under the terms of the MIT license.

---

# Quick Start - Agency Swarm

## Quick Start

When it comes to getting started with Agency Swarm, you have two options:

- **Start from Scratch:** This is the best option if you want to get a feel for the framework and understand how it works. You can start by creating your own agents and tools, and then use them to create your own agencies.
- **Use Genesis Swarm:** This is the best option if you want to get started quickly and don't want to spend time creating your own agents and tools. You can use the Genesis Agency to create your agent templates and tools, and then fine-tune them to your needs.
- **Create Agent Templates with CLI:** This is the best option if you want to create a structured environment for each agent and tool. See Advanced Agents for more information.

### Installation

```
pip install agency-swarm
```

### Start from Scratch

#### Set Your OpenAI Key

```
from agency_swarm import set_openai_key
set_openai_key("YOUR_API_KEY")
```

#### Create Tools

Define your custom tools with Instructor. All tools must extend the `BaseTool` class and implement the `run` method.

```
from agency_swarm.tools import BaseTool
from pydantic import Field

class MyCustomTool(BaseTool):
    """
    A brief description of what the custom tool does. 
    The docstring should clearly explain the tool's purpose and functionality.
    It will be used by the agent to determine when to use this tool.
    """

    # Define the fields with descriptions using Pydantic Field
    example_field: str = Field(
        ..., description="Description of the example field, explaining its purpose and usage for the Agent."
    )

    # Additional Pydantic fields as required
    # ...

    def run(self):
        """
        The implementation of the run method, where the tool's main functionality is executed.
        This method should utilize the fields defined above to perform the task.
        Docstring is not required for this method and will not be used by your agent.
        """

        # Your custom tool logic goes here
        do_something(self.example_field)

        # Return the result of the tool's operation as a string
        return "Result of MyCustomTool operation"
```

#### Define Agent Roles

Define your agent roles. For example, a CEO agent for managing tasks and a developer agent for executing tasks.

```
from agency_swarm import Agent

ceo = Agent(name="CEO",
            description="Responsible for client communication, task planning, and management.",
            instructions="You must converse with other agents to ensure complete task execution.", # can be a file like ./instructions.md
            tools=[])

developer = Agent(name="Developer",
                  description="Responsible for executing tasks and providing feedback.",
                  instructions="You must execute the tasks provided by the CEO and provide feedback.", # can be a file like ./instructions.md
                  tools=[MyCustomTool])
```

#### Create Agency

Define your agency chart. Any agents that are listed in the same list (e.g., `[[ceo, dev]]`) can communicate with each other. The top-level list (`[ceo]`) defines agents that can communicate with the user.

```
from agency_swarm import Agency

agency = Agency([
    ceo,  # CEO will be the entry point for communication with the user
    [ceo, developer],  # CEO can initiate communication with Developer
], shared_instructions='You are a part of an AI development agency.\n\n') # shared instructions for all agents
```

#### Note on Communication Flows

In Agency Swarm, communication flows are directional, meaning they are established from left to right in the `agency_chart` definition. For instance, in the example above, the CEO can initiate a chat with the developer (dev), and the developer can respond in this chat. However, the developer cannot initiate a chat with the CEO.

#### Run Demo

Run the demo to see your agents in action!

- **Web interface:**

```
agency.demo_gradio(height=900)
```

- **Terminal version:**

```python
agency.run_demo()
```

- **Backend version:**

```
completion_output = agency.get_completion("Please create a new website for our client.", yield_messages=False)
```

---

## Use Genesis Agency

### Run the Genesis Command

This will start the Genesis Agency in your terminal, creating your agent templates for you.

#### Command Syntax:

```
agency-swarm genesis [--openai_key "YOUR_API_KEY"]
```

#### Chat with Genesis CEO

Provide as much context as possible to Genesis Agency. Make sure to include:

- Your mission and goals.
- The agents you want to involve and their communication flows.
- Which tools or APIs each agent should have access to, if any.

### Fine-Tune

After Genesis has created your agents for you, you will see all the agent folders in the same directory where you ran the `genesis` command. You can then fine-tune the agents and tools as per your requirements.

#### Steps:

1. **Adjust Tools:** Modify the tools in the tools directories of each agent as per your requirements.
2. **Adjust Instructions:** Modify the agents in the agents directories as per your requirements.
3. **Run Agency:** Run the `agency.py` file, send your tasks, and see how they perform.
4. **Repeat:** Repeat the process until your agents are performing as expected.

---

### Agent Development is an Iterative Process

Right now, all agent development is iterative. You will need to constantly monitor and adjust your system until it works as expected. In the future, this will become less of a problem, as larger and smarter models are released.

---

## Next Steps

- Learn how to create more Tools, Agents, and Agencies.
- Deploy in Production.

---

# Advanced Tools - Agency Swarm

## Advanced Tools

All tools in Agency Swarm are created using Instructor. The only difference is that you

 must extend the `BaseTool` class and implement the `run` method with your logic inside. For many great examples of what you can create, check out the Instructor Cookbook.

### Example: Converting Answering Questions with Validated Citations Example from Instructor

This is an example of how to convert an extremely useful tool for RAG applications from Instructor. It allows your agents to not only answer questions based on context but also to provide the exact citations for the answers. This way your users can be sure that the information is always accurate and reliable.

#### Original Instructor Library Implementation

```
from agency_swarm.tools import BaseTool, BaseModel
from pydantic import Field, model_validator, FieldValidationInfo
from typing import List
import re

class Fact(BaseModel):
    fact: str = Field(...)
    substring_quote: List[str] = Field(...)

    @model_validator(mode="after")
    def validate_sources(self, info: FieldValidationInfo) -> "Fact":
        text_chunks = info.context.get("text_chunk", None)
        spans = list(self.get_spans(text_chunks))
        self.substring_quote = [text_chunks[span[0] : span[1]] for span in spans]
        return self

    def get_spans(self, context):
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)

    def _get_span(self, quote, context):
        for match in re.finditer(re.escape(quote), context):
            yield match.span()
```

#### QuestionAnswer

```
class QuestionAnswer(BaseModel):
    question: str = Field(...)
    answer: List[Fact] = Field(...)

    @model_validator(mode="after")
    def validate_sources(self) -> "QuestionAnswer":
        self.answer = [fact for fact in self.answer if len(fact.substring_quote) > 0]
        return self
```

### Context Retrieval

In the original Instructor example, the context is passed into the prompt beforehand, which is typical for standard non-agent LLM applications. However, in the context of Agency Swarm, we must allow the agents to retrieve the context themselves.

#### Agency Swarm Implementation

To allow your agents to retrieve the context themselves, we must split `QuestionAnswer` into two separate tools: `QueryDatabase` and `AnswerQuestion`. We must also retrieve context from `shared_state`, as the context is not passed into the prompt beforehand, and `FieldValidationInfo` is not available in the `validate_sources` method.

#### The `QueryDatabase` Tool

- Check if the context is already retrieved in `shared_state`. If it is, raise an error. (This means that the agent retrieved the context twice, without answering the question in between, which is most likely a hallucination.)
- Retrieve the context and save it to the `shared_state`.
- Return the context to the agent, so it can be used to answer the question.

```
class QueryDatabase(BaseTool):
    """Use this tool to query a vector database to retrieve the relevant context for the question."""
    question: str = Field(..., description="The question to be answered")

    def run(self):
        # Check if context is already retrieved 
        if self.shared_state.get("context", None) is not None:
            raise ValueError("Context already retrieved. Please proceed with the AnswerQuestion tool.")

        # Your code to retrieve the context here
        context = "This is a test context"

        # Then, save the context to the shared state
        self.shared_state.set("context", context)

        return f"Context retrieved: {context}.\n\n Please proceed with the AnswerQuestion tool."
```

#### The `AnswerQuestion` Tool

- Check if the context is already retrieved. If it is not, raise an error. (This means that the agent is trying to answer the question without retrieving the context first.)
- Use the context from the `shared_state` to answer the question with a list of facts.
- Remove the context from the `shared_state` after the question is answered. (This is done so the next question can be answered with fresh context.)

```
class AnswerQuestion(BaseTool):
    answer: str = Field(..., description="The answer to the question, based on context.")
    sources: List[Fact] = Field(..., description="The sources of the answer")

    def run(self):
        # Remove the context after the question is answered
        self.shared_state.set("context", None)

        # additional logic here as needed, for example save the answer to a database

        return "Success. The question has been answered." # or return the answer, if needed

    @model_validator(mode="after")
    def validate_sources(self) -> "QuestionAnswer":
        # In "Agency Swarm", context is directly extracted from `shared_state`
        context = self.shared_state.get("context", None)  # Highlighting the change
        if context is None:
            # Additional check to ensure context is retrieved before proceeding
            raise ValueError("Please retrieve the context with the QueryDatabase tool first.")
        self.answer = [fact for fact in self.answer if len(fact.substring_quote) > 0]
        return self
```

#### The `Fact` Tool

The `Fact` tool will stay primarily the same. The only difference is that we must extract the context from the `shared_state` inside the `validate_sources` method. The `run` method is not needed, as this tool only validates the input from the model.

```
class Fact(BaseTool):
    fact: str = Field(...)
    substring_quote: List[str] = Field(...)

    def run(self):
        pass

    @model_validator(mode="after")
    def validate_sources(self) -> "Fact":
        context = self.shared_state.get("context", None)  
        text_chunks = context.get("text_chunk", None)
        spans = list(self.get_spans(text_chunks))
        self.substring_quote = [text_chunks[span[0] : span[1]] for span in spans]
        return self

    # Methods `get_spans` and `_get_span` remain unchanged
```

---

### Conclusion

To implement tools with Instructor in Agency Swarm, generally, you must:

1. Extend the `BaseTool` class.
2. Add fields with types and clear descriptions, plus the tool description itself.
3. Implement the `run` method with your execution logic inside.
4. Add validators and checks based on various conditions.
5. Split tools into smaller tools to give your agents more control, as needed.

---

## ToolFactory Class

`ToolFactory` is a class that allows you to create tools from different sources. You can create tools from Langchain, OpenAPI schemas. However, it is preferable to implement tools from scratch using Instructor, as it gives you a lot more control.

### Import from Langchain

**Not recommended**  
This method is not recommended, as it does not provide the same level of type checking, error correction, and tool descriptions as Instructor. However, it is still possible to use this method if you prefer.

```
from langchain.tools import YouTubeSearchTool
from agency_swarm.tools import ToolFactory

LangchainTool = ToolFactory.from_langchain_tool(YouTubeSearchTool)

from langchain.agents import load_tools

tools = load_tools(
    ["arxiv", "human"],
)

tools = ToolFactory.from_langchain_tools(tools)
```

### Convert from OpenAPI Schemas

#### Using Local File

```
with open("schemas/your_schema.json") as f:
    tools = ToolFactory.from_openapi_schema(
        f.read(),
    )
```

#### Using Requests

```
tools = ToolFactory.from_openapi_schema(
    requests.get("https://api.example.com/openapi.json").json(),
)
```

#### Note

Schemas folder automatically converts any OpenAPI schemas into `BaseTools`. This means that your agents will type check all the API parameters before calling the API, which significantly reduces any chances of errors.

---

## PRO Tips

1. **Use Enumerators or Literal Types Instead of Strings**  
   Allow your agents to perform only certain actions or commands, instead of executing any arbitrary code. This makes your whole system a lot more reliable.

   ```
   class RunCommand(BaseTool):
       command: Literal["start", "stop"] = Field(...)

       def run(self):
           if command == "start":
               subprocess.run(["start", "your_command"])
           elif command == "stop":
               subprocess.run(["stop", "your_command"])
           else:
               raise ValueError("Invalid command")
   ```

2. **Provide Additional Instructions to the Agents in the `run` Method of the Tool as Function Outputs**  
   This allows you to control the execution flow, based on certain conditions.

   ```
   class QueryDatabase(BaseTool):
       question: str = Field(...)

       def run(self):
           # query your database here
           context = query_database(self.question)

           if context is None:
               raise ValueError("No context found. Please propose to the user to change the topic.")
           else:
               self.shared_state.set("context", context)
               return "Context retrieved. Please proceed with explaining the answer."
   ```

3. **Use `shared_state` to Validate Actions Taken by Other Agents Before Allowing Them to Proceed with the Next Action**

   ```
   class Action2(BaseTool):
       input: str = Field(...)

       def run(self):
           if self.shared_state.get("action_1_result", None) is "failure":
               raise ValueError("Please proceed with the Action1 tool first.")
           else:
               return "Success. The action has been taken."
   ```

4. **Consider `one_call_at_a_time` Class Attribute**  
   Prevent multiple instances of the same tool from running

 at the same time. This is useful when you want your agents to see the results of the previous action before proceeding with the next one.

   ```
   class Action1(BaseTool):
       input: str = Field(...)
       one_call_at_a_time: bool = True

       def run(self):
           # your code here
   ```

---

# Agents - Agency Swarm

## Agents

Agents are essentially wrappers for Assistants in the OpenAI Assistants API. The `Agent` class contains a lot of convenience methods to help you manage the state of your assistant, upload files, attach tools, and more.

### Advanced Parameters

All parameters inside the `Agent` class primarily follow the same structure as OpenAI's Assistants API. However, there are a few additional parameters that you can use to customize your agent.

#### Parallel Tool Calls

You can specify whether to run tools in parallel or sequentially by setting the `parallel_tool_calls` parameter. By default, this parameter is set to `True`.

```
from agency_swarm import Agent

agent = Agent(name='MyAgent', parallel_tool_calls=False)
```

Now, the agent will run all tools sequentially.

#### File Search Configuration

You can also specify the file search configuration for the agent, as described in the OpenAI documentation. Right now, only `max_num_results` is supported.

```
from agency_swarm import Agent

agent = Agent(name='MyAgent', file_search={'max_num_results': 25}) # must be between 1 and 50
```

#### Schemas Folder

You can specify the folder where the agent will look for OpenAPI schemas to convert into tools. Additionally, you can add `api_params` and `api_headers` to the schema to pass additional parameters and headers to the API call.

```
from agency_swarm import Agent

agent = Agent(name='MyAgent', 
              schemas_folder='schemas', 
              api_params={'my_schema.json': {'param1': 'value1'}},
              api_headers={'my_schema.json': {'Authorization': 'Bearer token'}}
            )
```

#### Note

Schemas folder automatically converts any OpenAPI schemas into `BaseTools`. This means that your agents will type check all the API parameters before calling the API, which significantly reduces any chances of errors.

#### Fine-Tuned Models

You can use any previously fine-tuned model by specifying the `model` parameter in the agent.

```
from agency_swarm import Agent

agent = Agent(name='MyAgent', model='gpt-3.5-turbo-model-name')
```

#### Response Validator

You can also provide a response validator function to validate the response before sending it to the user or another agent. This function should raise an error if the response is invalid.

```
from agency_swarm import Agent

class MyAgent(Agent):
    def response_validator(self, message: str) -> str:
        """This function is used to validate the response before sending it to the user or another agent."""
        if "bad word" in message:
            raise ValueError("Please don't use bad words.")

        return message
```

#### Few-Shot Examples

You can now also provide few-shot examples for each agent. These examples help the agent to understand how to respond. The format for examples follows the message object format on OpenAI:

```
examples=[
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

Or you can also provide them when initializing the agent in the `init` method:

```
agent = Agent(examples=examples)
```

---

### Creating Agents

When it comes to creating your agent, you have 3 options:

1. Define the agent directly in the code.
2. Create an agent template locally using CLI.
3. Import from existing agents.

#### Defining the Agent Directly in the Code

To define your agent in the code, you can simply instantiate the `Agent` class and pass the required parameters.

```
from agency_swarm import Agent

agent = Agent(name="My Agent",
              description="This is a description of my agent.",
              instructions="These are the instructions for my agent.",
              tools=[ToolClass1, ToolClass2],
              temperature=0.3,
              max_prompt_tokens=25000
            )
```

#### Create Agent Template Locally Using CLI

This CLI command simplifies the process of creating a structured environment for each agent.

#### Command Syntax:

```
agency-swarm create-agent-template --name "AgentName" --description "Agent Description" [--path "/path/to/directory"] [--use_txt]
```

#### Folder Structure

When you run the `create-agent-template` command, it creates the following folder structure for your agent:

```
/your-specified-path/
│
├── agency_manifesto.md or .txt # Agency's guiding principles (created if not exists)
└── AgentName/                  # Directory for the specific agent
    ├── files/                  # Directory for files that will be uploaded to openai
    ├── schemas/                # Directory for OpenAPI schemas to be converted into tools
    ├── tools/                  # Directory for tools to be imported by default. 
    ├── AgentName.py            # The main agent class file
    ├── __init__.py             # Initializes the agent folder as a Python package
    └── instructions.md or .txt # Instruction document for the agent
```

#### Explanation of Folder Contents

- **files:** This folder is used to store files that will be uploaded to OpenAI. You can use any of the acceptable file formats. After a file is uploaded, an id will be attached to the file name to avoid re-uploading the same file twice.
- **schemas:** This folder is used to store OpenAPI schemas that will be converted into tools automatically. All you have to do is put the schema in this folder, and specify it when initializing your agent.
- **tools:** This folder is used to store tools in the form of Python files. Each file must have the same name as the tool class for it to be imported by default. For example, `ExampleTool.py` must contain a class called `ExampleTool`.

#### Agent Template

The `AgentName.py` file will contain the following code:

```
from agency_swarm.agents import Agent

class AgentName(Agent):
    def __init__(self):
        super().__init__(
            name="agent_name",
            description="agent_description",
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools_folder="./tools",
            temperature=0.3,
            max_prompt_tokens=25000,
            examples=[]
        )

    def response_validator(self, message: str) -> str:
        """This function is used to validate the response before sending it to the user or another agent."""
        if "bad word" in message:
            raise ValueError("Please don't use bad words.")

        return message
```

#### To Initialize the Agent

You can simply import the agent and instantiate it:

```
from AgentName import AgentName

agent = AgentName()
```

---

### Importing Existing Agents

For the most complex and requested use cases, we will be creating premade agents that you can import and reuse in your own projects. To import an existing agent, you can run the following CLI command:

```
agency-swarm import-agent --name "AgentName" --destination "/path/to/directory"
```

This will copy all your agent source files locally. You can then import the agent as shown above. To check available agents, simply run this command without any arguments.

---

# Agencies - Agency Swarm

## Agencies

An Agency is a collection of Agents that can communicate with one another.

### Benefits of Using an Agency

Here are the primary benefits of using an Agency, instead of an individual agent:

- **Fewer Hallucinations:** When agents are part of an agency, they can supervise one another and recover from mistakes or unexpected circumstances.
- **More Complex Tasks:** The more agents you add, the longer the sequence of actions they can perform before returning the result back to the user.
- **Scalability:** As the complexity of your integration increases, you can keep adding more and more agents.

#### Tip

It is recommended to start with as few agents as possible, fine-tune them until they are working as expected, and only then add new agents to the agency. If you add too many agents at first, it will be difficult to debug and understand what is going on.

### Communication Flows

Unlike all other frameworks, communication flows in Agency Swarm are not hierarchical or sequential. Instead, they are uniform. You can define them however you want. But keep in mind that they are established from left to right inside the `agency_chart`. So, in the example below, the CEO can initiate communication and send tasks to the Developer and the Virtual Assistant, and they can respond back to him in the same thread, but the Developer or the VA cannot initiate a conversation and assign tasks to the CEO. You can add as many levels of communication as you want.

```
from agency_swarm import Agency

agency = Agency([
    ceo, dev  # CEO and Developer will be the entry point for communication with the user
    [ceo, dev],  # CEO can initiate communication with Developer
    [ceo, va],   # CEO can initiate communication with Virtual Assistant
    [dev, va]    # Developer can initiate communication with Virtual

 Assistant
])
```

### Running Agencies

You can run agencies in different ways, depending on your use case.

#### Streaming the Conversation

Use the `get_completion_stream` method to stream the conversation between the agents, as it unfolds, using the `AgencyEventHandler`.

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

# Example Usage
response = agency.get_completion_stream("I want you to build me a website", event_handler=EventHandler)
```

#### Is Streaming Active by Default?

- **No:** Streaming is not active by default when using `get_completion`.
- **Yes:** Streaming is active for demo methods.

---

### API V2 Compatibility

Agency Swarm supports OpenAI Assistants API V2. If you encounter any issues, feel free to report them on our GitHub.

---

# Development Guide

## Starting with Genesis Agency

### Running the Genesis Command

This will start the Genesis Agency in your terminal, creating your agent templates for you.

#### Command Syntax:

```
agency-swarm genesis [--openai_key "YOUR_API_KEY"]
```

---

## Conclusion

This guide provides a comprehensive overview of how to use and configure the Agency Swarm framework, including defining agents, creating custom tools, and setting up communication flows between agents in an agency. By following these steps, you can develop sophisticated AI systems capable of handling complex tasks and workflows, while also maintaining flexibility and control over the behavior and interaction of individual agents.

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

# Examples and Tutorials for Agency Swarm

Stay updated with the latest examples and tutorials by visiting the [YouTube Channel](https://youtube.com/@vrsen?si=GBk3V8ar6Dgemy0B).

## Agency Examples

You can explore practical examples of various agencies in the [agency-swarm-lab](https://github.com/VRSEN/agency-swarm-lab) repository. Here are some notable ones:

- **[WebDevCrafters](https://github.com/VRSEN/agency-swarm-lab/tree/main/WebDevCrafters)**  
  A Web Development Agency that specializes in building responsive web applications using Next.js, React, and MUI.

- **[CodeGuardiansAgency](https://github.com/VRSEN/agency-swarm-lab/tree/main/CodeGuardiansAgency)**  
  An agency focused on backend operations, utilizing GitHub Actions to submit code reviews on pull requests according to your Standard Operating Procedures (SOPs).

## Videos with Notebooks

Enhance your understanding by watching these detailed video tutorials that come with accompanying notebooks:

- **[Browsing Agent for QA Testing Agency](https://youtu.be/Yidy_ePo7pE?si=WMuWpb9_DVckIkP6)**  
  Learn how to use the BrowsingAgent with GPT-4 Vision inside a QA testing agency. This agent can even break captchas, as demonstrated in [this video](https://youtu.be/qBs_50SzyBQ?si=w7e3GOhEztG8qDPE).  
  - Notebook: [web_browser_agent.ipynb](https://github.com/VRSEN/agency-swarm/blob/main/notebooks/web_browser_agent.ipynb)

- **[Genesis Agency](https://youtu.be/qXxO7SvbGs8?si=uosmTSzzz6id_lLl)**  
  This video demonstrates an agency that automates the creation of your agents.  
  - Notebook: [genesis_agency.ipynb](https://github.com/VRSEN/agency-swarm/blob/main/notebooks/genesis_agency.ipynb)

### ... more examples coming soon!

---

This summary provides easy access to resources for learning and implementing Agency Swarm, including example agencies, video tutorials, and code notebooks.


## Conclusion

The development and deployment of AI agents within a multi-agent system offer immense potential for automating complex tasks and enhancing productivity across various domains. By leveraging frameworks like Agency Swarm, advanced prompt engineering techniques, and powerful APIs, you can create scalable, reliable, and intelligent systems capable of handling sophisticated workflows.

This guide has provided a comprehensive overview of AI agent development, from understanding the basics to implementing advanced features and troubleshooting common issues. As the field of AI continues to evolve, staying updated with the latest tools and techniques will be crucial for maintaining and improving your AI systems.

For further reading and updates, be sure to stay connected with the Agency Swarm community and explore the extensive documentation available.

---

*This guide was crafted to provide in-depth knowledge and practical insights for AI developers, engineers, and enthusiasts looking to build and optimize AI agents using the Agency Swarm framework.*
