# Conceptual Guide

This guide introduces key parts of the LangChain framework and its architecture.

## Architecture

LangChain as a framework consists of multiple packages, each with a specific role.

### langchain-core

The `langchain-core` package contains base abstractions for different components and ways to compose them together. It defines interfaces for core components like LLMs, vector stores, retrievers, and more, without including third-party integrations. Dependencies are kept very lightweight.

### Partner Packages

Popular integrations are split into their own packages, such as `langchain-openai`, `langchain-anthropic`, etc., for better support. Less popular integrations are housed in `langchain-community`.

### langchain

The `langchain` package contains chains, agents, and retrieval strategies that constitute an application's cognitive architecture. These components are generic across all integrations and are not tied to any specific third-party service.

### langchain-community

The `langchain-community` package includes third-party integrations maintained by the LangChain community. It covers various components like LLMs, vector stores, and retrievers, with all dependencies kept optional for a lightweight package.

### langgraph

`langgraph` is an extension of LangChain aimed at building robust, stateful multi-actor applications using LLMs by modeling steps as edges and nodes in a graph. It offers both high-level interfaces for common agent types and a low-level API for custom flows.

### langserve

The `langserve` package facilitates the deployment of LangChain chains as REST APIs, making it easy to set up a production-ready API.

### LangSmith

LangSmith is a developer platform for debugging, testing, evaluating, and monitoring LLM applications.

## LangChain Expression Language (LCEL)

LCEL is a declarative way to chain LangChain components. Designed from the outset for production use, LCEL supports features like streaming, async support, optimized parallel execution, retries, fallbacks, and seamless integration with LangSmith for tracing.

### Key Features of LCEL

- **First-class streaming support**: Stream tokens directly from an LLM to a parser for faster response times.
- **Async support**: Chains can be invoked synchronously or asynchronously, allowing the same code to be used in prototypes and production.
- **Optimized parallel execution**: Automatically executes parallel steps to reduce latency.
- **Retries and fallbacks**: Configurable retries and fallbacks improve reliability at scale.
- **Access intermediate results**: Intermediate results are accessible and streamable, useful for debugging or user notifications.
- **Input and output schemas**: LCEL chains come with Pydantic and JSONSchema schemas for validation.
- **Seamless LangSmith tracing**: All steps are automatically logged for observability and debuggability.

LCEL provides a consistent and customizable framework over legacy subclassed chains. For migration guidance from legacy chains, see the relevant migration guide.

## Runnable Interface

The `Runnable` protocol simplifies the creation of custom chains in LangChain. Many LangChain components implement this protocol, which includes methods for streaming (`stream`, `astream`), invoking (`invoke`, `ainvoke`), batching (`batch`, `abatch`), and logging (`astream_log`, `astream_events`).

### Standard Interface

- **stream**: Stream back chunks of the response.
- **invoke**: Call the chain on an input.
- **batch**: Call the chain on a list of inputs.

Corresponding async methods:

- **astream**: Async stream back chunks of the response.
- **ainvoke**: Async call the chain on an input.
- **abatch**: Async call the chain on a list of inputs.
- **astream_log**: Stream back intermediate steps as they happen.
- **astream_events**: Beta feature to stream events as they occur in the chain.

### Input and Output Types

The input and output types vary by component:

| Component     | Input Type                                      | Output Type                 |
| ------------- | ----------------------------------------------- | --------------------------- |
| Prompt        | Dictionary                                      | PromptValue                 |
| ChatModel     | Single string, list of chat messages, or PromptValue | ChatMessage             |
| LLM           | Single string, list of chat messages, or PromptValue | String                   |
| OutputParser  | The output of an LLM or ChatModel               | Depends on the parser        |
| Retriever     | Single string                                   | List of Documents            |
| Tool          | Single string or dictionary                     | Depends on the tool          |

All runnables expose input and output schemas for inspection:

- **input_schema**: Auto-generated input Pydantic model.
- **output_schema**: Auto-generated output Pydantic model.

## Components

LangChain provides standard, extendable interfaces and external integrations for various components used in building LLM applications.

### Chat Models

Chat models use sequences of messages as inputs and return chat messages as outputs. These models support distinct roles for conversation messages, such as AI, users, and system instructions. LangChain does not host any chat models but relies on third-party integrations.

### LLMs

LLMs take a string as input and return a string. They are generally older models compared to chat models. LangChain does not host any LLMs, relying on third-party integrations instead.

### Messages

Messages are inputs and outputs for language models. They have a role, content, and response metadata properties, with different message classes for different roles (e.g., `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`).

### Prompt Templates

Prompt templates translate user input and parameters into instructions for an LLM. They can output either strings or messages and include types such as `StringPromptTemplate`, `ChatPromptTemplate`, and `MessagesPlaceholder`.

### Example Selectors

Example selectors dynamically select examples to include in a prompt, improving model performance.

### Output Parsers

Output parsers transform model outputs into structured formats. They support streaming and various output types such as JSON, XML, CSV, and more.

### Chat History

Chat history keeps track of past messages in a conversation, enabling more context-aware responses.

### Documents and Document Loaders

Documents contain information, and document loaders are used to load Document objects from various data sources.

### Text Splitters

Text splitters divide long documents into smaller chunks for processing. LangChain offers various types of text splitters, including recursive, HTML, Markdown, and more.

### Embedding Models

Embedding models create vector representations of text for use in natural language search and context retrieval.

### Vector Stores

Vector stores handle storing and searching over unstructured data by embedding it and performing vector searches.

### Retrievers

Retrievers return documents based on unstructured queries. They can be created from vector stores or other data sources.

### Key-value Stores

Key-value stores manage storage of arbitrary data, useful for techniques like indexing and caching embeddings.

### Tools

Tools are utilities that models can call during execution. They consist of a name, description, JSON schema for inputs, and a function. Tools can be grouped into toolkits for specific tasks.

### Agents

Agents are systems that use LLMs to determine actions and inputs. LangGraph extends LangChain's agent concept for more customized and controllable agents.

### Callbacks

Callbacks allow hooking into various stages of an LLM application, useful for logging, monitoring, and streaming.

## Techniques

### Streaming

Streaming allows for intermediate results to be shown before the final response is ready. LangChain supports streaming through `.stream()`, `.astream()`, `.astream_events()`, and callbacks.

### Tokens

Tokens are the basic units that language models use to process text. Understanding tokenization is important for optimizing model performance.

### Function/Tool Calling

Function or tool calling allows a model to generate structured output by invoking user-defined tools. This is useful for scenarios requiring structured output or interaction with external APIs.

### Structured Output

Structured output constrains LLM responses to a specific format, such as JSON, YAML, or custom schemas. LangChain supports structured output through methods like `.with_structured_output()`, raw prompting, JSON mode, and tool calling.

### Retrieval

Retrieval augments LLMs by providing relevant information, improving response accuracy. Techniques include query translation, routing, query construction, indexing, post-processing, and generation.

### Text Splitting

LangChain offers multiple text splitters to divide long documents into manageable chunks, improving processing efficiency.

### Evaluation

Evaluation ensures LLM applications meet quality standards. LangSmith offers tools for tracing, annotating, and running evaluations over time.

### Tracing

Tracing tracks the steps in an application from input to output, helping diagnose issues and improve reliability.

For more information on any of these topics, refer to the relevant how-to guides and documentation within the LangChain ecosystem.
