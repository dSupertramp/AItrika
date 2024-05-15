# Large language models (LLMs)

AItrika supports several Large Language Models (LLMs), that are used to extract informations like results, introductions and bibliography / references.
In order to use LLMs, you need respective keys.
Open-weights models uses by default HuggingFace's emdebbing model called `BAAI/bge-small-en-v1.5`.

## OpenAI

In order to use OpenAI models, you need to have an OpenAI API key.
You can get one [here](https://platform.openai.com/account/api-keys).

Once you have an API key,you can use it like this:

```python
llm = OpenAILLM(
    model_endpoint=model_endpoint,
    documents=documents,
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

## HuggingFace

In order to use HuggingFace models, you need to have a HuggingFace API Key and the model endpoint (default is `microsoft/Phi-3-mini-4k-instruct`).

Once you have an API key and the model endpoint, you can use it like this:

```python
model_endpoint = "microsoft/Phi-3-mini-4k-instruct"
llm = HuggingFaceLLM(
    model_endpoint=model_endpoint,
    documents=documents,
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
```

## Groq

In order to use Groq models, you need to have a Groq API Key and the name of the model (default is `llama3-70b-8192`)
You can get one [here](https://platform.openai.com/account/api-keys).

Once you have an API key and the model endpoint, you can use it like this:

```python
model_name = "llama3-70b-8192"
llm = GroqLLM(
    model_name=model_name,
    documents=documents,
    api_key=os.getenv("GROQ_API_KEY"),
)
```
