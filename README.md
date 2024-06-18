# Hierarchical Semantic Chunking

**Objective:**
The entire purpose of this repository is to develop an algorithm that can intelligently create meaningful chunks of the text before sending to a limited token-size LLM.

**Pros:**
It uses an advanced approach to find the breakpoints than what you get from the off-the-shelf Langchain semantic chunking package (https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/).

**Model:** SentenceTransformer

**Input**: Text -  split into sentences.

**Output**: Indices of the sentences which represent the breakpoints for creating chunks.

**Time complexity:**

**Space complexity:**



