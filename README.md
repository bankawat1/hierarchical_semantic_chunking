# Hierarchical Semantic Chunking for LLMs <WIP>

**Objective:**
The entire purpose of this repository is to develop an algorithm that can intelligently create meaningful chunks of the text before sending to a limited token-size LLM.

![ppt](https://github.com/bankawat1/hierarchical_semantic_chunking/assets/19544675/09c26ad5-aac7-4064-bb09-b9ee0edfdf47)

Note: This version of H-Semantic chunking is focused on transcripts (i.e. conversation between two person)

**Pros:**
It uses an advanced approach to find the breakpoints which is inspired from an off-the-shelf Langchain semantic chunking package (https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/).
This is an extends the above algorithm.

**Model:** SentenceTransformer

**Input**: Text -  split into sentences.


**Output**: Indices of the sentences which represent the breakpoints for creating chunks.

**Time complexity:**

**Space complexity:**





