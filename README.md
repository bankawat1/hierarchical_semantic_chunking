# Hierarchical Semantic Chunking for LLMs <WIP>
![ppt](https://github.com/bankawat1/hierarchical_semantic_chunking/assets/19544675/09c26ad5-aac7-4064-bb09-b9ee0edfdf47)

**Objective:**
Hierarchical Semantic Chunking extends the algorithm offered by Lang-chain's Semantic chunking algorithm . The purpose of this repository is to develop an algorithm that can intelligently create meaningful chunks of a document, later these chunks can be used in two ways:
1. Splitting a very long document into multiple chunks which can be later sent over to a limited token-size LLM iteratively.
2. Generate longer descriptive summaries - which would not be possible even with a very large context window LLM since the general behavior of an LLM is to summarize a document/transcript into a short concise paragraph making it drop a lot of useful information. Hence, This is achieved by concatenating the generated summaries for individual chunks. Remember you may have to do post-processing to aggregate the results into a meaningful final summary.

Assumption: It is assumed that the document is longer than 500 sentences/utterances. There is a clear context switching which does not go back-forth with current or previous chunk's context. That means that all the chunks can be extracted separately without any dependency on each other.



Note: This version of H-Semantic chunking is focused on transcripts (i.e. a conversation between two persons)

**Advantage:**
It uses an advanced approach to find the breakpoints which extends the off-the-shelf Langchain semantic chunking package (https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/).

**Usage:** 
Call invoke_semantic_chunking() function passing document as text to it.

**Model:** SentenceTransformer (all-MiniLM-L6). Weight file to be downloaded before using this library.

**Input**: Any text document that is long enough to be split into chunks. Recommended to have anything greater than 300 sentences.

**Output**: Indices of the sentences which represent the breakpoints for creating chunks.

**Time complexity:** TBD

**Space complexity:** TBD





