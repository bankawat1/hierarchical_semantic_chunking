def main():
    # print("Hello from hierarchical-semantic-chunking!")
    from semantic_chunking import invoke_semantic_chunking

    with open("eval_set/eval_input_bankagent_conversation.txt", "r") as file:
        text = file.read()
    results = invoke_semantic_chunking(text)
    print(results)


if __name__ == "__main__":
    main()
