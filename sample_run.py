import logging
import logging.config
import yaml

def setup_logging(default_path="logging.yaml", default_level=logging.INFO):
    """Setup logging configuration"""
    try:
        with open(default_path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error in Logging Configuration. Using default configs. {e}")
        logging.basicConfig(level=default_level)

def main():
    setup_logging()
    from semantic_chunking import invoke_semantic_chunking

    with open("eval_set/eval_input_bankagent_conversation.txt", "r") as file:
        text = file.read()
    results = invoke_semantic_chunking(text)
    print(results)


if __name__ == "__main__":
    main()
