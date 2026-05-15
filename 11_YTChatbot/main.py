from indexing import get_vector_store
from augmentation import answer_with_augmentation


def main():

    vector_store = get_vector_store()
    print("Enter a question about the video transcript (empty to quit):")
    
    while True:
        try:
            question = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question.strip():
            print("Goodbye.")
            break

        answer = answer_with_augmentation(vector_store, question)
        print("\nAnswer:\n", answer.content)


if __name__ == '__main__':

    main()