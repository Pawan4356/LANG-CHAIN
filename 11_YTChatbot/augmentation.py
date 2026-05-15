from retriever import get_retriever
from generation import generate_answer


def format_docs(retrieved_docs: list):

    return "\n".join(getattr(d, 'page_content', str(d)) for d in retrieved_docs)


def answer_with_augmentation(vector_store, question, k=5):

    retriever = get_retriever(vector_store)
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)
    
    return generate_answer(context, question)