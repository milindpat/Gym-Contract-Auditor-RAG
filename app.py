import os
from typing import List, Dict, Any
from rag_pipeline import ContractRAG

SOURCE_PATH = "data/Gym_Membership_Agreement_SourceOfTruth.txt"
OUTPUT_PATH = "outputs/sample_output.txt"


def save_sample_output(path: str, question: str, retrieved: List[Dict[str, Any]], answer: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"QUESTION: {question}\n\n")
        f.write("RETRIEVED:\n")
        for r in retrieved:
            f.write(f"- {r['id']} | score={r['score']:.3f}\n")
        f.write("\nANSWER:\n")
        f.write(answer + "\n\n")


def run_demo(rag: ContractRAG) -> None:
    demo_questions = [
        "How can I cancel my monthly plan and how much notice is required?",
        "What is the freeze policy and what is the fee?",
        "How many guests can I bring and when are guests allowed?",
        "Does the gym accept cryptocurrency payments?",
    ]

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    for question in demo_questions:
        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")

        retrieved = rag.retrieve(question, top_k=4)
        print("\nRETRIEVED:")
        for r in retrieved:
            print(f"- {r['id']} | score={r['score']:.3f}")

        answer = rag.answer(question)
        print("\nANSWER:")
        print(answer)

        save_sample_output(OUTPUT_PATH, question, retrieved, answer)


def interactive_mode(rag: ContractRAG) -> None:
    print("\nGym Contract Auditor (RAG)")
    print("Ask a question about the contract.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() == "exit":
            print("Session ended.")
            break

        retrieved = rag.retrieve(question, top_k=4)

        print("\n" + "=" * 90)
        print(f"QUESTION: {question}")

        print("\nRETRIEVED:")
        for r in retrieved:
            print(f"- {r['id']} | score={r['score']:.3f}")

        print("\nANSWER:")
        print(rag.answer(question))
        print("=" * 90)


def main() -> None:
    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError(f"Source file not found: {SOURCE_PATH}")

    rag = ContractRAG(SOURCE_PATH)
    run_demo(rag)
    interactive_mode(rag)


if __name__ == "__main__":
    main()