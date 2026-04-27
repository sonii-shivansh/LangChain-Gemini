from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def audit_chunks(chunks):
    """Analyze chunk quality before embedding"""
    
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    print("=" * 50)
    print("CHUNK AUDIT REPORT")
    print("=" * 50)
    print(f"Total chunks        : {len(chunks)}")
    print(f"Average chunk size  : {sum(sizes) / len(sizes):.0f} chars")
    print(f"Smallest chunk      : {min(sizes)} chars")
    print(f"Largest chunk       : {max(sizes)} chars")
    print(f"Chunks under 50 chars: {sum(1 for s in sizes if s < 50)}  ← potential junk")
    print(f"Chunks over 2000    : {sum(1 for s in sizes if s > 2000)}  ← too large")
    print()

    print("SAMPLE CHUNKS:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}]")
        print(f"  Size    : {len(chunk.page_content)} chars")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Preview : {chunk.page_content[:80]}...")


# Usage
with open("sample.txt", "w") as f:
    f.write("""Machine learning transforms how we process data.
Neural networks learn patterns from examples.

Deep learning uses multiple layers of neurons.
It excels at image and speech recognition.

Natural language processing handles text.
It powers chatbots and translation systems.

Reinforcement learning trains through rewards.
Agents learn optimal strategies over time.""")

loader = TextLoader("sample.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

audit_chunks(chunks)