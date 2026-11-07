"""
One-time ingestion script. Run this ONCE to populate ChromaDB with all knowledge.
Usage: python knowledge_base/ingest.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL,
    CHROMA_COLLECTION_SPIRITUAL, CHROMA_COLLECTION_CBT,
    CHROMA_COLLECTION_CULTURAL, CHROMA_COLLECTION_PSYCHOEDU,
    CHROMA_COLLECTION_GUIDED,
    DATA_DIR
)

DATA_PATH = Path(__file__).parent / "data"


def get_embeddings():
    print(f"  Loading embedding model: {EMBEDDING_MODEL} via Ollama...")
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )


def embed_and_add(collection, documents: list[str], metadatas: list[dict],
                  ids: list[str], embeddings_model):
    print(f"    Embedding {len(documents)} documents...")
    embeddings = embeddings_model.embed_documents(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"    Added {len(documents)} documents to collection.")


def ingest_spiritual(client, embeddings):
    print("\n[1/4] Ingesting Bhagavad Gita verses...")
    try:
        client.delete_collection(CHROMA_COLLECTION_SPIRITUAL)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION_SPIRITUAL)

    data_file = DATA_PATH / "gita_verses.json"
    verses = json.loads(data_file.read_text(encoding="utf-8-sig"))

    documents, metadatas, ids = [], [], []
    for v in verses:
        # Build rich document text for better semantic retrieval
        doc_text = (
            f"Bhagavad Gita Chapter {v['chapter']}, Verse {v['verse']}: {v['text']} "
            f"Concept: {v['concept']}. Psychology: {v['psychology']}. "
            f"Keywords: {v['keywords']}"
        )
        documents.append(doc_text)
        metadatas.append({
            "id": v["id"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "concept": v["concept"],
            "psychology": v["psychology"],
            "culture_tag": v.get("culture_tags", ["universal"])[0] if v.get("culture_tags") else "universal",
            "themes": ",".join(v.get("theme", [])),
            "target_emotions": ",".join(v.get("target_emotions", [])),
            "source": "Bhagavad Gita",
        })
        ids.append(v["id"])

    embed_and_add(collection, documents, metadatas, ids, embeddings)
    print(f"  [✓] Spiritual collection: {len(verses)} verses ingested.")


def ingest_cbt(client, embeddings):
    print("\n[2/4] Ingesting CBT techniques...")
    try:
        client.delete_collection(CHROMA_COLLECTION_CBT)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION_CBT)

    data_file = DATA_PATH / "cbt_techniques.json"
    techniques = json.loads(data_file.read_text(encoding="utf-8-sig"))

    documents, metadatas, ids = [], [], []
    for t in techniques:
        steps_text = " -> ".join(t.get("steps", []))
        doc_text = (
            f"CBT Technique: {t['name']}. Category: {t['category']}. "
            f"Description: {t['description']} "
            f"Steps: {steps_text}. "
            f"Keywords: {t['keywords']}"
        )
        documents.append(doc_text)
        metadatas.append({
            "id": t["id"],
            "name": t["name"],
            "category": t["category"],
            "target_disorders": ",".join(t.get("target_disorders", [])),
            "target_emotions": ",".join(t.get("target_emotions", [])),
            "source": "CBT Knowledge Base",
        })
        ids.append(t["id"])

    embed_and_add(collection, documents, metadatas, ids, embeddings)
    print(f"  [✓] CBT collection: {len(techniques)} techniques ingested.")


def ingest_cultural(client, embeddings):
    print("\n[3/4] Ingesting cultural practices...")
    try:
        client.delete_collection(CHROMA_COLLECTION_CULTURAL)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION_CULTURAL)

    data_file = DATA_PATH / "cultural_practices.json"
    practices = json.loads(data_file.read_text(encoding="utf-8-sig"))

    documents, metadatas, ids = [], [], []
    for p in practices:
        doc_text = (
            f"Cultural Practice ({p['culture']}): {p['practice']}. "
            f"{p['description']} "
            f"Keywords: {p['keywords']}"
        )
        documents.append(doc_text)
        metadatas.append({
            "id": p["id"],
            "practice": p["practice"],
            "culture": p["culture"],
            "is_collectivist": str(p.get("is_collectivist", False)),
            "applicable_emotions": ",".join(p.get("applicable_emotions", [])),
            "source": "Cultural Knowledge Base",
        })
        ids.append(p["id"])

    embed_and_add(collection, documents, metadatas, ids, embeddings)
    print(f"  [✓] Cultural collection: {len(practices)} practices ingested.")


def ingest_psychoedu(client, embeddings):
    print("\n[4/4] Ingesting psychoeducation content...")
    try:
        client.delete_collection(CHROMA_COLLECTION_PSYCHOEDU)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION_PSYCHOEDU)

    # Generate psychoeducation summaries from CBT data as base
    data_file = DATA_PATH / "cbt_techniques.json"
    techniques = json.loads(data_file.read_text(encoding="utf-8-sig"))

    psychoedu_content = [
        {"id": "PE_001", "title": "What is Anxiety?", "content": "Anxiety is a natural emotional response to perceived threats or uncertainty. It involves physiological arousal (racing heart, shallow breathing, muscle tension), cognitive symptoms (worry, catastrophic thinking, difficulty concentrating), and behavioral responses (avoidance, reassurance-seeking). Anxiety becomes problematic when it is excessive, persistent, and interferes with daily functioning. Common forms include Generalized Anxiety Disorder, Social Anxiety, Panic Disorder, and specific phobias. Keywords: anxiety worry fear panic nervous uncertain", "category": "psychoeducation", "condition": "anxiety"},
        {"id": "PE_002", "title": "Understanding Depression", "content": "Depression is more than sadness - it is a pervasive low mood that affects thoughts, feelings, behavior, and physical health. Core symptoms include persistent low mood, loss of interest (anhedonia), fatigue, sleep disturbances, appetite changes, difficulty concentrating, and feelings of worthlessness. Depression is not a character weakness or choice. It involves neurobiological changes and responds well to therapy and lifestyle interventions. Keywords: depression sadness low mood anhedonia hopeless worthless sleep appetite", "category": "psychoeducation", "condition": "depression"},
        {"id": "PE_003", "title": "Burnout vs. Stress", "content": "Burnout is a state of chronic stress that leads to physical and emotional exhaustion, cynicism, and feelings of ineffectiveness. Unlike regular stress (too much pressure), burnout involves feeling empty, depleted, and unable to meet constant demands. Key signs: emotional exhaustion, depersonalization, reduced sense of accomplishment. Recovery involves rest, boundary setting, values realignment, and professional support. Keywords: burnout stress exhaustion depletion overwhelm work academic pressure cynicism", "category": "psychoeducation", "condition": "burnout"},
        {"id": "PE_004", "title": "Grief and Loss", "content": "Grief is the natural response to loss - of a person, relationship, health, identity, or way of life. It is not linear. The Kubler-Ross stages (denial, anger, bargaining, depression, acceptance) describe possible experiences, not a required sequence. Complicated grief occurs when grief is prolonged, intense, and functionally impairing. Healthy grieving involves allowing emotions, maintaining connections, finding meaning, and gradually re-engaging with life. Keywords: grief loss death mourning bereavement acceptance healing sadness", "category": "psychoeducation", "condition": "grief"},
        {"id": "PE_005", "title": "Academic and Performance Pressure", "content": "Academic stress involves anxiety about exams, grades, parental expectations, peer competition, and future uncertainty. It can manifest as test anxiety, perfectionism, procrastination, burnout, and imposter syndrome. Common in South Asian students due to collectivist family expectations and high-stakes examination culture. Helpful strategies include breaking tasks, self-compassion, process-focus, sleep hygiene, and social support. Keywords: exam stress academic pressure student performance anxiety perfectionism failure grades", "category": "psychoeducation", "condition": "academic_stress"},
    ]

    documents, metadatas, ids = [], [], []
    for p in psychoedu_content:
        documents.append(p["content"])
        metadatas.append({
            "id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "condition": p["condition"],
            "source": "Psychoeducation Database",
        })
        ids.append(p["id"])

    embed_and_add(collection, documents, metadatas, ids, embeddings)
    print(f"  [✓] Psychoeducation collection: {len(psychoedu_content)} articles ingested.")


def ingest_guided_responses(client, embeddings):
    print("\n[5/5] Ingesting therapeutic guided responses (exemplars)...")
    try:
        client.delete_collection(CHROMA_COLLECTION_GUIDED)
    except Exception:
        pass
    collection = client.create_collection(CHROMA_COLLECTION_GUIDED)

    data_file = DATA_PATH / "guided_responses.json"
    if not data_file.exists():
        print("  [!] guided_responses.json not found, skipping.")
        return

    items = json.loads(data_file.read_text(encoding="utf-8-sig"))
    documents, metadatas, ids = [], [], []
    for item in items:
        doc_text = item.get("rag_text") or (
            f"Topic: {item.get('category')}. "
            f"User Concern: {item.get('user_prompt')}. "
            f"Support Strategy: {item.get('support_strategy')}. "
            f"Guided Response: {item.get('guided_response')} "
            f"Keywords: {item.get('keywords', '')}"
        )
        documents.append(doc_text)
        metadatas.append({
            "id": item["id"],
            "category": item.get("category", "General"),
            "support_strategy": item.get("support_strategy", ""),
            "user_prompt": item.get("user_prompt", "")[:300],
            "guided_response": item.get("guided_response", "")[:800],
            "target_emotions": ",".join(item.get("target_emotions", [])),
            "source": "Therapeutic Guided Responses",
        })
        ids.append(item["id"])

    embed_and_add(collection, documents, metadatas, ids, embeddings)
    print(f"  [✓] Guided responses collection: {len(items)} exemplars ingested.")


def main():
    print("=" * 60)
    print("  Mental Health AI - Knowledge Base Ingestion")
    print("=" * 60)
    print(f"  ChromaDB path: {CHROMA_DIR}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print()

    embeddings = get_embeddings()
    client = get_client()

    ingest_spiritual(client, embeddings)
    ingest_cbt(client, embeddings)
    ingest_cultural(client, embeddings)
    ingest_psychoedu(client, embeddings)
    ingest_guided_responses(client, embeddings)

    print("\n" + "=" * 60)
    print("  [✓] Knowledge base ingestion complete!")
    print("  You can now run: streamlit run streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
