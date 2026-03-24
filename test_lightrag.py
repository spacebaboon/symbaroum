"""
Test LightRAG with Ollama on a small sample of Symbaroum content.
Tests entity extraction quality before committing to full indexing.
"""
import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import ollama as ollama_client

load_dotenv()

WORKING_DIR = "./index/lightrag_test"
os.makedirs(WORKING_DIR, exist_ok=True)

# Sample text - known problem areas from our testing
SAMPLE_TEXT = """
steadfast
Mental disciplines and stratagems for resisting improper influences have been developed among the ranks of the Templars and Black Cloaks. Also among Mystics there are individuals who study the art of resistance. The character has a mind as hard as steel, inspiring it to fight on even against insurmountable odds.

Novice
Reaction. The character can make a second attempt to succeed with a Strong or Resolute test when trying to break an ongoing physical effect: traps, poisons or alchemical effects.

Adept
Reaction. The character is unshakable, and can make a second attempt to shrug off ongoing powers that affects either its will or senses.

Master
Reaction. The psyche of the character strikes back against anyone who tries to affect it. Whenever the character is the victim of a mental attack that fails, the attacker suffers 1D6 damage that ignores Armor.

confusion
The Mystic's understanding of the labyrinth of the senses can make an enemy get lost inside its own mind. Confused enemies become paralyzed or lose the ability to distinguish friend from foe. The Confusion is ongoing until the Mystic fails a Resolute test.

scene 8-2 : final battle
If the player characters do not speak up for surrendering the pathfinders, Argasto decides to fight for them. In that case, combat with the elves is inevitable. The elves are dangerous. Godrai is a summer elf in full bloom and Saran-Ri a young summer elf skilled in battle.

Godrai, elf of late summer
Abilities: Acrobatics (master), Alchemy (novice), Marksman (adept), Poisoner (adept), Quick Draw (novice), Ritualist (novice, Turn Weather).
Weapons: Bow 5, Sword 4, both with poison.

Saran-Ri, elf of early summer
The shapeshifting elf comes rushing through the snow in his beamon form.
Abilities: Iron Fist (novice), Natural Warrior (novice), Shapeshift (adept).
Traits: Long-lived. As beamon: Armored (I), Natural weapon (I).
"""

async def nomic_embed(texts: list[str]) -> np.ndarray:
    """Direct Ollama embedding bypassing LightRAG's hardcoded 1024 dimension."""
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11436")
    data = await client.embed(
        model="nomic-embed-text",
        input=texts,
        keep_alive="1h",  # reset keepalive on every call
    )
    return np.array(data["embeddings"])

async def ollama_chat_nothink(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Custom LLM function using chat API with think=False."""
    kwargs.pop("hashing_kv", None)
    kwargs.pop("max_tokens", None)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    client = ollama_client.AsyncClient(host="http://127.0.0.1:11434")
    response = await client.chat(
        model="qwen3:14b",
        messages=messages,
        think=False,
        options={"num_ctx": 16384},
    )
    return response.message.content


async def main():
    print("Initialising LightRAG with Ollama...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_chat_nothink,  # custom function, not ollama_model_complete
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=nomic_embed,
        ),
        rerank_model_func=None,
        addon_params={
            "entity_types": [
                "Character",
                "Creature",
                "Ability",
                "Condition",
                "Attribute",
                "Faction",
                "Location",
                "Artifact",
                "Trait",
                "Adventure",
                "Rule",
            ]
        },
    )
    
    await rag.initialize_storages()  # Add this line
    
    print("Inserting sample text...")

    await rag.ainsert(SAMPLE_TEXT)
    print("Insertion complete. Testing queries...\n")

    queries = [
        ("mix", "What is the Steadfast ability?"),
        ("mix", "Who are Godrai and Saran-Ri?"),
        ("mix", "What abilities and conditions are related to Resolute?"),
        ("mix", "Can Steadfast help against the Confused condition?"),
    ]

    for mode, query in queries:
        print(f"=== [{mode}] {query} ===")
        result = await rag.aquery(query, param=QueryParam(mode=mode))
        print(result[:500])
        print()

if __name__ == "__main__":
    asyncio.run(main())
