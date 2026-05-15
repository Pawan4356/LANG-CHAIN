## Complete Flow

This folder contains a small Retrieval-Augmented Generation (RAG) app built on YouTube transcripts.

### What the app does

1. It picks one YouTube video id from [`video_ids.py`](11_YTChatbot/video_ids.py).
2. It downloads the transcript with `youtube_transcript_api`.
3. It splits the transcript into smaller chunks.
4. It converts those chunks into embeddings with a Hugging Face embedding model.
5. It stores the embeddings in a FAISS vector store.
6. It retrieves the most relevant chunks for a user question.
7. It sends the retrieved context plus the question to a chat model.
8. It prints the generated answer in the terminal.

### Runtime flow

When you run `python main.py`, the execution order is:

`main.py` -> `indexing.py` -> `retriever.py` -> `augmentation.py` -> `generation.py`

In practice, the request flow is:

1. `main()` builds the vector store.
2. The user types a question.
3. `answer_with_augmentation()` retrieves the most relevant transcript chunks.
4. `generate_answer()` prompts the model with those chunks and the question.
5. The answer is printed back to the terminal.

### Example

```
>>> Summarize the video.
```

```
Answer:

Based solely on the provided transcript, the video segment features David Goggins discussing his philosophy on hard work and authenticity:

1.  **The Ugly Reality of Hard Work:** Goggins emphatically states that true hard work is not glamorous, motivating, or documentary-worthy. He describes it as "ugly," like a "train wreck," a "nightmare," or being "stuck in a fucking dungeon."
2.  **The "Stick" Motivation:** His drive comes entirely from the "stick" – confronting and avoiding weakness, failure, and being a "loser" or a "piece of shit." There is no "carrot" (positive reward) in his approach.
3.  **Rigorous Discipline & Preparation:** His actions are never spontaneous ("on the fly"). Everything is pre-planned and consistent ("It's always the same thing"). He maintains intense discipline, like studying for 4+ hours daily even after passing a test, just to retain knowledge.
4.  **Authenticity & Truth:** Goggins values the ability to look another person (specifically, another man) in the eye and speak absolute, earned truth based on real work. He finds immense satisfaction in this authenticity, contrasting it with people who present a false image of who they want to be.
5.  **Frustration & Passion:** He expresses deep frustration at being misunderstood. His passionate, profanity-laden speech stems from the intensity of his lived experience and the difficulty of conveying the sheer magnitude of the work required ("What built this guy?").
6.  **Individual Process:** While sharing his own highly structured approach, he acknowledges that the process of translating mental struggle into action is highly individual ("You do it your way").

**In short:** The segment shows David Goggins explaining that real hard work is brutally difficult and unglamorous ("all stick, no carrot"), driven by confronting weakness, requiring extreme pre-planned discipline, and resulting in hard-earned authenticity that he values above all else. He expresses frustration at conveying this reality and emphasizes the process is personal.
```

### Notes

- The app expects the required API keys and Hugging Face access to be configured in your environment.
- If a transcript is unavailable, the indexing step cannot build the vector store for that video.
- The current setup uses the first id in `video_ids.py`, but you can extend it to loop over multiple videos.
