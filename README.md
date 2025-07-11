# 🧠 Jia Jia – GenAI Project Portfolio

This is a collection of my recent projects in Generative AI, built during my MSc Data Science at LSE and reflecting my transition into AI product development. My focus has been on LLM-based recommendation systems, retrieval-augmented generation (RAG), and prompt engineering with open-source tools.

---

## 📌 1. [LLM-Based Product Recommender](https://github.com/zench2302/llm-recommender)

#### Description:  
Developed an end-to-end recommendation system powered by LLMs for Amazon product reviews. The system embeds product metadata and review content using Flan-T5 and MiniLM, then computes similarity via Faiss for real-time Top-N recommendations.

#### Key Technologies:
- Flan-T5 for summarization and structure refinement  
- MiniLM embeddings  
- Faiss vector search (ANN)  
- Prompt-based review reasoning  
- Scikit-learn, Pandas, Streamlit (prototype UI)

#### Highlights:  
- Indexed 50K+ reviews with semantic search  
- Designed prompt chain for category-aware recommendations  
- Optimized for fast response on local CPU setup

#### System and Recommendation Pipeline 

<img src="assets/LLM_pipeline.png" width="640">


#### Selected Outputs
<img src="assets/LLM_recitem3.png" width="720">


#### Evaluation Snapshot

| Metric                  | Score (non-cold start) |
|-------------------------|------------------------|
| Semantic Match (CosSim) | 71.8%                  |
| Ad Diversity            | 82.6%                  |
| Avg. Product Rating     | 4.31                   |

---

## 📌 2. [Twitch Topic Extraction & Sentiment Analysis](https://github.com/zench2302/Twitch_stream_analytics)

**Description:**  
Built a real-time system to extract dominant topics and sentiment streams from Twitch chat logs. Designed to detect mood shifts and trending discussion in live streams.

**Key Technologies:**  
- BERTopic + UMAP + HDBSCAN for topic modeling  
- Vader + GPT-based sentiment refinement  
- WebSocket-based data ingestion  
- Tokenized message streams with temporal segmentation

**Highlights:**  
- Capable of handling 10K+ chat lines per minute  
- Visual clustering of evolving discussion topics  
- Used GPT-4 for refining topic labeling and summary

---

## 📌 3. [AI-Powered Travel Planner - Vibego](https://github.com/zench2302/vibego2)

**Description:**  
An AI-assisted travel itinerary generator that combines OpenAI's GPT API with Google Maps data to produce city-specific, time-aware plans.

**Key Technologies:**  
- OpenAI (GPT-4) for summarization and suggestion  
- Google Maps API for location + travel time data  
- Prompt chaining for adaptive personalization  
- Firebase for session handling

**Highlights:**  
- Modular prompt design to adapt to user preferences  
- Route optimization based on timing and transport  
- Designed for deployment as a lightweight web app

---

## 📫 Contact

To learn more, visit my main profile: [github.com/zench2302](https://github.com/zench2302)  
Or connect on LinkedIn: [linkedin.com/in/jia-jia-7a73359a](https://linkedin.com/in/jia-jia-7a73359a)
