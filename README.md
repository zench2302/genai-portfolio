# 🧠 Jia Jia – GenAI/ML Project Portfolio

Welcome! This is a collection of my recent projects in Generative AI, built during my MSc Data Science at LSE and reflecting my transition into AI product development. My focus has been on LLM-based recommendation systems, retrieval-augmented generation (RAG), and prompt engineering with open-source tools.


### 📚 Table of Contents

## 🧠 GenAI Project Portfolio
  - 📌 [LLM-Based Product Recommender](#-1-llm-based-product-recommender)
  - 📌 [Twitch Topic Extraction & Sentiment Analysis](#-2-twitch-topic-extraction--sentiment-analysis)
  - 📌 [AI-Powered Travel Planner - Vibego](#-3-ai-powered-travel-planner---vibego)
## ⚙️ Machine Learning Projects
  - 📌 [Legacy Donation Analysis](#-4-legacy-donation-analysis)

- 🧑‍🎨 [🎨 Creative Tech](#creative-tech)
  - 🎧 [Live Coding with Strudel](#-live-coding-with-strudel-exploring-code-based-music-interaction)

---

## 📌 1. [LLM-Based Product Recommender](https://github.com/zench2302/llm-recommender)

### Description: 

Developed an end-to-end recommendation system powered by LLMs for Amazon product reviews. The system embeds product metadata and review content using Flan-T5 and MiniLM, then computes similarity via Faiss for Top-5 recommendations.

### Key Technologies:  
- Nous-Hermes-2 (Mistral) for generating user profiles from review history
- Flan-T5 for generating ad-style recommendation reasons  
- BGE (BAAI) and MiniLM embeddings for product and user vectorization  
- FAISS for approximate nearest neighbor (ANN) vector search and candidate retrieval  
- Prompt engineering for review summarization and recommendation reasoning

### System and Recommendation Pipeline

<img src="assets/LLM_pipeline.png" width="640">

### Prompts Engineering
**(1) User Profiling Prompt (Mistral)**
```text
You are a professional shopping assistant.

Analyze the following user reviews and summarize their preferences.

Reviews:
""" <user reviews> """

Return JSON with:
- "preferred_products"
- "liked_features"
- "dislikes"
- "potential_interests"
```
<details>
<summary>🔧 Full Python Implementation (click to expand)</summary>
  
```python
def generate_user_profile(user_reviews):
    prompt = f"""
You are a professional shopping assistant.

Analyze the following user reviews and summarize their preferences.

Reviews:
\"\"\"{user_reviews[:Config.MAX_REVIEW_LENGTH]}\"\"\"

Return JSON with:
- "preferred_products"
- "liked_features"
- "dislikes"
- "potential_interests"
"""
    inputs = profile_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(profile_model.device)
    outputs = profile_model.generate(
        **inputs,
        max_new_tokens=Config.MAX_NEW_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        repetition_penalty=Config.REPETITION_PENALTY,
        pad_token_id=profile_tokenizer.eos_token_id
    )
    raw_output = profile_tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_str = raw_output[raw_output.find("{"):raw_output.rfind("}")+1]
    return json.loads(json_str)
```
</details>

**(2) Ad-Slogan Prompt summary (Flan-T5)**
```text
You are an expert e-commerce copywriter creating unique, playful ad slogans.

Product:
- Title: ...
- Description: ...
- Rating: ...

User:
- Likes: ...
- Dislikes: ...

Your task:
- Write ONE catchy slogan (≤12 words)
- Avoid repeating product name or brand
- Use playful, emotional, or surprising tone
```

<details>
<summary>🔧 Full Python Implementation (click to expand)</summary>
  
```python
def build_ad_prompt(product_info, user_profile):
    title = product_info.get('title', 'Unknown Product')
    description = " ".join(product_info.get('description', [])) if isinstance(product_info.get('description'), list) else product_info.get('description', '')
    details = product_info.get('details', '')
    avg_rating = product_info.get('average_rating', 0)
    preferred_products = ", ".join(user_profile.get('preferred_products', []))
    liked_features = ", ".join(user_profile.get('liked_features', []))
    dislikes = ", ".join(user_profile.get('dislikes', []))
    potential_interests = ", ".join(user_profile.get('potential_interests', []))

    return f"""
You are an expert e-commerce copywriter creating unique, playful ad slogans.

Product:
- Title: {title}
- Description: {description}
- Details: {details}
- Average rating: {avg_rating}

User:
- Preferred products: {preferred_products}
- Likes: {liked_features}
- Dislikes: {dislikes}
- Interests: {potential_interests}

Your task:
- Write ONE catchy slogan (≤12 words) that excites this user.
- Match the product type (nails, hair, skincare, lashes, tools, etc.).
- Highlight the user’s likes, avoid their dislikes.
- Use playful, emotional, or surprising language.
- Do NOT copy product name, specs, or brand.
- If irrelevant, return only: SKIP.

Output:
"""
```
</details>

💡 Full prompt templates available in [PROMPTS.md](PROMPTS.md)


### Selected Outputs

<img src="assets/LLM_recitem3.png" width="720">


### Evaluation Snapshot
The following metrics reflect the system's performance on non-cold-start users only!

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
- Developed predictive models for alumni donation likelihood using structured engagement datasets
- Evaluated Gradient-boosted models (**XGBoost, LightGBM, CatBoost**) alongside logistic regression, random forest
- Fine-tuned models via Grid & Random Search algorithms, implemented model stacking for optimal performance
- Delivered data preprocessing, feature engineering, and model evaluation using Google Colab

---
## 📌 4. [Legacy Donation Analysis](https://github.com/zench2302/Capstone_LegacyAnalysisProspection)
**Description:**  
Study the major influential features in making legacy donation for alumni.

**Key Technologies:**  
- gradient-based models: GBDT, XGBoost, LightGBM, CatBoost
- traditional models: logistic regression with/without L1
- Unsupervised learning: Factor Analysis, Cluster Analsis.
- feature engineering

**Highlights:**  
- Modular prompt design to adapt to user preferences  
- Route optimization based on timing and transport  
- Designed for deployment as a lightweight web app

---

# Creative Tech
### 1. 🎧 Live Coding with Strudel: Exploring Code-Based Music Interaction
London Data Week 2025 – Algorave Workshop @ King’s Institute for AI

In this hands-on workshop, I explored the intersection of code, rhythm, and creative expression using Strudel — a browser-based live coding tool for generative music. Participants learned to construct musical patterns through time-based code snippets and were invited to perform their compositions in an open stage format.

While I didn’t create a full performance piece, I gained direct experience in:

- Writing loop-based musical patterns using declarative syntax

- Understanding how code structure translates to rhythm and timing

- Experiencing the dynamics of collaborative, real-time generative systems

The session also prompted reflection on the human-computer interaction aspect of creative coding: how intuitive (or not) such tools are for newcomers, and how live-coded performances affect audience perception. This exploration connects back to my broader interest in generative AI and interactive systems design, where usability, expressiveness, and emotional impact must be balanced.


## 📫 Contact

To learn more, visit my main profile: [github.com/zench2302](https://github.com/zench2302)  
Or connect on LinkedIn: [linkedin.com/in/jia-jia-7a73359a](https://linkedin.com/in/jia-jia-7a73359a)
