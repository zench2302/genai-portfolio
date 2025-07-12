
# 🧠 Prompt Engineering – LLM-Based Product Recommender

This document outlines the prompt engineering strategies used in my recommendation system to support:

- 🧾 User preference extraction from review text  
- 💬 Personalized, ad-style slogan generation  
- 🔍 Output quality control through post-generation filters  

---

## 🎯 1. User Profiling via Mistral (Nous-Hermes-2)

We use a decoder-only LLM (Mistral) to transform a user's review history into a structured JSON object capturing their preferences. This profiling output feeds directly into the recommendation generation process.

### 🔧 Prompt Design

```python
def generate_user_profile(user_reviews):
    prompt = f\"\"\"
You are a professional shopping assistant.

Analyze the following user reviews and summarize their preferences.

Reviews:
\"\"\"{user_reviews[:Config.MAX_REVIEW_LENGTH]}\"\"\"

Return JSON with:
- "preferred_products"
- "liked_features"
- "dislikes"
- "potential_interests"
\"\"\"
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

---

### 📤 Example Output

```json
{
  "preferred_products": ["vitamin C serum", "hydrating toner"],
  "liked_features": ["lightweight texture", "natural ingredients"],
  "dislikes": ["strong fragrance"],
  "potential_interests": ["sensitive skin products"]
}
```

---

## 💡 2. Ad-Style Slogan Generation via Flan-T5

We use a carefully structured prompt to guide Flan-T5 in generating short, expressive slogans for product recommendations. These slogans aim to be emotionally appealing while respecting user preferences.

### 🧾 Prompt Template

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

    return f\"\"\"
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
\"\"\"
```

---

## ✅ Output Filtering Strategy

To ensure diversity, quality, and relevance of generated slogans, we apply:

- **Title overlap filter**: discard if slogan directly duplicates product title  
- **Minimum length & SKIP filter**: discard irrelevant or weak outputs  
- **Semantic duplication filter**: use cosine similarity to avoid near-identical slogans  

```python
def generate_ad_line_strict(prompt, title=None, used_vectors=None):
    inputs = slogan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = slogan_model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=slogan_tokenizer.eos_token_id
    )
    ad_line = slogan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if title and ad_line.lower() in title.lower():
        return None
    if len(ad_line) < 5 or "skip" in ad_line.lower():
        return None

    if used_vectors is not None:
        ad_vec = embed_model.encode(ad_line)
        if any(util.cos_sim(ad_vec, v).item() > 0.8 for v in used_vectors):
            return None
        used_vectors.append(ad_vec)

    return ad_line
```

---

## 📌 Summary

These prompt pipelines form the core of the system's ability to transform user behavior into structured preference data and generate emotionally compelling, personalized recommendations.

Together, they demonstrate:  
- Modular, interpretable prompting strategies  
- Integration of structured inputs into language model generation  
- Output filtering for semantic diversity and quality assurance
