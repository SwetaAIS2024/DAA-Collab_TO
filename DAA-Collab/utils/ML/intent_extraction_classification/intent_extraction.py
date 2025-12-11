from typing import List
import re
import os

# Try to load spaCy if available (for non-air-gapped systems)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not available. Using rule-based extraction.")


def user_intent_extraction(user_instruction_prompt: str) -> List[str]:
    """Extract keywords using available method"""
    intent_extraction_result = extraction_logic(user_instruction_prompt)
    return intent_extraction_result


def extraction_logic(instr: str) -> List[str]:
    """
    Hybrid approach: Use spaCy if available, fallback to rule-based
    Works in air-gapped systems without external dependencies
    """
    if SPACY_AVAILABLE:
        return extraction_logic_nlp(instr)
    else:
        return extraction_logic_rule_based(instr)


def extraction_logic_nlp(instr: str) -> List[str]:
    """NLP-based extraction using spaCy (when available)"""
    doc = nlp(instr.lower())
    keywords = []
    
    # Identify negation contexts
    negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'none', "n't", 'without'}
    negated_tokens = set()
    
    # Mark tokens that are negated
    for token in doc:
        if token.text in negation_words or token.dep_ == 'neg':
            # Mark the next 3 tokens as negated (context window)
            for child in token.head.subtree:
                negated_tokens.add(child.i)
    
    # Extract named entities (clean articles)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'DATE', 'TIME', 'FAC', 'EVENT']:
            # Check if entity is in negation context
            if not any(token.i in negated_tokens for token in ent):
                cleaned_ent = ' '.join(word for word in ent.text.split() 
                                      if word not in ['the', 'a', 'an', 'any', 'some'])
                if cleaned_ent:
                    keywords.append(cleaned_ent)
    
    # Extract nouns and verbs (skip negated ones)
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop:
            if (len(token.text) > 2 and 
                token.lemma_ not in negation_words and
                token.i not in negated_tokens):
                keywords.append(token.lemma_)
    
    # Extract noun chunks (skip negated phrases)
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) >= 2:
            # Check if chunk is in negation context
            if not any(token.i in negated_tokens for token in chunk):
                cleaned = ' '.join(word for word in chunk.text.split() 
                                 if word not in ['the', 'a', 'an', 'any', 'some'] 
                                 and word not in negation_words)
                if cleaned:
                    keywords.append(cleaned)
    
    # Remove duplicates and filter substrings
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            if not any(kw in existing and kw != existing for existing in unique_keywords):
                unique_keywords.append(kw)
                seen.add(kw)
    
    return unique_keywords

def extraction_logic_rule_based(instr: str) -> List[str]:
    """
    Rule-based extraction for air-gapped systems
    Uses POS patterns and domain knowledge without external models
    """
    keywords = []
    
    # Domain-specific keywords (minimal baseline)
    domain_keywords = {
        'speed', 'congestion', 'delay', 'incident', 'accident', 'crash',
        'traffic', 'volume', 'flow', 'density', 'anomaly', 'pattern',
        'trend', 'forecast', 'route', 'road', 'highway', 'vehicle',
        'show', 'display', 'visualize', 'analyze', 'detect', 'predict'
    }
    
    # Common stop words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'me', 'my', 'any', 'some', 'this', 'that', 'these', 'those'
    }
    
    # Tokenize
    words = re.findall(r'\b\w+\b', instr.lower())
    
    # Extract domain keywords
    for word in words:
        if word in domain_keywords:
            keywords.append(word)
    
    # Extract content words (nouns, verbs, adjectives) - heuristic approach
    # Words longer than 4 chars, not stop words, capitalized or domain-relevant
    for word in words:
        if (len(word) > 4 and 
            word not in stop_words and 
            word not in domain_keywords and
            not word.isdigit()):
            # Simple heuristic: if word appears to be content word
            if word.endswith(('tion', 'ness', 'ment', 'ing', 'ed', 'ly', 'ity')):
                keywords.append(word)
    
    # Extract bigrams (two-word phrases)
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        # If both words are meaningful
        if (words[i] not in stop_words and words[i+1] not in stop_words and
            len(words[i]) > 2 and len(words[i+1]) > 2):
            keywords.append(bigram)
    
    # Remove duplicates
    seen = set()
    keywords = [x for x in keywords if not (x in seen or seen.add(x))]
    
    return keywords[:15]  # Limit to top 15 keywords


# Alternative: Train your own model offline
def train_custom_keyword_extractor(training_data_path: str):
    """
    Train a custom TF-IDF based extractor using your traffic dataset
    This can be run once and saved for air-gapped deployment
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pandas as pd
        import joblib
        
        # Load training data
        df = pd.read_csv(training_data_path)
        
        # Train TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=100,
            stop_words='english'
        )
        vectorizer.fit(df['prompt'])
        
        # Save for offline use
        model_path = "DAA-Collab/utils/ML/models/keyword_extractor.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(vectorizer, model_path)
        
        print(f"✅ Custom keyword extractor saved to {model_path}")
        return vectorizer
    except Exception as e:
        print(f"⚠️ Could not train custom extractor: {e}")
        return None