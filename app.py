from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import language_tool_python
from spellchecker import SpellChecker
import re
from spacy import displacy

app = Flask(__name__)

# Load spaCy model for similarity calculation and syntax tree generation
nlp = spacy.load('en_core_web_lg')

# Load BERT model
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Initialize LanguageTool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

# Initialize spell checker
spell = SpellChecker()

# Intermediate representations
class Token:
    def __init__(self, text, pos):
        self.text = text
        self.pos = pos

class SemanticFeatures:
    def __init__(self, embeddings, named_entities):
        self.embeddings = embeddings
        self.named_entities = named_entities

# Tokenization and Lexical Analysis
def tokenize(sentence):
    tokens = []
    # Tokenize the sentence using spaCy
    doc = nlp(sentence)
    for token in doc:
        tokens.append(Token(token.text, token.pos_))
    return tokens

# Syntax Highlighting
def highlight_syntax(sentence):
    # Define syntax highlighting patterns
    patterns = {
        r'(function|def)\s+(\w+)\s*\(': r'<span class="func">\1</span> <span class="name">\2</span>(',
        r'(if|else|elif|for|while)\s*\(': r'<span class="keyword">\1</span>(',
        r'(True|False|None)\b': r'<span class="const">\1</span>',
        r'(["\'])(?:(?=(\\?))\2.)*?\1': r'<span class="string">\g<0></span>',
        r'#[^\n]*': r'<span class="comment">\g<0></span>'
    }

    # Apply syntax highlighting patterns
    for pattern, replacement in patterns.items():
        sentence = re.sub(pattern, replacement, sentence)

    return sentence

# Semantic Analysis
def analyze_semantics(sentence):
    embeddings = None
    named_entities = None
    # Analyze the semantics of the sentence using spaCy
    doc = nlp(sentence)
    embeddings = doc.vector
    named_entities = [ent.text for ent in doc.ents]
    return SemanticFeatures(embeddings, named_entities)

# Generate Syntax Tree
def generate_syntax_tree(sentence):
    doc = nlp(sentence)
    # Generate the syntax tree in SVG format
    svg = displacy.render(doc, style='dep', options={'compact': True, 'bg': '#09f'})
    return svg

# Construct Syntax Table
def construct_syntax_table(sentence):
    doc = nlp(sentence)
    syntax_table = []
    for token in doc:
        syntax_table.append((token.text, token.dep_, token.head.text, token.head.pos_))
    return syntax_table

# Calculate Similarity Score using spaCy
def calculate_similarity_spacy(sentence1, sentence2):
    # Calculate similarity using spaCy word embeddings
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    
    # Check if both docs are not empty
    if doc1.vector_norm and doc2.vector_norm:
        similarity_score_spacy = doc1.similarity(doc2)
    else:
        similarity_score_spacy = None
    
    return similarity_score_spacy

# Calculate Similarity Score using TF-IDF
def calculate_similarity_tfidf(sentence1, sentence2):
    # Calculate similarity using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence1, sentence2])
    similarity_score_tfidf = cosine_similarity(tfidf_matrix)[0][1]
    return similarity_score_tfidf

# Calculate Similarity Score using Jaccard Method
def calculate_similarity_jaccard(sentence1, sentence2):
    # Tokenize sentences
    tokens1 = set(sentence1.split())
    tokens2 = set(sentence2.split())
    # Calculate Jaccard similarity
    jaccard_similarity = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    return jaccard_similarity

# Calculate Similarity Score using BERT
def calculate_similarity_bert(sentence1, sentence2):
    # Calculate similarity using BERT embeddings
    embeddings1 = bert_model.encode(sentence1)
    embeddings2 = bert_model.encode(sentence2)
    similarity_score_bert = cosine_similarity([embeddings1], [embeddings2])[0][0]
    return similarity_score_bert

# Grammar Check
def check_grammar(sentence):
    grammar_errors = tool.check(sentence)
    return grammar_errors

# Spelling Correction
def correct_spelling(sentence):
    # Tokenize the sentence
    tokens = sentence.split()
    # Correct spelling of each token
    corrected_tokens = [spell.correction(token) for token in tokens]
    # Filter out None values
    corrected_tokens = [token for token in corrected_tokens if token is not None]
    # Join the tokens back to form the corrected sentence
    corrected_sentence = ' '.join(corrected_tokens)
    return corrected_sentence

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input sentences from the form
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']

        # Tokenization and Lexical Analysis
        tokens1 = tokenize(sentence1)
        tokens2 = tokenize(sentence2)

        # Highlight Syntax
        highlighted_sentence1 = highlight_syntax(sentence1)
        highlighted_sentence2 = highlight_syntax(sentence2)

        # Semantic Analysis
        semantic_features1 = analyze_semantics(sentence1)
        semantic_features2 = analyze_semantics(sentence2)

        # Generate Syntax Tree
        syntax_tree1 = generate_syntax_tree(sentence1)
        syntax_tree2 = generate_syntax_tree(sentence2)

        # Construct Syntax Table
        syntax_table1 = construct_syntax_table(sentence1)
        syntax_table2 = construct_syntax_table(sentence2)

        # Similarity Score using spaCy
        similarity_score_spacy = calculate_similarity_spacy(sentence1, sentence2)

        # Similarity Score using TF-IDF
        similarity_score_tfidf = calculate_similarity_tfidf(sentence1, sentence2)

        # Similarity Score using Jaccard Method
        similarity_score_jaccard = calculate_similarity_jaccard(sentence1, sentence2)

        # Similarity Score using BERT
        similarity_score_bert = calculate_similarity_bert(sentence1, sentence2)

        # Grammar Check
        grammar_errors1 = check_grammar(sentence1)
        grammar_errors2 = check_grammar(sentence2)

        # Spelling Correction
        corrected_sentence1 = correct_spelling(sentence1)
        corrected_sentence2 = correct_spelling(sentence2)

        # Render the template with intermediate representations
        return render_template('index.html', tokens1=tokens1, tokens2=tokens2,
                               highlighted_sentence1=highlighted_sentence1,
                               highlighted_sentence2=highlighted_sentence2,
                               semantic_features1=semantic_features1,
                               semantic_features2=semantic_features2,
                               syntax_tree1=syntax_tree1, syntax_tree2=syntax_tree2,
                               syntax_table1=syntax_table1, syntax_table2=syntax_table2,
                               similarity_score_spacy=similarity_score_spacy,
                               similarity_score_tfidf=similarity_score_tfidf,
                               similarity_score_jaccard=similarity_score_jaccard,
                               similarity_score_bert=similarity_score_bert,
                               grammar_errors1=grammar_errors1, grammar_errors2=grammar_errors2,
                               corrected_sentence1=corrected_sentence1, corrected_sentence2=corrected_sentence2)
    else:
        # If it's a GET request, simply render the form
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
