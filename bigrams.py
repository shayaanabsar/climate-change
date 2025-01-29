
import os
import re
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

lemmatizer =  WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
additional_stopwords = ["et", "al", "the", "of", "and", "is", "be", "to", "in", "for", "on", "with", "by", 
                        "change", "guideline", "factor", "climate", "ipcc", "emission"]
stop_words.update(additional_stopwords)

all_sentences = []
def clean_text(text, climate_text=False):
    text = re.sub(r'\d+', '', text).lower()  # Remove digits
    sentences = sent_tokenize(text)
    
    cleaned_text = ''

    for i, s in enumerate(sentences):
        s = re.sub(r'[^\w\s]', '', s) # remove punctuation after so we can first split into sentences.
        s = [lemmatizer.lemmatize(word) for word in s.split() if word not in stop_words]
        s = ' '.join(s)
        if climate_text: all_sentences.append(s)
        cleaned_text += s
    
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words] # This gets rid of full stops meaning can't split on sentences
    return ' '.join(words)


#print(clean_text('''This chapter serves as an introduction to Part B of this volume. It provides
#context for an assessment of regional aspects of climate change in
#different parts of the world, which are presented in the following nine
#chapters. While the main focus of those chapters is on the regional
#dimensions of impacts, adaptation, and vulnerability (IAV), this chapter
#also offers links to regional aspects of the physical climate reported by
#Working Group I (WGI) and of mitigation analysis reported by Working
#Group III (WGIII). The chapter frames the discussion of both global and
#regional issues in a decision-making context. This context identifies
#different scales of decisions that are made (e.g., global, international,
#regional, national, subnational, local) and the different economic or impact
#sectors that are often the objects of decision making (e.g., agriculture,
#water resources, energy). 
#Within this framing, the chapter then provides three levels of synthesis.
#First there is an evaluation of the state of knowledge of changes in the
#physical climate system, and associated impacts and vulnerabilities, and
#the degree of confidence that we have in understanding those on a
#regional basis as relevant to decision making. Second, the regional
#context of the sectoral findings presented in Part A of this volume is
#discussed. Third, there is an analysis of the regional variation revealed
#in subsequent chapters of Part B.'''))

#print(all_sentences)


def load_and_clean_data(directory, climate_text=False):
    texts = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Try reading the file with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(clean_text(file.read(), climate_text=climate_text))
        except UnicodeDecodeError:
            # Fallback to ISO-8859-1 (latin-1) encoding if UTF-8 fails
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                texts.append(clean_text(file.read(), climate_text=climate_text))
    return texts

predefined_bigrams_C0 = [
    ("air", "pollution"), ("air", "quality"), ("air", "temperature"),
    ("biomass", "energy"), ("carbon", "dioxide"), ("carbon", "emission"),
    ("carbon", "energy"), ("carbon", "neutral"), ("carbon", "price"),
    ("carbon", "sink"), ("carbon", "tax"), ("clean", "air"),
    ("clean", "energy"), ("clean", "water"), ("climate", "change"),
    ("coastal", "area"), ("coastal", "region"), ("electric", "vehicle"),
    ("energy", "climate"), ("energy", "conversion"), ("energy", "efficient"),
    ("energy", "environment"), ("environmental", "sustainability"),
    ("extreme", "weather"), ("flue", "gas"), ("forest", "land"),
    ("gas", "emission"), ("ghg", "emission"), ("global", "decarbonization"),
    ("global", "warm"), ("greenhouse", "gas"), ("heat", "power"),
    ("Kyoto", "protocol"), ("natural", "hazard"), ("new", "energy"),
    ("ozone", "layer"), ("renewable", "energy"), ("sea", "level"),
    ("sea", "water"), ("snow", "ice"), ("solar", "energy"),
    ("solar", "thermal"), ("sustainable", "energy"), ("water", "resource"),
    ("water", "resources"), ("wave", "energy"), ("weather", "climate"),
    ("wind", "energy"), ("wind", "power"), ("wind", "resource"),

    # Bigrams from Table IA. IV - Panel A (Opportunity Bigrams)
    ("heat", "power"), ("new", "energy"), ("plug", "hybrid"),
    ("rooftop", "solar"), ("renewable", "electricity"),
    ("renewable", "energy"), ("wind", "power"),
    ("renewable", "resource"), ("solar", "farm"),
    ("sustainable", "energy"), ("electric", "vehicle"),
    ("wind", "energy"), ("solar", "energy"),
    ("hybrid", "car"), ("clean", "energy"),
    ("electric", "hybrid"), ("geothermal", "power"),

    # Bigrams from Table IA. IV - Panel B (Regulatory Bigrams)
    ("greenhouse", "gas"), ("gas", "emission"), ("carbon", "tax"),
    ("emission", "trade"), ("carbon", "reduction"), ("reduce", "emission"),
    ("air", "pollution"), ("carbon", "price"), ("dioxide", "emission"),
    ("carbon", "market"), ("carbon", "emission"), ("reduce", "carbon"),
    ("environmental", "standard"), ("epa", "regulation"),
    ("mercury", "emission"), ("carbon", "dioxide"),
    ("energy", "regulatory"), ("nox", "emission"),
    ("energy", "independence"),

    # Bigrams from Table IA. IV - Panel C (Physical Bigrams)
    ("coastal", "area"), ("forest", "land"), ("storm", "water"),
    ("natural", "hazard"), ("water", "discharge"),
    ("global", "warm"), ("sea", "level"), ("heavy", "snow"),
    ("sea", "water"), ("ice", "product"), ("snow", "ice"),
    ("nickel", "metal"), ("air", "water"), ("warm", "climate")
]

directory_climate = r'IPCC/raw_txt/IPCC'
directory_non_climate = r'books'

climate_texts = load_and_clean_data(directory_climate, climate_text=True)
non_climate_texts = load_and_clean_data(directory_non_climate)

##Example of cleaned text
#for text in climate_texts:
#	print('.'.count(text)) #This printed 0 for all values meaning full stops are being removed, preventing sentences from being parsed

print("Sample Cleaned Climate Text:", climate_texts[:2])
print("Sample Cleaned Non-Climate Text:", non_climate_texts[:2])

def extract_bigrams(texts, min_freq=10):
    all_bigrams = Counter()
    for text in texts:
        bigram_list = list(bigrams(text.split()))
        all_bigrams.update(bigram_list)
    return {bigram: freq for bigram, freq in all_bigrams.items() if freq >= min_freq}


bigrams_CR = extract_bigrams(climate_texts)
bigrams_N = extract_bigrams(non_climate_texts)


print(f"Filtered Bigrams in CR: {list(bigrams_CR.keys())[:5]}")
print(f"Filtered Bigrams in N: {list(bigrams_N.keys())[:5]}")


# ## RCCE Calculation


def compute_rcce(texts, bigrams_CR, bigrams_N):
    exposures = []
    for text in texts:
        bigram_list = list(bigrams(text.split()))
        total_bigrams = len(bigram_list)
        relevant_bigrams = sum(1 for bigram in bigram_list if bigram in bigrams_CR and bigram not in bigrams_N)
        exposures.append(relevant_bigrams / total_bigrams if total_bigrams > 0 else 0)
    return exposures

rcce_climate = compute_rcce(climate_texts, bigrams_CR, bigrams_N)


print(len(rcce_climate))

 
# ## Filter M for Noise Reduction


def extract_sentences_with_bigrams(texts, bigrams_CR):
	set_m = []

	for sentence in all_sentences:
		bigram_list = list(bigrams(sentence.split()))
		if any(bigram in bigrams_CR for bigram in bigram_list):
			set_m.append(sentence.strip())
	return set_m


set_m = extract_sentences_with_bigrams(climate_texts, bigrams_CR)


print(len(set_m))

 
# ## Partitioning into sets R and S


def partition_m_with_rcce(set_m, bigrams_CR, rcce_scores):
    set_r, set_s = [], []
    rcce_r, rcce_s = [], []

    for i, sentence in enumerate(set_m):
        bigram_list = list(bigrams(sentence.split()))
        if any(bigram in predefined_bigrams_C0 for bigram in bigram_list):
            set_r.append(sentence)
            #rcce_r.append(rcce_scores[i])
        else:
            set_s.append(sentence)
            #rcce_s.append(rcce_scores[i])

    return set_r, set_s, rcce_r, rcce_s


set_r, set_s, rcce_r, rcce_s = partition_m_with_rcce(set_m, bigrams_CR, rcce_climate)
print(f"Set R: {len(set_r)} sentences, Set S: {len(set_s)} sentences")

 
# ## Train Classifiers on Set R and Set S


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold


def train_classifiers(set_r, set_s):
    all_sentences = set_r + set_s
    labels = [1] * len(set_r) + [0] * len(set_s)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(all_sentences)
    y = labels

    # Split data into training and test sets using stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define classifiers
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier()
    }

    # Define parameter grids for each classifier
    param_grid = {
        'Naive Bayes': {'alpha': [0.1, 0.5, 1.0]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Random Forest': {'n_estimators': [50, 100, 200]}
    }

    # Use stratified K-fold cross-validation to optimize hyperparameters
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train each classifier and find the best parameters using GridSearchCV
    best_models = {}
    for name, model in classifiers.items():
        grid = GridSearchCV(model, param_grid[name], cv=stratified_cv, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_

    # Predict probabilities for test set using all three classifiers
    predictions_nb = best_models['Naive Bayes'].predict_proba(X_test)[:, 1]
    predictions_svm = best_models['SVM'].predict_proba(X_test)[:, 1]
    predictions_rf = best_models['Random Forest'].predict_proba(X_test)[:, 1]

    # Use shape[0] instead of len(X_test) since X_test is a sparse matrix
    threshold = 0.8
    target_set_t = [all_sentences[i] for i in range(X_test.shape[0]) if predictions_nb[i] > threshold or predictions_svm[i] > threshold or predictions_rf[i] > threshold]
    
    return target_set_t

# Run the classifier training and create Target Set T
target_set_t = train_classifiers(set_r, set_s)
print("Target Set T (sample sentences):", target_set_t[:5])


from sklearn.model_selection import StratifiedKFold, GridSearchCV

def train_classifiers_with_rcce(set_r, set_s, rcce_r, rcce_s):
    all_sentences = set_r + set_s
    labels = [1] * len(set_r) + [0] * len(set_s)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(all_sentences)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

    # Train classifiers
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Get predictions
    predictions = model.predict_proba(X_test)[:, 1]
    threshold = 0.8
    target_set_t = [all_sentences[i] for i in range(len(X_test.toarray())) if predictions[i] > threshold]

    return target_set_t

# Train classifiers and create Set T
target_set_t = train_classifiers_with_rcce(set_r, set_s, rcce_r, rcce_s)
print("Target Set T (sample sentences):", target_set_t[:5])

 
# ## Predict relevance and create target set T


def extract_bigrams_from_sentences(sentences):
    bigram_counter = Counter()
    for sentence in sentences:
        bigram_list = list(bigrams(sentence.split()))
        bigram_counter.update(bigram_list)
    return bigram_counter

# Extract bigrams from Set T (climate-related sentences)
bigrams_t = extract_bigrams_from_sentences(target_set_t)

# Extract bigrams from Set S \ T (sentences in Set S that are not in Set T)
set_s_t_sentences = [sentence for sentence in set_s if sentence not in target_set_t]
bigrams_s_t = extract_bigrams_from_sentences(set_s_t_sentences)


def calculate_discriminative_bigrams(bigrams_t, bigrams_s_t):
    # Calculate the difference in bigram frequencies between Set T and Set S \ T
    discriminative_bigrams = {
        bigram: bigrams_t[bigram] - bigrams_s_t.get(bigram, 0)
        for bigram in bigrams_t
        if bigrams_t[bigram] > bigrams_s_t.get(bigram, 0)
    }

    # Sort bigrams by their discriminative power (frequency difference)
    sorted_discriminative_bigrams = sorted(discriminative_bigrams.items(), key=lambda item: item[1], reverse=True)

    # Select the top 5% of discriminative bigrams
    top_5_percent_count = int(len(sorted_discriminative_bigrams) * 0.05)
    return sorted_discriminative_bigrams[:top_5_percent_count]

# Calculate top 5% discriminative bigrams
top_discriminative_bigrams = calculate_discriminative_bigrams(bigrams_t, bigrams_s_t)
print("Top 5% Discriminative Bigrams:", top_discriminative_bigrams[:10])


for bigram, score in top_discriminative_bigrams:
	if bigram not in predefined_bigrams_C0:
		print(bigram)


#from docx import Document

#def save_bigrams_to_docx(bigrams, file_name="top_discriminative_bigramss.docx"):
#    # Create a new Document
#    document = Document()
    
#    # Add a title to the document
#    document.add_heading("Top 5% Discriminative Bigrams", level=1)
    
#    # Add each bigram and its score to the document
#    for bigram, score in bigrams:
#        document.add_paragraph(f"{bigram[0]} {bigram[1]}: {score}")
    
#    # Save the document
#    document.save(file_name)
#    print(f"Document saved as {file_name}")

## Save the top discriminative bigrams to a .docx file
#save_bigrams_to_docx(top_discriminative_bigrams)


 
# ## Final Bigram Identification and Ranking


def create_final_bigram_library(predefined_bigrams, discriminative_bigrams):
	# Convert predefined bigrams to a set
	bigrams_C0 = set(predefined_bigrams)

	# Extract just the bigram tuples from the discriminative_bigrams (ignoring scores)
	bigrams_CS = set(bigram for bigram, score in discriminative_bigrams)
	
	scores = {bigram : score for bigram, score in discriminative_bigrams}

	# Combine the sets to form the final bigram library (C)
	final_bigrams_C = list(bigrams_CS.difference(bigrams_C0))
	final_bigrams_C.sort(key = lambda b: scores[b], reverse=True)
	#ordered_bigrams = sorted(discriminative_bigrams, key=lambda )
	#print(discriminative_bigrams[:5])
	return final_bigrams_C

# Create the final bigram library
final_bigrams_library_C = create_final_bigram_library(predefined_bigrams_C0, top_discriminative_bigrams)
print("Final Climate Change Bigrams Library:", list(final_bigrams_library_C)[:10]) 


print(final_bigrams_library_C)
print(len(final_bigrams_library_C))


