# import pandas as pd
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# nltk.download('punkt')
# nltk.download('stopwords')

# data = pd.read_csv("habitat.csv")

# X = data['text']
# y = data['label']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer

# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

# def preprocess_text(text):
#     words = word_tokenize(text)
#     words = [stemmer.stem(word) for word in words if word.isalnum() and word.lower() not in stop_words]
#     return ' '.join(words)

# X_train_preprocessed = [preprocess_text(text) for text in X_train]
# X_test_preprocessed = [preprocess_text(text) for text in X_test]

# vectorizer = TfidfVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train_preprocessed)
# X_test_vec = vectorizer.transform(X_test_preprocessed)

# model = LogisticRegression()
# model.fit(X_train_vec, y_train)

# y_pred = model.predict(X_test_vec)

# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print(report)

# import joblib
# joblib.dump(model, 'habitatModel.pkl')
# joblib.dump(vectorizer, 'vectorModel.pkl')

from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline

def isHabitat(context, questions):
    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
    questions = questions
    tokenizer.encode(questions[0], truncation=True, padding=True)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    context = context
    ans = nlp({
        'question' : questions[0],
        'context': context
    })
    word = context[ans['start']:  ans['end']]
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    stop_words = set(stopwords.words('english'))
    tokenized = word_tokenize(word)
    tagged = nltk.pos_tag(tokenized)
    flag = True
    for i, j in tagged:
        if j == 'NN':
            flag = False
            return True
            break
        else:
            flag=True
    if flag==True:
        return False
    
