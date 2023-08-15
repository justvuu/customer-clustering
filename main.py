import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import ttk, messagebox



def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text


def get_feature_vector(location, languages, completed_amount, level, rating_score, reviews, collect_count, tags, five_star, four_star, three_star, two_star, one_star, pro,description,programming_languages,expertises, frontends, backends):
	feature_vector = np.zeros(len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + 
		len(unique_programming_languages) + len(unique_expertises) + len(unique_frontends) + len(unique_backends) + 
		tfidf_vectorizer_desc.get_feature_names_out().shape[0] + 11)

	if isinstance(location, float):
		pass
	else:
		for j in range(len(unique_locations)):
		    if location == unique_locations[j]:
		        feature_vector[j] = 1

	if isinstance(languages, float):
		pass
	else:
		for j in range(len(unique_languages)):
			for l in languages.split(', '):
				if l == unique_languages[j]:
					feature_vector[j + len(unique_locations)] = 1

	if isinstance(level, float):
		pass
	else:
		for j in range(len(unique_levels)):
			if level == unique_levels[j]:
				feature_vector[j + len(unique_locations) + len(unique_languages)] = 1

	if isinstance(tags, float):
		pass
	else:
		for j in range(len(unique_tags)):
			for t in tags.split(', '):
				if t == unique_tags[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels)] = 1

	if isinstance(programming_languages, float):
		pass
	else:
		programming_languages = programming_languages.split(', ')
		for j in range(len(unique_programming_languages)):
			for p in programming_languages:
				if p == unique_programming_languages[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags)] = 1

	if isinstance(expertises, float):
		pass
	else:
		expertises = expertises.split(', ')
		for j in range(len(unique_expertises)):
			for e in expertises:
				if e== unique_expertises[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + len(unique_programming_languages)] = 1

	if isinstance(frontends, float):
		pass
	else:
		frontends = frontends.split(', ')
		for j in range(len(unique_frontends)):
			for f in frontends:
				if f == unique_frontends[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + len(unique_programming_languages) + len(unique_expertises)] = 1

	if isinstance(backends, float):
		pass
	else:
		backends = backends.split(', ')
		for j in range(len(unique_backends)):
			for b in backends:
				if b == unique_backends[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + 
					len(unique_programming_languages) + len(unique_expertises) + len(unique_frontends)] = 1

	if not isinstance(description, float):
		preprocessed_desc = preprocess_text(description)
		desc_tfidf_vector = tfidf_vectorizer_desc.transform([preprocessed_desc])
		feature_vector[-desc_tfidf_vector.shape[1]:] = desc_tfidf_vector.toarray()

	feature_vector[-1] = completed_amount

	feature_vector[-1] = rating_score

	reviews = 0 if isinstance(reviews, float) else str(reviews).replace(",", "")
	feature_vector[-1] = reviews

	collect_count = 0 if isinstance(collect_count, float) else str(collect_count).replace(",", "")
	feature_vector[-1] = collect_count

	five_star = 0 if isinstance(five_star, float) else str(five_star).replace(",", "")
	feature_vector[-1] = five_star

	four_star = 0 if isinstance(four_star, float) else str(four_star).replace(",", "")
	feature_vector[-1] = four_star

	three_star = 0 if isinstance(three_star, float) else str(three_star).replace(",", "")
	feature_vector[-1] = three_star

	two_star = 0 if isinstance(two_star, float) else str(two_star).replace(",", "")
	feature_vector[-1] = two_star

	one_star = 0 if isinstance(one_star, float) else str(one_star).replace(",", "")
	feature_vector[-1] = one_star

	pro = 0 if isinstance(pro, float) else pro
	feature_vector[-1] = pro



	return feature_vector


data = pd.read_csv("data.csv")

unique_locations = []
for column in data:
	if column == 'location':
		unique_locations.extend(data[column].unique())

languages = data["languages"].str.split(", ")
# print(languages)

unique_languages = []
for language in languages:
	if isinstance(language, float): continue
	for l in language:
	    if l not in unique_languages:
	        unique_languages.append(l)

tags = data["tags"].str.split(", ")
unique_tags = []
for tag in tags:
	if isinstance(tag, float): continue
	for t in tag:
		if t not in unique_tags:
			unique_tags.append(t)


programming_languages = data["programming_language"].str.split(", ")
unique_programming_languages = []
for programming_language in programming_languages:
	if isinstance(programming_language, float): continue
	for p in programming_language:
		if p not in unique_programming_languages:
			unique_programming_languages.append(p)

expertises = data["expertise"].str.split(", ")
unique_expertises = []
for expertise in expertises:
	if isinstance(expertise, float): continue
	for e in expertise:
		if(e == ''): continue
		if e not in unique_expertises:
		    unique_expertises.append(e)

frontends = data["frontend_framework"].str.split(", ")
unique_frontends = []
for frontend in frontends:
	if isinstance(frontend, float): continue
	for f in frontend:
		if(f == ''): continue
		if f not in unique_frontends:
		    unique_frontends.append(f)

backends = data["backend_framework"].str.split(", ")
unique_backends = []
for backend in backends:
	if isinstance(backend, float): continue
	for b in backend:
		if(b == ''): continue
		if b not in unique_backends:
		    unique_backends.append(b)

unique_levels = []
unique_levels.extend(data['level'].unique())
tfidf_vectorizer_desc = TfidfVectorizer(max_features=1000)
data["description"].fillna("", inplace=True)
tfidf_vectorizer_desc.fit(data["description"])

feature_vectors = []
for i in range(len(data)):
	feature_vector = np.zeros(len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + 
		len(unique_programming_languages) + len(unique_expertises) + len(unique_frontends) + len(unique_backends) + 
		tfidf_vectorizer_desc.get_feature_names_out().shape[0] + 11)

	location = data["location"][i]
	if isinstance(location, float):
		pass
	else:
		for j in range(len(unique_locations)):
		    if location == unique_locations[j]:
		        feature_vector[j] = 1

	languages = data["languages"][i]
	if isinstance(languages, float):
		pass
	else:
		for j in range(len(unique_languages)):
			for l in languages.split(', '):
				if l == unique_languages[j]:
					feature_vector[j + len(unique_locations)] = 1

	level = data["level"][i]
	if isinstance(level, float):
		pass
	else:
		for j in range(len(unique_levels)):
			if level == unique_levels[j]:
				feature_vector[j + len(unique_locations) + len(unique_languages)] = 1

	tags = data["tags"][i]
	if isinstance(tags, float):
		pass
	else:
		for j in range(len(unique_tags)):
			for t in tags.split(', '):
				if t == unique_tags[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels)] = 1

	programming_languages = data["programming_language"][i]
	if isinstance(programming_languages, float):
		pass
	else:
		programming_languages = programming_languages.split(', ')
		for j in range(len(unique_programming_languages)):
			for p in programming_languages:
				if p == unique_programming_languages[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags)] = 1

	expertises = data["expertise"][i]
	if isinstance(expertises, float):
		pass
	else:
		expertises = expertises.split(', ')
		for j in range(len(unique_expertises)):
			for e in expertises:
				if e== unique_expertises[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + len(unique_programming_languages)] = 1

	frontends = data["frontend_framework"][i]
	if isinstance(frontends, float):
		pass
	else:
		frontends = frontends.split(', ')
		for j in range(len(unique_frontends)):
			for f in frontends:
				if f == unique_frontends[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + len(unique_programming_languages) + len(unique_expertises)] = 1

	backends = data["backend_framework"][i]
	if isinstance(backends, float):
		pass
	else:
		backends = backends.split(', ')
		for j in range(len(unique_backends)):
			for b in backends:
				if b == unique_backends[j]:
					feature_vector[j + len(unique_locations) + len(unique_languages) + len(unique_levels) + len(unique_tags) + 
					len(unique_programming_languages) + len(unique_expertises) + len(unique_frontends)] = 1

	description = data["description"][i]
	if not isinstance(description, float):
		preprocessed_desc = preprocess_text(description)
		desc_tfidf_vector = tfidf_vectorizer_desc.transform([preprocessed_desc])
		feature_vector[-desc_tfidf_vector.shape[1]:] = desc_tfidf_vector.toarray()

	completed_amount = str(data["completed_amount"][i]).replace(",", "")
	feature_vector[-1] = completed_amount

	rating_score = data["rating_score"][i]
	feature_vector[-1] = rating_score

	reviews = data["reviews"][i]
	reviews = 0 if isinstance(reviews, float) else reviews.replace(",", "")
	feature_vector[-1] = reviews

	collect_count = data["collect_count"][i]
	collect_count = 0 if isinstance(collect_count, float) else collect_count.replace(",", "")
	feature_vector[-1] = collect_count

	five_star = data["five_star"][i]
	five_star = 0 if isinstance(five_star, float) else five_star.replace(",", "")
	feature_vector[-1] = five_star

	four_star = data["four_star"][i]
	four_star = 0 if isinstance(four_star, float) else four_star.replace(",", "")
	feature_vector[-1] = four_star

	three_star = data["three_star"][i]
	three_star = 0 if isinstance(three_star, float) else three_star.replace(",", "")
	feature_vector[-1] = three_star

	two_star = data["two_star"][i]
	two_star = 0 if isinstance(two_star, float) else two_star.replace(",", "")
	feature_vector[-1] = two_star

	one_star = data["one_star"][i]
	one_star = 0 if isinstance(one_star, float) else one_star.replace(",", "")
	feature_vector[-1] = one_star

	pro = data["pro"][i]
	pro = 0 if isinstance(pro, float) else pro
	feature_vector[-1] = pro



	feature_vectors.append(feature_vector)


kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(feature_vectors)

# Predict the cluster for each user
# labels = kmeans.predict(feature_vectors)
input_data = get_feature_vector('Vietnam', 'English', '500', 'Level 2', 5, 383, 289, 'Wordpress, PHP, Javascript', 382, 0,0,0,0,1, 'Description', 'Python', 'Design', 'React.js', 'Django')
# print(input_data.shape)
# print(feature_vectors[0].shape)
# label = kmeans.predict([input_data])
# print(label)

# Print the cluster labels for each user
# for i in range(len(data)):
#     print("User {}: {}".format(data["username"][i], labels[i]))

			
        

# print(unique_features)

root = tk.Tk()

root.title("Clustering")
root.geometry("800x600")

# name_var = tk.StringVar()
# passw_var = tk.StringVar()
location_var = tk.StringVar()
languages_var = tk.StringVar()
completed_amount_var = tk.StringVar()
level_var = tk.StringVar()
rating_score_var = tk.StringVar()
reviews_var = tk.StringVar()
collect_count_var = tk.StringVar()
tags_var = tk.StringVar()
five_star_var = tk.StringVar()
four_star_var = tk.StringVar()
three_star_var = tk.StringVar()
two_star_var = tk.StringVar()
one_star_var = tk.StringVar()
pro_var = tk.StringVar()
description_var = tk.StringVar()
programming_language_var = tk.StringVar()
expertise_var = tk.StringVar()
frontend_framework_var = tk.StringVar()
backend_framework_var=tk.StringVar()


def submit():
  location = location_var.get()
  languages = languages_var.get()
  completed_amount = completed_amount_var.get()
  level = level_var.get()
  rating_score = rating_score_var.get()
  reviews = reviews_var.get()
  collect_count = collect_count_var.get()
  tags = tags_var.get()
  five_star = five_star_var.get()
  four_star = four_star_var.get()
  three_star = three_star_var.get()
  two_star = two_star_var.get()
  one_star = one_star_var.get()
  pro = pro_var.get()
  description = description_var.get()
  programming_language = programming_language_var.get()
  expertise = expertise_var.get()
  frontend_framework = frontend_framework_var.get()
  backend_framework = backend_framework_var.get()

  input_data = get_feature_vector(location, languages, completed_amount, level, rating_score, reviews, collect_count, tags, five_star, four_star, three_star, two_star, one_star, pro,
  									description, programming_language, expertise, frontend_framework, backend_framework)


  prediction = kmeans.predict([input_data])
  prediction_label.config(text=f"Predicted Cluster: {prediction[0]}")


labels_and_entries = [
    ("Location", location_var), ("Languages", languages_var), ("Completed amount", completed_amount_var),
    ("Level", level_var), ("Rating score", rating_score_var), ("Reviews", reviews_var),
    ("Collect count", collect_count_var), ("Tags", tags_var), ("Five star", five_star_var),
    ("Four star", four_star_var), ("Three star", three_star_var), ("Two star", two_star_var),
    ("One star", one_star_var), ("Pro", pro_var), ("Description", description_var),
    ("Programming language", programming_language_var), ("Expertise", expertise_var),
    ("Frontend framework", frontend_framework_var), ("Backend framework", backend_framework_var)
]

left_column = tk.Frame(root)
left_column.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

row = 0
for i, (label_text, var) in enumerate(labels_and_entries[:len(labels_and_entries)//2+1]):
    label = tk.Label(left_column, text=label_text)
    label.grid(row=row, column=0, sticky="e", padx=10, pady=5)
    entry = tk.Entry(left_column, textvariable=var)
    entry.grid(row=row, column=1, padx=10, pady=5)
    row += 1

# Create right column
right_column = tk.Frame(root)
right_column.grid(row=0, column=1, padx=20, pady=20, sticky="ne")

for i, (label_text, var) in enumerate(labels_and_entries[len(labels_and_entries)//2+1:], start=row):
    label = tk.Label(right_column, text=label_text)
    label.grid(row=i, column=0, sticky="e", padx=10, pady=5)
    entry = tk.Entry(right_column, textvariable=var)
    entry.grid(row=i, column=1, padx=10, pady=5)

sub_btn = tk.Button(root, text='Submit', command=submit)
sub_btn.grid(row=row, column=1, columnspan=2, pady=10)
prediction_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"))
prediction_label.grid(row=row + 1, column=0, columnspan=2, pady=10)

root.mainloop()









