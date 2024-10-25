from sklearn.feature_extraction import DictVectorizer

# Sample data
data = [
    {'city': 'New York', 'temperature': 85},
    {'city': 'Los Angeles', 'temperature': 90},
    {'city': 'New York', 'temperature': 80}
]

# DictVectorizer with sparse=False
vectorizer_dense = DictVectorizer(sparse=False)
X_dense = vectorizer_dense.fit_transform(data)

print("Dense Output:")
print(X_dense)
print("Feature Names (Dense):")
print(vectorizer_dense.get_feature_names_out())

# DictVectorizer with sparse=True
vectorizer_sparse = DictVectorizer(sparse=True)
X_sparse = vectorizer_sparse.fit_transform(data)

print("\nSparse Output:")
print(X_sparse)
print("Feature Names (Sparse):")
print(vectorizer_sparse.get_feature_names_out())
