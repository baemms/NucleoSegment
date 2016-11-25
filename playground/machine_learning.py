"""
Examples for machine learning
"""

#X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
X, y = train_data, train_target

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())

clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())
