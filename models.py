from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn import metrics
import seaborn as sns

class My_Classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        sns.set_theme()

    def valuate(self):
        self.acc = metrics.accuracy_score(self.y_test, self.y_pred)
        self.f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        return (self.acc, self.f1)

    def confusion_matrix(self, axs, y_label):
        #fig, axs = plt.subplots(1, 3, figsize=(10, 8))
        confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_pred)
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=y_label)
        # cm_display.plot(ax=axs)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_label, yticklabels=y_label, ax=axs)
        axs.text(0, 12.5, f"accuracy: {self.acc:.4f}", fontsize=10)
        axs.text(0, 13, f"f1 score: {self.f1:.4f}", fontsize=10)
        axs.grid(False)

class My_NaiveBayes(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf = GaussianNB()

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()
    
class My_KNN(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()
    
class My_RDF(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()
    
class My_SVM(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf  = SVC(kernel='rbf', gamma=0.1, C=1.0, random_state=42)

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()
    
class My_LDA(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf  = LinearDiscriminantAnalysis()

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()
    
class My_QDA(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf  = QuadraticDiscriminantAnalysis()

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()
    
class My_XGB(My_Classifier):
    
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        self.clf = XGBClassifier(
                        objective='multi:softmax',
                        n_estimators=100,       # Number of boosting rounds (trees)
                        learning_rate=0.1,      # Step size shrinkage to prevent overfitting
                        max_depth=9             # Maximum depth of a tree
                    )

    def forward(self):
        self.clf.fit(self.x_train, self.y_train)
        self.y_pred = self.clf.predict(self.x_test)
        return self.valuate()