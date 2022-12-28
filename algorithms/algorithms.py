
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Algorithms:
    def __init__(self, x_training, y_training, y_test, x_test, n_estimators_tree, criterion_tree):
        self.x_training = x_training
        self.y_training = y_training
        self.y_test = y_test
        self.x_test = x_test
        self.n_estimators_tree = n_estimators_tree
        self.criterion_tree = criterion_tree

    def check_base_y(self):
        for y in self.y_test:
            if type(y) != str:
                return True
            else:
                return False

    def get_result_format(self, result):
        if self.check_base_y():
            return {
                'percentage': '',
                'report': '',
                'result': result
            }
        else:
            return {
                'percentage': '%.2f' % (accuracy_score(self.y_test, result) * 100) + '%',
                'report': classification_report(self.y_test, result),
                'result': result
            }

    def naive_bayes(self):
        result = GaussianNB().fit(self.x_training, self.y_training)
        result = result.predict(self.x_test)

        return self.get_result_format(result)

    def tree_decision(self):
        tree_decision = DecisionTreeClassifier(criterion=self.criterion_tree, random_state=0)
        tree_decision.fit(self.x_training, self.y_training)
        result = tree_decision.predict(self.x_test)

        return self.get_result_format(result)

    def random_forest(self):
        random_forest = RandomForestClassifier(n_estimators=self.n_estimators_tree, criterion=self.criterion_tree, random_state=0)
        random_forest.fit(self.x_training, self.y_training)
        result = random_forest.predict(self.x_test)

        return self.get_result_format(result)



