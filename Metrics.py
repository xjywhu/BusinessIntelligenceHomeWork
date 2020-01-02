from sklearn.metrics import cohen_kappa_score, hinge_loss, jaccard_similarity_score
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score


class Metrics(object):

    def __init__(self, y_true, y_pred):
         self.y_true = y_true
         self.y_pred = y_pred

    def kappa(self):
        return cohen_kappa_score(self.y_true, self.y_pred)

    def jaccard(self):
        return jaccard_similarity_score(self.y_true, self.y_pred)

    def hinge(self):
        return hinge_loss(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average="macro")

    def f1(self):
        return f1_score(self.y_true, self.y_pred, average="macro")

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average="macro")

    def evaluate(self):
        print("kappa:"+str(self.kappa()))
        print("jaccard:" + str(self.jaccard()))
        print("f1-score:"+str(self.f1()))
        print("accuracy:" + str(self.accuracy()))
        print("precision:"+str(self.precision()))
        print("recall:" + str(self.recall()))
        # print("hinge:" + str(self.hinge()))
