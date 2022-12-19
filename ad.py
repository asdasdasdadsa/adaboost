import numpy as np
import dt
import plotly.figure_factory as ff
import plotly.express as px

class AD():

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.y = []
        self.node = []
        self.al = []

    def init_w(self):
        return np.array([1/self.N for i in range(self.N)])

    def decision_stump(self, dn, cl, w):
        d = dt.DT(1, 0.05, 5, 2)
        root = dt.Node()
        d.build_tree(dn, cl, root, 0, w)
        return d, root

    def error(self, d, dn, cl, node):
        # z = 0
        # for i in range(len(dn)):
        #     if d.pass_tree(node, dn[i]) != cl[i]:
        #         z += self.w[i]
        # return z
        return np.sum(self.w[d.pass_tree_all(node, dn) != cl])

    def alpha(self, e):
        return np.log((1-e)/e)

    def up_w(self, d, dn, cl, node):
        # z = np.zeros(len(self.w)) + 1
        # w = np.array([self.w[i]*np.exp(self.al*z[d.pass_tree_all(node, dn) != cl]) for i in range(self.N)])

        I = d.pass_tree_all(node, dn) != cl
        alfa = self.al[len(self.al)-1]
        z = np.zeros(self.N)
        z[I] = alfa
        z[~I] = 0
        w = self.w * np.exp(z)

        return w / np.sum(w)

    def adadoost(self, dn, cl):
        self.w = self.init_w()
        for i in range(self.M):
            d, root = self.decision_stump(dn, cl, self.w)
            self.y.append(d)
            self.node.append(root)
            # pred = d.pass_tree()

            e = self.error(d, dn, cl, root)
            self.al.append(self.alpha(e))
            self.w = self.up_w(d, dn, cl, root)
            if e == 0.5:
                break

    def pass_ad(self, dn):
        if np.sum(np.array(self.al) * np.array([self.y[j].pass_tree(self.node[j], dn) for j in range(len(self.y))])) >= 0:
            return 1
        else:
            return -1

    def confusion_matrix(self, dn, cl):
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(dn)):
            if self.pass_ad(dn[i]) == cl[i]:
                if cl[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if cl[i] == 1:
                    FN += 1
                else:
                    FP += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)
        z = [[TP, FP], [FN, TN]]
        test = ['1','-1']
        z_text = [[str(TP), str(FP)], [str(FP), str(TN)]]
        # fig = ff.create_annotated_heatmap(z, x=test, y=test, annotation_text=z_text)
        # fig.show()
        fig = px.imshow(z, text_auto=True, title="precision = " + str(precision) + "; recall = " + str(recall) +
                                                 "; f1_score = " + str(f1_score))
        fig.show()





