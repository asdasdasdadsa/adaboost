import numpy as np


class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
        self.T = None


class DT:
    def __init__(self, max_depth, min_entropy, min_elem, K):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.root = Node()
        self.K = K

    def terminal_node_output(self, cl, dn, w):
        # nn = np.zeros(self.K)
        # dd = np.unique(cl, return_counts=True)

        q = cl == 1
        if np.argmax(np.array([np.sum(w[q]), np.sum(w[~q])]) / np.sum(w)) == 0:
            return 1
        else:
            return -1
        #
        # nn[dd[0]] = dd[1]
        # return nn / len(cl)

        # return tno(cl, dn)

    def entropy(self, cl, dn, w):
        ind = cl == 1
        i0 = np.sum(w[ind])
        i1 = np.sum(w[~ind])
        z = np.array([i0, i1]) / np.sum(w) #np.unique(cl, return_counts=True)[1]
        log_z = np.log(z)
        return - np.sum(np.multiply(z, log_z))

    def information_gain(self, cl, dn, dn_j, cl_j, w):
        try:
            I = self.entropy(cl, dn, np.concatenate((w[0], w[1]), axis=None))
            for i in range(2):
                I -= np.sum(w[i]) * self.entropy(cl_j[i], dn_j[i], w[i]) / (np.sum(w[0])+np.sum(w[1]))
            return I
        except IndexError:
            print("::::::::::::::::::::::::::::::::::::::::::")
            return 0

    def gen_fun(self, cl, dn, w):
        indexxx = 0
        top = []
        for i in range(len(dn[0])):#количество признаков
            psi = lambda x: x[i]
            J = np.linspace(0, 1, 100)
            for j in J:
                ind_left = dn[:, i] > j
                ind_right = ~ind_left
                dn_left = dn[ind_left]
                dn_right = dn[ind_right]
                cl_left = cl[ind_left]
                cl_right = cl[ind_right]
                w_left = w[ind_left]
                w_right = w[ind_right]

                if indexxx %10 == 0:
                    print(indexxx)
                indexxx += 1
                infor = self.information_gain(cl, dn, [dn_left, dn_right], [cl_left, cl_right], [w_left, w_right])
                if len(top) == 0:
                    top.append([psi, infor, [dn_left, dn_right], [cl_left, cl_right], j, [w_left, w_right]])
                else:
                    if top[0][1] < infor:
                        top[0] = [psi, infor, [dn_left, dn_right], [cl_left, cl_right], j, [w_left, w_right]]
        return top[0]

    def build_tree(self, dn, cl, node, depth, w):
        entropy_val = self.entropy(cl, dn, w)
        if depth >= self.max_depth or entropy_val <= self.min_entropy or len(dn) <= self.min_elem:
            node.T = self.terminal_node_output(cl, dn, w)
        else:
            f = self.gen_fun(cl, dn, w)
            node.split_ind = f[0]
            node.split_val = f[4]
            dn_left, cl_left, w_left = np.array(f[2])[0], np.array(f[3])[0], np.array(f[5])[0]
            dn_right, cl_right, w_right = np.array(f[2])[1], np.array(f[3])[1], np.array(f[3])[1]

            print(str(entropy_val) + ",,, " + str(depth) + " ,,, ")

            node.left = Node()
            node.right = Node()
            self.build_tree(dn_left, cl_left, node.left, depth + 1, w_left)
            self.build_tree(dn_right, cl_right, node.right, depth + 1, w_right)

    def pass_tree(self, node, dn):
        if node.T is None:
            if node.split_ind(dn) > node.split_val:
                return self.pass_tree(node.left, dn)
            else:
                return self.pass_tree(node.right, dn)
        else:
            return node.T

    def pass_tree_all(self, node, dn):
        z = []
        for i in range(len(dn)):
            z.append(self.pass_tree(node, dn[i]))
        return z

    def accuracy_tree(self, cl, dn, node):
        err = 0
        for i in range(len(dn)):
            if np.argmax(self.pass_tree(node, dn[i])) == cl[i]: #np.argmax(cl[i]):
                err += 1
        return err / len(dn)

