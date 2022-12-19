import dss
import ad

d = dss.Titanic("C:/Users/lul-0/PycharmProjects/adaboost/train_data.csv", "C:/Users/lul-0/PycharmProjects/adaboost/test_data.csv")
dannn = d()
x_train, t_train, x_test, t_test = dannn['train_input'], dannn['train_target'], dannn['test_input'], dannn['test_target']
aadd = ad.AD(len(x_train), 20)
aadd.adadoost(x_train, t_train)
aadd.confusion_matrix(x_test, t_test)
