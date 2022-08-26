from xgboost import XGBClassifier
from xgboost import plot_importance
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import shap
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import DataFrame

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

df = pd.read_excel('kodata+장려금(중복허용,받은기업들만).xlsx')
print(df.isnull().sum())
print(df.shape)
df.drop_duplicates('사업자등록번호',inplace=True)
print(df.shape)


df = df[[
         '사업일수',    #11
         '자산총계_2022', '자본총계_2022', '영업이익_2022', '매출액_2022', 'ROA','고용율', '통합종업원수','장려금종류']]

print(df.head())

print(df.isnull().sum())
y = df[['장려금종류']]
X = df.drop('장려금종류',axis = 1)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1020)
sc = StandardScaler()
sc.fit(X_train)
X_train, X_val ,y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

print(df.isnull().sum())
#sns.pairplot(df, hue='SVC_CL_CDNM', size=1.5)
#plt.show()
cat_features = list(range(0, X.shape[1]))

print(cat_features)
clf = CatBoostClassifier(
    iterations=5,
    learning_rate=0.1,
    #loss_function='CrossEntropy'
)


clf.fit(X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        verbose=False
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())
# y_pred = model.predict(X_test)
#
# predictions = [round(value) for value in y_pred]
#
# accuracy = accuracy_score(y_test, predictions)
# print('////', accuracy_score(y_test,y_pred)*100)
#
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(y_test[:10])
# print(y_pred[:10])
#
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
# print("Accuracy of XG Boost Classification: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation of XG Boost Classification: {:.2f} %".format(accuracies.std()*100))
#
# preds = bst.predict(dtest)
# best_preds = np.asarray([np.argmax(line) for line in preds])
# print(y_test[:5])
# print(preds[:5])
# print(precision_score(y_test, best_preds, average="macro"))
# ax = xgboost.plot_importance(bst)
# ax.figure.savefig("fi.png")
# ///////////////////
# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(model, ax=ax)
# plt.title(accuracy_score(y_test,y_pred)*100)
# plt.savefig('장려금영향_의영컬럼cat.png')
# plt.title(accuracy_score(y_test,y_pred)*100)
# plt.show()
#///////////////////
# #'총자산회전율', '총자산증가율','ROA'
# plt.figure()
# explainer = shap.Explainer(model.predict, X)
# shap_values = explainer(X)
#
# idx = 3
# #shap.plots.waterfall(shap_values[0])
# shap.plots.waterfall(shap_values[0], max_display=12, show=True)
# plt.savefig('장려금영향shap.png')
# plt.show()
# #
# from sklearn.model_selection import GridSearchCV
#
# # XGBoost 분류기 생성
# xgb_clf = xgboost.XGBClassifier()
#
# # 초모수 격자생성
# xgb_param_grid = {'max_depth': [3,5,7],
#               'subsample': [0.6, 0.8, 1.0],
#                 }
# #'learning_rate': [0.01, 0.03, 0.05]
# # Create a GridSearchCV object
# hr_grid = GridSearchCV(estimator=xgb_clf,
#                        param_grid=xgb_param_grid,
#                        scoring='roc_auc',
#                        n_jobs=8,
#                        cv=5,
#                        refit=True,
#                        return_train_score=True)
#
# hr_grid.fit(X_train, y_train)
#
# hr_grid_df = pd.DataFrame(hr_grid.cv_results_)
# print(hr_grid_df.loc[:, ['mean_test_score', "params"]])
