# import numpy as np
# import pandas as pd
# from collections import defaultdict
# from scipy.stats import hmean
# from scipy.spatial.distance import cdist
# from scipy import stats
# import numbers


# def weighted_hamming(data):
#     """ Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
#         the values between point A and point B are different, else it is equal the relative frequency of the
#         distribution of the value across the variable. For multiple variables, the harmonic mean is computed
#         up to a constant factor.

#         @params:
#             - data = a pandas data frame of categorical variables

#         @returns:
#             - distance_matrix = a distance matrix with pairwise distance for all attributes
#     """
#     categories_dist = []
    
#     for category in data:
#         X = pd.get_dummies(data[category])
#         X_mean = X * X.mean()
#         X_dot = X_mean.dot(X.transpose())
#         X_np = np.asarray(X_dot.replace(0,1,inplace=False))
#         categories_dist.append(X_np)
#     categories_dist = np.array(categories_dist)
#     distances = hmean(categories_dist, axis=0)
#     return distances


# def distance_matrix(data, numeric_distance = "euclidean", categorical_distance = "jaccard"):
#     """ Compute the pairwise distance attribute by attribute in order to account for different variables type:
#         - Continuous
#         - Categorical
#         For ordinal values, provide a numerical representation taking the order into account.
#         Categorical variables are transformed into a set of binary ones.
#         If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
#         variables are all normalized in the process.
#         If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.
        
#         Note: If weighted-hamming distance is chosen, the computation time increases a lot since it is not coded in C 
#         like other distance metrics provided by scipy.

#         @params:
#             - data                  = pandas dataframe to compute distances on.
#             - numeric_distances     = the metric to apply to continuous attributes.
#                                       "euclidean" and "cityblock" available.
#                                       Default = "euclidean"
#             - categorical_distances = the metric to apply to binary attributes.
#                                       "jaccard", "hamming", "weighted-hamming" and "euclidean"
#                                       available. Default = "jaccard"

#         @returns:
#             - the distance matrix
#     """
#     possible_continuous_distances = ["euclidean", "cityblock"]
#     possible_binary_distances = ["euclidean", "jaccard", "hamming", "weighted-hamming"]
#     number_of_variables = data.shape[1]
#     number_of_observations = data.shape[0]

#     # Get the type of each attribute (Numeric or categorical)
#     is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
#     is_all_numeric = sum(is_numeric) == len(is_numeric)
#     is_all_categorical = sum(is_numeric) == 0
#     is_mixed_type = not is_all_categorical and not is_all_numeric

#     # Check the content of the distances parameter
#     if numeric_distance not in possible_continuous_distances:
#         print("The continuous distance " + numeric_distance + " is not supported.")
#         return None
#     elif categorical_distance not in possible_binary_distances:
#         print("The binary distance " + categorical_distance + " is not supported.")
#         return None

#     # Separate the data frame into categorical and numeric attributes and normalize numeric data
#     if is_mixed_type:
#         number_of_numeric_var = sum(is_numeric)
#         number_of_categorical_var = number_of_variables - number_of_numeric_var
#         data_numeric = data.iloc[:, is_numeric]
#         data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
#         data_categorical = data.iloc[:, [not x for x in is_numeric]]

#     # Replace missing values with column mean for numeric values and mode for categorical ones. With the mode, it
#     # triggers a warning: "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
#     # but the value are properly replaced
#     if is_mixed_type:
#         data_numeric.fillna(data_numeric.mean(), inplace=True)
#         for x in data_categorical:
#             data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)
#     elif is_all_numeric:
#         data.fillna(data.mean(), inplace=True)
#     else:
#         for x in data:
#             data[x].fillna(data[x].mode()[0], inplace=True)

#     # "Dummifies" categorical variables in place
#     if not is_all_numeric and not (categorical_distance == 'hamming' or categorical_distance == 'weighted-hamming'):
#         if is_mixed_type:
#             data_categorical = pd.get_dummies(data_categorical)
#         else:
#             data = pd.get_dummies(data)
#     elif not is_all_numeric and categorical_distance == 'hamming':
#         if is_mixed_type:
#             data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).transpose()
#         else:
#             data = pd.DataFrame([pd.factorize(data[x])[0] for x in data]).transpose()

#     if is_all_numeric:
#         result_matrix = cdist(data, data, metric=numeric_distance)
#     elif is_all_categorical:
#         if categorical_distance == "weighted-hamming":
#             result_matrix = weighted_hamming(data)
#         else:
#             result_matrix = cdist(data, data, metric=categorical_distance)
#     else:
#         result_numeric = cdist(data_numeric, data_numeric, metric=numeric_distance)
#         if categorical_distance == "weighted-hamming":
#             result_categorical = weighted_hamming(data_categorical)
#         else:
#             result_categorical = cdist(data_categorical, data_categorical, metric=categorical_distance)
#         result_matrix = np.array([[1.0*(result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] *
#                                number_of_categorical_var) / number_of_variables for j in range(number_of_observations)] for i in range(number_of_observations)])

#     # Fill the diagonal with NaN values
#     np.fill_diagonal(result_matrix, np.nan)

#     return pd.DataFrame(result_matrix)


# def knn_impute(target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
#                categorical_distance="jaccard", missing_neighbors_threshold = 0.5):
#     """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
#         attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
#         remains missing. If there is a problem in the parameters provided, returns None.
#         If to many neighbors also have missing values, leave the missing value of interest unchanged.

#         @params:
#             - target                        = a vector of n values with missing values that you want to impute. The length has
#                                               to be at least n = 3.
#             - attributes                    = a data frame of attributes with n rows to match the target variable
#             - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
#                                               value between 1 and n.
#             - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
#                                               Default = "mean"
#             - numeric_distances             = the metric to apply to continuous attributes.
#                                               "euclidean" and "cityblock" available.
#                                               Default = "euclidean"
#             - categorical_distances         = the metric to apply to binary attributes.
#                                               "jaccard", "hamming", "weighted-hamming" and "euclidean"
#                                               available. Default = "jaccard"
#             - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
#                                               the correct value. Default = 0.5

#         @returns:
#             target_completed        = the vector of target values with missing value replaced. If there is a problem
#                                       in the parameters, return None
#     """

#     # Get useful variables
#     possible_aggregation_method = ["mean", "median", "mode"]
#     number_observations = len(target)
#     is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

#     # Check for possible errors
#     if number_observations < 3:
#         print("Not enough observations.")
#         return None
#     if attributes.shape[0] != number_observations:
#         print("The number of observations in the attributes variable is not matching the target variable length.")
#         return None
#     if k_neighbors > number_observations or k_neighbors < 1:
#         print("The range of the number of neighbors is incorrect.")
#         return None
#     if aggregation_method not in possible_aggregation_method:
#         print ("The aggregation method is incorrect.")
#         return None
#     if not is_target_numeric and aggregation_method != "mode":
#         print("The only method allowed for categorical target variable is the mode.")
#         return None

#     # Make sure the data are in the right format
#     target = pd.DataFrame(target)
#     attributes = pd.DataFrame(attributes)

#     # Get the distance matrix and check whether no error was triggered when computing it
#     distances = distance_matrix(attributes, numeric_distance, categorical_distance)
#     if distances is None:
#         return None

#     # Get the closest points and compute the correct aggregation method
#     for i, value in enumerate(target.iloc[:, 0]):
#         if pd.isnull(value):
#             order = distances.iloc[i,:].values.argsort()[:k_neighbors]
#             closest_to_target = target.iloc[order, :]
#             missing_neighbors = [x for x  in closest_to_target.isnull().iloc[:, 0]]
#             # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
#             if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
#                 continue
#             elif aggregation_method == "mean":
#                 target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
#             elif aggregation_method == "median":
#                 target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
#             else:
#                 target.iloc[i] = stats.mode(closest_to_target, nan_policy='omit')[0][0]

#     return target


# import pandas as pd 
# df = pd.read_csv("_drafts/comsepsis.csv")
# new_df = {}

# ans = knn_impute(target=df['HR'], attributes=df.drop(['HR', 'SepsisLabel'], 1),
#                                     aggregation_method="median", k_neighbors=10, numeric_distance='cityblock',
#                                     categorical_distance='hamming', missing_neighbors_threshold=0.8)

# # print(ans)

from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np 
# #pip install -U scikit-learn
data = "_drafts/comsepsis.csv"
X = pd.read_csv(data)
# imputer = KNNImputer()
# k = imputer.fit_transform(df)
# print(k)

# def manhatt(x, y, missing_values=np.nan, squared=False):
#         x = np.ma.array(x, mask=np.isnan(x))
#         y = np.ma.array(y, mask=np.isnan(y))
#         dist = np.nansum(np.abs(x-y))
#         return dist

# We should drop the Sepsis Label and then do the Impute

X.drop(columns=['SepsisLabel'], inplace=True)
# print(X)
# imputer = KNNImputer(n_neighbors=10, metric=manhatt)
X = X.astype('float')
# X[X == 0] = np.nan # or use np.nan
# print(X)
# ans = imputer.fit_transform(X)
# print(ans)

# from impyute.imputation.cs import mice

# # start the MICE training
# imputed_training=mice(X.values)
# print(imputed_training)

# import numpy as np
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imp = IterativeImputer(max_iter=10, random_state=0)
# imp.fit(X.values)
# IterativeImputer(random_state=0)
# X_test = X
# print(np.round(imp.transform(X_test)))


from sklearn.metrics.pairwise import nan_euclidean_distances, manhattan_distances
nan = float("NaN")
X.fillna(0, inplace=True)
X = X.values
print(manhattan_distances(X, X)) # distance between rows of X
distance_man = manhattan_distances(X, X)
distance_man = pd.DataFrame(distance_man)

for i in distance_man.columns:
    print('index', distance_man[i].nsmallest(4).index)
    print('values', distance_man[i].nsmallest(4))



    # force_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    # X, Y = check_pairwise_arrays(X, Y, accept_sparse=False,
    #                              force_all_finite=force_all_finite, copy=copy)
    # # Get missing mask for X
    # missing_X = _get_mask(X, missing_values)

    # # Get missing mask for Y
    # missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)

    # # set missing values to zero
    # X[missing_X] = 0
    # Y[missing_Y] = 0

    # distances = euclidean_distances(X, Y, squared=True)

    # # Adjust distances for missing values
    # XX = X * X
    # YY = Y * Y
    # distances -= np.dot(XX, missing_Y.T)
    # distances -= np.dot(missing_X, YY.T)

    # np.clip(distances, 0, None, out=distances)

    # if X is Y:
    #     # Ensure that distances between vectors and themselves are set to 0.0.
    #     # This may not be the case due to floating point rounding errors.
    #     np.fill_diagonal(distances, 0.0)

    # present_X = 1 - missing_X
    # present_Y = present_X if Y is X else ~missing_Y
    # present_count = np.dot(present_X, present_Y.T)
    # distances[present_count == 0] = np.nan
    # # avoid divide by zero
    # np.maximum(1, present_count, out=present_count)
    # distances /= present_count
    # distances *= X.shape[1]

    # if not squared:
    #     np.sqrt(distances, out=distances)

    # return distances