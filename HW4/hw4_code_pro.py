import numpy as np
from collections import Counter

from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    
    # Regression
    unique_values = np.sort(np.unique(feature_vector))
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    
    
    # Auxiliary data structures ( to avoid cycles)
    feature_vector_augm = np.tile(feature_vector, len(thresholds)).reshape(len(thresholds), len(feature_vector))

    target_vector_augm = np.tile(target_vector, len(thresholds)).reshape(len(thresholds), len(target_vector))
    
    
    # Working with left leaf
    left_mask = np.where(feature_vector_augm < thresholds.reshape(-1, 1), 1, 0)
    card_left_splits = np.sum(left_mask, axis=1)

    left_ones_prop = np.sum(target_vector_augm * left_mask, axis=1) / card_left_splits
    left_zeros_prop = 1 - left_ones_prop

    left_H = (1 - left_ones_prop**2 - left_zeros_prop**2) * -1 * card_left_splits / len(feature_vector)
    
    
    # Working with right leaf
    right_mask = (left_mask - 1) * -1
    card_right_splits = len(feature_vector) - card_left_splits

    right_ones_prop = np.sum(target_vector_augm * right_mask, axis=1) / card_right_splits
    right_zeros_prop = 1 - right_ones_prop

    right_H = (1 - right_ones_prop**2 - right_zeros_prop**2) * -1 * card_right_splits / len(feature_vector)
    
    
    # final aggregation
    ginis = right_H + left_H
    threshold_best = thresholds[np.argmax(ginis)]
    gini_best = np.max(ginis)
    
    return thresholds, ginis, threshold_best, gini_best



class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, current_depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and current_depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and sub_X.shape[0] <= self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                print(self._feature_types)
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue
            
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:

                if feature_type == "real":
                    threshold_best_cand = threshold
                    split = feature_vector < threshold_best_cand
                    
                elif feature_type == "categorical":
                    threshold_best_cand = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                    split = np.isin(sub_X[:, feature], threshold_best_cand)
                else:
                    raise ValueError
                
                
                if self._min_samples_leaf is not None and (
                        np.sum(split.astype(int)) < self._min_samples_leaf or
                        (split.shape[0] - np.sum(split.astype(int))) < self._min_samples_leaf
                    ):
                        continue
                
                threshold_best = threshold_best_cand
                feature_best = feature
                gini_best = gini

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}

        if sub_X[split].shape[0] != 0:
            self._fit_node(sub_X[split], sub_y[split], node["left_child"], current_depth + 1)
            
        if sub_X[np.logical_not(split)].shape[0] != 0:
            self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], current_depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        
        elif self._feature_types[node["feature_split"]] == "categorical":
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_params(self, *args, **kwargs):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }
    
    def set_params(self, **params):
        self._feature_types = params.get("feature_types", self._feature_types)
        self._max_depth = params.get("max_depth", self._max_depth)
        self._min_samples_split = params.get("min_samples_split", self._min_samples_split)
        self._min_samples_leaf = params.get("min_samples_leaf", self._min_samples_leaf)
        
        return self
