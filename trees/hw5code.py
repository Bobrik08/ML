import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    data = np.column_stack([feature_vector, target_vector])
    data = data[data[:, 0].argsort()]

    x = data[:, 0]
    y = data[:, 1]

    R_m = len(x)

    mask = data[:-1, 0] != data[1:, 0]

    threshold = (data[:-1, 0][mask] + data[1:, 0][mask]) / 2

    if len(threshold) == 0:
        return threshold, np.array([]), None, None

    R_l_all = np.arange(1, R_m)
    R_r_all = R_m - R_l_all

    ones_l_all = np.cumsum(y)[:-1]
    ones_r_all = y.sum() - ones_l_all

    zeros_l_all = R_l_all - ones_l_all
    zeros_r_all = R_r_all - ones_r_all

    R_l = R_l_all[mask]
    R_r = R_r_all[mask]

    ones_l = ones_l_all[mask]
    ones_r = ones_r_all[mask]

    zeros_l = zeros_l_all[mask]
    zeros_r = zeros_r_all[mask]

    p1_l = ones_l / R_l
    p1_r = ones_r / R_r

    p0_l = zeros_l / R_l
    p0_r = zeros_r / R_r

    Gini_R = 1 - p1_r**2 - p0_r**2
    Gini_L = 1 - p1_l**2 - p0_l**2

    Q = (-1) * (Gini_R * R_r / R_m + Gini_L * R_l / R_m)

    index = Q.argmax()

    threshold_best = threshold[index]

    return threshold, Q, threshold_best, Q[index]

    


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
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
                    ratio[key] =  current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            current_split = feature_vector < threshold

            if self._min_samples_leaf is not None:
                if current_split.sum() < self._min_samples_leaf:
                    continue
                if np.logical_not(current_split).sum() < self._min_samples_leaf:
                    continue


            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

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
        self._fit_node(sub_X[split], sub_y[split], node["left_child"],depth=depth+1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"],depth=depth+1)
    def _predict_node(self, x, node):

        if node["type"] == "terminal":
            return node["class"]

        feature_best = node["feature_split"]

        if self._feature_types[feature_best] == "real":
            if x[feature_best] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif self._feature_types[feature_best] == "categorical":
            if x[feature_best] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])


        
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
