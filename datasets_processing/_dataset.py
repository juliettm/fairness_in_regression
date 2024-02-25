import math
import random
from pathlib import Path


def get_the_middle(values_list):
    return [values_list[math.trunc(len(values_list) / 2)]] if (len(values_list) % 2.0) > 0 \
        else [values_list[math.trunc(len(values_list) / 2) - 1], values_list[math.trunc(len(values_list) / 2)]]

def convert_to_continuous(binary_output, prob_class_1, min_val, max_val, threshold):
    def scale_0_to_1(original_value):
        return (original_value - 0.5) / 0.5

    if binary_output == 0:
        min_val = min_val
        max_val = threshold - 1
        prob = scale_0_to_1(1 - prob_class_1)
        result = max_val - (max_val - min_val) * prob if prob_class_1 < 0.5 else threshold - 1
    else:
        min_val = threshold
        max_val = max_val
        prob = scale_0_to_1(prob_class_1)
        result = min_val + (max_val - min_val) * prob if prob_class_1 > 0.5 else threshold
    return result


class BaseDataset:
    def __init__(self, data_dir=None, random_seed=0, outcome_type='binary'):

        if data_dir is None:
            data_dir = '.datasets'

        data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.random_seed = random_seed

        self.outcome_type = outcome_type
        self._name = None
        self._att = None
        self._outcome_original = None
        self._explanatory_variables = None
        self._outcome_label = None
        self._favorable_label_binary = None
        self._favorable_label_continuous = None
        self._protected_att_name = None
        self._privileged_classes = None
        self._privileged_groups = None
        self._unprivileged_groups = None
        self._continuous_label_name = None
        self._binary_label_name = None

        # variables for computation
        self._cut_point = None
        self._fav_dict = None
        self._nonf_dict = None
        self._non_favorable_label_continuous = None
        self._middle_fav = None
        self._middle_non_fav = None

        self._ds = None

    def set_cut_point(self, new_cut_point):
        self._cut_point = new_cut_point

    def get_cut_point(self):
        return self._cut_point

    def continuous_to_binary(self, column):
        return column.apply(lambda x: 1 if x >= self._cut_point else 0)

    def compute_ordinal_class_prob(self, class_type, prob):
        ordinal_class = 0
        if class_type == 'Favourable':
            for k, v in self._fav_dict.items():
                if v[0] <= prob <= v[1]:
                    ordinal_class = k
        else:
            for k, v in self._nonf_dict.items():
                if v[0] <= prob <= v[1]:
                    ordinal_class = k
        return ordinal_class

    def compute_ordinal_class(self, class_type):
        if class_type == 'Favourable':
            val = random.choice(self._middle_fav)
        else:
            val = random.choice(self._middle_non_fav)
        return val

    def compute_ordinal_output(self, output, probability_output, prob=True):
        ordinal_output = []
        for o in range(len(output)):
            class_type = 'Favourable' if output[o] in self.favorable_label_binary else 'Non-favourable'
            if prob:
                ordinal_output.append(self.compute_ordinal_class_prob(class_type, probability_output[o][output[o]]))
            else:
                ordinal_output.append(self.compute_ordinal_class(class_type))

        return ordinal_output

    def compute_continuous_output(self, output, probability_output):
        # TODO se puede hacer en base a la probabilidad tb
        min_v = min(self.ds[self.continuous_label_name])
        max_v = max(self.ds[self.continuous_label_name])
        ordinal_output = []
        for o in range(len(output)):
            print(output[o], probability_output[o])
            ordinal_output.append(convert_to_continuous(output[o], probability_output[o], min_v, max_v, self._cut_point))
            # if o < self._cut_point:
            #     ordinal_output.append(random.uniform(min(self.ds[self.continuous_label_name]), self._cut_point))
            # else:
            #     ordinal_output.append(random.uniform(self._cut_point, max(self.ds[self.continuous_label_name])))

        return ordinal_output

    def compute_diff_error(self, y, y_predicted):
        diff = 0
        if self._outcome_original == 'ordinal':
            max_value = max(max(self._non_favorable_label_continuous), max(self._favorable_label_continuous))
            min_value = min(min(self._non_favorable_label_continuous), min(self._favorable_label_continuous))
            max_difference = max_value - min_value
            for e in range(len(y)):
                diff += 1 - (abs(y[e] - y_predicted[e]) / max_difference)
        else:
            max_difference = max(self.ds[self.continuous_label_name]) - min(self.ds[self.continuous_label_name])
            for e in range(len(y)):
                diff += 1 - (abs(y[e] - y_predicted[e]) / max_difference)

        return diff / len(y)

    def assign_ranges_to_ordinal(self, fav, sort_desc=False):
        if fav:
            values = self._favorable_label_continuous
        else:
            values = self._non_favorable_label_continuous
        if sort_desc:
            list.sort(values, reverse=True)
        else:
            list.sort(values)
        val = 0.5 / len(values)
        max = 1
        dict_values = {}
        for v in values:
            max_v = max
            min_v = max_v - val
            max = min_v
            dict_values[v] = [min_v, max_v]
        return dict_values

    def reset(self, random_seed):
        self.__init__(random_seed=random_seed)

    def preprocess(self):
        raise NotImplementedError

    @property
    def outcome_label(self):
        return self._outcome_label

    @property
    def favorable_label_binary(self):
        return self._favorable_label_binary

    @property
    def favorable_label_continuous(self):
        return self._favorable_label_continuous

    @property
    def non_favorable_label_continuous(self):
        return self._favorable_label_continuous

    @property
    def protected_att_name(self):
        return self._protected_att_name

    @property
    def privileged_classes(self):
        return self._privileged_classes

    @property
    def privileged_groups(self):
        return self._privileged_groups

    @property
    def unprivileged_groups(self):
        return self._unprivileged_groups

    @property
    def ds(self):
        return self._ds

    @property
    def explanatory_variables(self):
        return self._explanatory_variables

    @property
    def continuous_label_name(self):
        return self._continuous_label_name

    @property
    def binary_label_name(self):
        return self._binary_label_name

    @property
    def name(self):
        return self._name
