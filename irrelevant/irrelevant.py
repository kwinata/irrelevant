import itertools
import re

import numpy as np
from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction.text import CountVectorizer


RELEVANT_KEY_NAME = 'relevant'
CLASH_KEY_NAME = 'clash'


class Irrelevant:
    """
    Find wrongly categorized products.

    The strategy used is based on the difference of likelihood of a keyword
    to appear in each of the category. We will take a keyword as a 'determining keyword' if the difference is
    significant.

    For example, let us consider an imaginary brand 'Good_cheap' and we are classifying
    between 'Dress' and 'Watch'. If the brand 'Good_cheap' production is 80% 'Watch' and 20% 'Dress', we won't be
    brave enough to judge that everything that contains the brand name 'Good_cheap' is watch. But if 'Good_cheap'
    main product line is 99% 'Watch' with only some exceptional special edition 'Dress', we will be brave enough
    to use the brand name as determining factor/keyword.

    This module compare not between two categories, but a category with **the rest** (complement of the category), i.e.
    when there is only two category, those cases are equivalent.

    There is two 'main' way to use this class, one is to set the training parameter or using initial_keywords
    to directly inject the determining keywords.

    Args:
        relevant_dominance : minimum ratio of P(keyword in relevant)/P(keyword in clash) for relevant keyword
        relevant_tf : minimum P(keyword in relevant) for relevant keyword
        clash_dominance : minimum ratio of P(keyword in clash)/P(keyword in clash) for clash keyword
        clash_tf : minimum P(keyword in clash) for clash keyword
        maximum_clash_in_relevant : maximum P(keyword in relevant) for clash keyword
        n_relevant : maximum number of top relevant keywords taken (ordered based on term frequency)
        n_clash : maximum number of top clash keywords taken (ordered based on ratio / bias strength)
        initial_keywords : A dict of label as key and lists of relevant and clash keywords as dictionary.
            This parameter is used to use the `judge` method without fitting.
            e.g.: `{'label1': {'relevant': ['shirt', 'tee', 'skirt', 'jacket'], 'clash': ['bag', 'shoes']}}`
    Examples:
        >>> cspam = Irrelevant(relevant_dominance=5.0, clash_dominance=20.0)
        >>> cspam.judge(['Black shirt', 'Black shoes'], ['clothes', 'clothes'])
        Traceback (most recent call last):
            ...
        ValueError: Model untrained or trained with empty data.
        >>> cspam.fit(['Red Shirt number 1', 'Red shoes'], ['clothes', 'footwear'])
        >>> cspam.judge(['Black shirt', 'Black shoes'], ['clothes', 'clothes'])
        [set(), {'shoes'}]
        >>> keywords_ = {'label1': {'relevant': ['shirt', 'tee'], 'clash': ['bag', 'shoes']}}
        >>> cspam = Irrelevant(initial_keywords=keywords_)
        >>> cspam.judge(['red shirt', 'black shoes'], ['label1', 'label1'])
        [set(), {'shoes'}]
    """
    def __init__(self, relevant_dominance: float = 4.0, relevant_tf: float = 0.001, clash_dominance: float = 10.0,
                 clash_tf: float = 0.005, maximum_clash_in_relevant: float = 0.05, n_relevant: int = 50,
                 n_clash: int = 30, initial_keywords: Dict[object, Dict[str, List[str]]] = None) -> None:

        self.__relevant_dominance = relevant_dominance
        self.__relevant_tf = relevant_tf
        self.__clash_dominance = clash_dominance
        self.__clash_tf = clash_tf
        self.__maximum_clash_in_relevant = maximum_clash_in_relevant
        self.__classes = []
        self.__keywords_ = []
        self.__num_of_kw = {RELEVANT_KEY_NAME: n_relevant, CLASH_KEY_NAME: n_clash}

        if initial_keywords is not None:
            for key, values in initial_keywords.items():
                self.__classes.append(key)
                self.__keywords_.append({RELEVANT_KEY_NAME: values[RELEVANT_KEY_NAME],
                                         CLASH_KEY_NAME: values[CLASH_KEY_NAME]})

    def fit(self, corpus: List[str], y: List[object]) -> None:
        """
        This method will calculate and store the determining keywords directly to ``self.keywords_``

        Args:
            corpus: the texts to be analyzed
            y: the labels for each text in the corpus
        Raises:
            ValueError: If `corpus` length is not equal to `y`
        Examples:
            >>> cspam = Irrelevant()
            >>> cspam.fit(['Red Shirt number 1', 'Red shoes'], ['clothes', 'footwear'])
            >>> cspam.fit(['Red Shirt number 1'], ['clothes', 'footwear']) # uneven length
            Traceback (most recent call last):
                ...
            ValueError: Uneven length of corpus and y
            >>> cspam.judge(['Black shirt', 'Black shoes'], ['clothes', 'clothes']) # retains last successful fit
            [set(), {'shoes'}]
        """
        if len(corpus) != len(y):
            raise ValueError("Uneven length of corpus and y")

        self.__keywords_ = []
        self.__classes = []

        for i, c in enumerate(corpus):
            corpus[i] = str(c).lower()

        y = np.array(y)
        self.__classes = list(set(y))
        for current_class in self.__classes:
            this_group = list(itertools.compress(corpus, y == current_class))
            other_groups = list(itertools.compress(corpus, y != current_class))

            corpus_dict = {RELEVANT_KEY_NAME: this_group, CLASH_KEY_NAME: other_groups}
            entry_count = {cat: sum(1 for _ in corpus_dict[cat]) for cat in corpus_dict}

            vectorizer = {cat: CountVectorizer(binary=True) for cat in corpus_dict}

            x = {cat: vectorizer[cat].fit_transform(corpus_dict[cat]) for cat in corpus_dict}
            feature_names = {cat: vectorizer[cat].get_feature_names() for cat in corpus_dict}

            feature_total_appearance = {cat: np.transpose(np.sum(x[cat], axis=0)) for cat in corpus_dict}
            feature_map = {cat: {feature: i for i, feature in enumerate(feature_names[cat])} for cat in corpus_dict}
            feature_set = {cat: set(feature_names[cat]) for cat in corpus_dict}

            normalizer = float(entry_count[RELEVANT_KEY_NAME]) / entry_count[CLASH_KEY_NAME]
            all_feature_set = feature_set[RELEVANT_KEY_NAME].union(feature_set[CLASH_KEY_NAME])

            keywords = {
                RELEVANT_KEY_NAME: {},
                CLASH_KEY_NAME: {},
            }
            for feature in all_feature_set:
                score = {
                   cat: feature_total_appearance[cat][feature_map[cat][feature]] if feature in feature_map[cat]
                   else 0 for cat in corpus_dict
                }
                ratio = float(score[RELEVANT_KEY_NAME])/(score[CLASH_KEY_NAME] * normalizer)\
                         if score[CLASH_KEY_NAME] != 0 else int(1e9)
                ratio_inverse = 1/ratio if ratio != 0.0 else int(1e9)
                if (ratio > self.__relevant_dominance and
                        score[RELEVANT_KEY_NAME] >= self.__relevant_tf * (entry_count[
                            RELEVANT_KEY_NAME])):
                    keywords[RELEVANT_KEY_NAME][feature] = float(score[RELEVANT_KEY_NAME])
                elif (ratio_inverse > self.__clash_dominance and
                      score[CLASH_KEY_NAME] >= self.__clash_tf * (entry_count[CLASH_KEY_NAME]) and
                      score[RELEVANT_KEY_NAME] < self.__maximum_clash_in_relevant * entry_count[
                          RELEVANT_KEY_NAME]):
                    keywords[CLASH_KEY_NAME][feature] = float(ratio_inverse)
            sorted_keywords = {}
            for key in [RELEVANT_KEY_NAME, CLASH_KEY_NAME]:
                values = list(keywords[key].values())
                indices = np.argsort(values)
                indices = np.flip(indices)
                sorted_keywords[key] = np.array(list(keywords[key].keys()))[indices]
                sorted_keywords[key] = sorted_keywords[key].tolist()[:self.__num_of_kw[key]]

            self.__keywords_.append(sorted_keywords)

    def judge(self, corpus: List[str], y: List[object]) -> List[Set[str]]:
        """
        Judge whether a corpus is miscategorized or not. Will return a list of set, with empty set denoting there is
        no clash and set of keywords denoting the `clash` keywords.

        This method works by using the logic `miscategorized if not relevant, and clashed`. So there is two condition
        for a text to be considered as miscategorized:

        #. The text must not contain any relevant keyword.

        #. The text must contain some clash keyword.

        Here is the reason of why we should use both condition:

        #. If the condition `The text does not contain any relevant keyword` suffices for miscategorization,\
          we will face problem with some product that only contain its brand name, e.g.\
          ``fipper wide blue``, ``nike air jordan``, ``long champ``. If we use this condition, those item will be\
          considered as miscategorized as they lack the 'relevant' keyword like ``slipper``, ``shoes``, ``bag``, resp.
        #. If the condition `The text does not contain any relevant keyword` suffices for miscategorization,\
          then text such as ``nike bag with free shirt``, ``women tee with baby picture`` may be wrongly miscategorized\
          as ``bag`` and ``children`` (instead of ``adult``).

        This method checks for exact match for 'clash keywords' and checks `circumfix` for 'relevant keywords'.
        This means that keyword 'shoe' will match all of 'shoe', 'shoes', 'runningshoes' as a 'relevant keyword' but
        will only match 'shoe' (and not 'shoes' or 'runningshoes') as a 'clash keyword'.

        Args:
            corpus: the list of texts to be judged
            y: the current labels for each text in ``corpus``
        Returns:
            The set of intersection of clash keywords with each text in ``corpus`` according to the label in ``y``
        Raises:
            ValueError: If the model is untrained and not initialized with ``initial_keywords``
            ValueError: If there is a label in the ``y`` that is not in ``self.classes_``
            ValueError: If the length of ``corpus`` and ``y`` is different
        Examples:
            >>> keywords_ = {'clothes': {'relevant': ['shirt', 'tee'], 'clash': ['bag', 'shoes']}}
            >>> cspam = Irrelevant(initial_keywords=keywords_)
            >>> corpus = ['Red shirt', 'Good magictees', 'tee bag', 'black baggy', 'black bag']
            >>> y = ['clothes'] * len(corpus)
            >>> cspam.judge(corpus, y)
            [set(), set(), set(), set(), {'bag'}]
        """
        classes = set(y)

        if len(self.__classes) == 0:
            raise ValueError("Model untrained or trained with empty data.")

        unrecognized_label = classes - set(self.__classes)
        if len(unrecognized_label) != 0:
            raise ValueError("Unrecognized label:", unrecognized_label)

        if len(corpus) != len(y):
            raise ValueError("Uneven corpus and y length")

        analyzer = CountVectorizer(binary=True).build_analyzer()
        label_to_index = {label: i for i, label in enumerate(self.__classes)}
        judge_result = []
        for i in range(len(corpus)):
            keywords_in_label = self.__keywords_[label_to_index[y[i]]]
            relevant_keywords = set(keywords_in_label[RELEVANT_KEY_NAME])
            clash_keywords = set(keywords_in_label[CLASH_KEY_NAME])
            tokens = set(analyzer(str(corpus[i]).lower()))

            relevant_pattern = "|".join(list(relevant_keywords))
            if tokens.intersection(clash_keywords) and (re.search(relevant_pattern, str(corpus[i]).lower()) is None):
                judge_result.append(tokens.intersection(clash_keywords))
            else:
                judge_result.append(set())
        return judge_result

    def get_keywords(self, group):
        return self.__keywords_[list(self.__classes).index(group)]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
