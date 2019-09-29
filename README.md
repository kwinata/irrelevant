# This is “Irrelevant!”

This is an excerpt from a project on my intern at Shopee. This is a simple approach to help
eliminate wrongly categorized product within the platform by using the term frequency. A previous approach
using SVM has been tried but was not pursued further because of difficulty in rule extraction.

To generate the html documentation, run the `compile.sh` script (after installing the `requirements.txt`).
To test, just run `python irrelevant/irrelevant.py -v`.

## Code API documentation


#### class irrelevant.irrelevant.Irrelevant(relevant_dominance=4.0, relevant_tf=0.001, clash_dominance=10.0, clash_tf=0.005, maximum_clash_in_relevant=0.05, n_relevant=50, n_clash=30, initial_keywords=None)
Find wrongly categorized products.

The strategy used is based on the difference of likelihood of a keyword
to appear in each of the category. We will take a keyword as a ‘determining keyword’ if the difference is
significant.

For example, let us consider an imaginary brand ‘Good_cheap’ and we are classifying
between ‘Dress’ and ‘Watch’. If the brand ‘Good_cheap’ production is 80% ‘Watch’ and 20% ‘Dress’, we won’t be
brave enough to judge that everything that contains the brand name ‘Good_cheap’ is watch. But if ‘Good_cheap’
main product line is 99% ‘Watch’ with only some exceptional special edition ‘Dress’, we will be brave enough
to use the brand name as determining factor/keyword.

This module compare not between two categories, but a category with **the rest** (complement of the category), i.e.
when there is only two category, those cases are equivalent.

There is two ‘main’ way to use this class, one is to set the training parameter or using initial_keywords
to directly inject the determining keywords.


* **Parameters**

    
    * **relevant_dominance** (`float`) – minimum ratio of P(keyword in relevant)/P(keyword in clash) for relevant keyword


    * **relevant_tf** (`float`) – minimum P(keyword in relevant) for relevant keyword


    * **clash_dominance** (`float`) – minimum ratio of P(keyword in clash)/P(keyword in clash) for clash keyword


    * **clash_tf** (`float`) – minimum P(keyword in clash) for clash keyword


    * **maximum_clash_in_relevant** (`float`) – maximum P(keyword in relevant) for clash keyword


    * **n_relevant** (`int`) – maximum number of top relevant keywords taken (ordered based on term frequency)


    * **n_clash** (`int`) – maximum number of top clash keywords taken (ordered based on ratio / bias strength)


    * **initial_keywords** (`Optional`[`Dict`[`object`, `Dict`[`str`, `List`[`str`]]]]) – A dict of label as key and lists of relevant and clash keywords as dictionary.
    This parameter is used to use the judge method without fitting.
    e.g.: {‘label1’: {‘relevant’: [‘shirt’, ‘tee’, ‘skirt’, ‘jacket’], ‘clash’: [‘bag’, ‘shoes’]}}


### Examples

```python
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
```


#### fit(corpus, y)
This method will calculate and store the determining keywords directly to `self.keywords_`


* **Parameters**

    
    * **corpus** (`List`[`str`]) – the texts to be analyzed


    * **y** (`List`[`object`]) – the labels for each text in the corpus



* **Raises**

    **ValueError** – If corpus length is not equal to y


### Examples

```python
>>> cspam = Irrelevant()
>>> cspam.fit(['Red Shirt number 1', 'Red shoes'], ['clothes', 'footwear'])
>>> cspam.fit(['Red Shirt number 1'], ['clothes', 'footwear']) # uneven length
Traceback (most recent call last):
    ...
ValueError: Uneven length of corpus and y
>>> cspam.judge(['Black shirt', 'Black shoes'], ['clothes', 'clothes']) # retains last successful fit
[set(), {'shoes'}]
```


* **Return type**

    `None`



#### judge(corpus, y)
Judge whether a corpus is miscategorized or not. Will return a list of set, with empty set denoting there is
no clash and set of keywords denoting the clash keywords.

This method works by using the logic miscategorized if not relevant, and clashed. So there is two condition
for a text to be considered as miscategorized:


1. The text must not contain any relevant keyword.


2. The text must contain some clash keyword.

Here is the reason of why we should use both condition:


1. If the condition The text does not contain any relevant keyword suffices for miscategorization,          we will face problem with some product that only contain its brand name, e.g.          `fipper wide blue`, `nike air jordan`, `long champ`. If we use this condition, those item will be          considered as miscategorized as they lack the ‘relevant’ keyword like `slipper`, `shoes`, `bag`, resp.


2. If the condition The text does not contain any relevant keyword suffices for miscategorization,          then text such as `nike bag with free shirt`, `women tee with baby picture` may be wrongly miscategorized          as `bag` and `children` (instead of `adult`).

This method checks for exact match for ‘clash keywords’ and checks circumfix for ‘relevant keywords’.
This means that keyword ‘shoe’ will match all of ‘shoe’, ‘shoes’, ‘runningshoes’ as a ‘relevant keyword’ but
will only match ‘shoe’ (and not ‘shoes’ or ‘runningshoes’) as a ‘clash keyword’.


* **Parameters**

    
    * **corpus** (`List`[`str`]) – the list of texts to be judged


    * **y** (`List`[`object`]) – the current labels for each text in `corpus`



* **Return type**

    `List`[`Set`[`str`]]



* **Returns**

    The set of intersection of clash keywords with each text in `corpus` according to the label in `y`



* **Raises**

    
    * **ValueError** – If the model is untrained and not initialized with `initial_keywords`


    * **ValueError** – If there is a label in the `y` that is not in `self.classes_`


    * **ValueError** – If the length of `corpus` and `y` is different


### Examples

```python
>>> keywords_ = {'clothes': {'relevant': ['shirt', 'tee'], 'clash': ['bag', 'shoes']}}
>>> cspam = Irrelevant(initial_keywords=keywords_)
>>> corpus = ['Red shirt', 'Good magictees', 'tee bag', 'black baggy', 'black bag']
>>> y = ['clothes'] * len(corpus)
>>> cspam.judge(corpus, y)
[set(), set(), set(), set(), {'bag'}]
```
