# Python-Classifier-Using-Decision-tree-and-AdaBoost
Implementation of a language classifier that distinguishes between any given English and Dutch language. Used Decision Tree and AdaBoost algorithm.

	Feature selection for the model

Multiple functions are defined to distinguish between the 2 languages, English and Dutch. These functions describe the common features that help the model differentiate between both of the languages. Some of the prominent features observed and utilized in training the model are: 

Common Dutch words
Common English words
If the letter ‘q’ is present in the sentence
If the word ‘oo’ is present in the sentence
If the word ‘een’ is present in the sentence
If the word ‘aa’ is present in the sentence
If the word ‘ij’ is present in the sentence
If the word ‘ee’ is present in the sentence
If the word ‘de’ is present in the sentence
If the word ‘en’ is present in the sentence
If the word ‘van’ is present in the sentence

All these features would be tested for each line in the provided input file and the result for if the feature is present in the sentence or not is given as a Boolean value. Thus, each sentence would generate a list containing 11 Boolean value for each feature availability in that sentence. 


	 Decision Tree Learning Algorithm

A decision tree is a predictive model used for classifying data. This would be useful to find an objective value based on the input values given to the model. 
The best feature is selected by calculating the entropy and information gain values, and these values are utilized in this algorithm to distinguish between the different languages. The root node is calculated using these values.

The formulas used are:

		Entropy = -∑P(vk)log2p(vk)

		Remainder (A) = ∑ ((pk+nk)/(p+n)) * Entropy

		Information Gain = Entropy – Remainder (A)


The subtrees are also generated using the same process to identify the language of the particular sentence.

	 Ada Boost Learning Algorithm

The Ada Boost algorithm works on weighted data whereas decision tree does not.
The algorithm consists of various weak learners and a strong learner. A single input feature is taken into consideration by the AdaBoost weak learners. Then a single split decision tree is created which is also known as the decision stump. When the first decision stump is drawn every observation is weighted equivalently. After performing this step, an analysis is performed on the results that are obtained from the first decision stump. It checks if any observations are classified unfairly. If so then normalization is performed on it. The process of checking if the observations are classified wrongly and performing normalization on them continues until all the observations are categorized correctly.
