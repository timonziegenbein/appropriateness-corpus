This repository contains the source code used to obtain the results reported in the paper [Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238/) published at the ACL2023.
--
Most notably, we publish the **Appropriateness Corpus**, which is a collection of 2191 arguments annotated for appropriateness and its subdimensions, which we derive in the paper (see figure below). The corpus is available [here](https://github.com/timonziegenbein/appropriateness-corpus/blob/main/data/appropriateness-corpus/appropriateness_corpus_conservative.csv). 

## What does Appropriateness mean?
An argument “has an appropriate style if the used language supports the creation of credibility and emotions as well as if it is proportional to the issue.” Their annotation guidelines further suggest that “the choice of words and the grammatical complexity should [...] appear suitable for the topic discussed within the given setting [...], matching the way credibility and emotions are created [...]”. 
> [Wachsmuth et al. (2017)](https://aclanthology.org/E17-1017/)

## What makes an Argument (In)appropriate?
![](https://github.com/timonziegenbein/appropriateness-corpus/blob/main/annotation-guidelines/appropriateness-taxonomy-vertical.svg)

**Toxic Emotions (TE)**: An argument has toxic emotions if the emotions appealed to are deceptive or their intensities do not provide room for critical evaluation of the issue by the reader.
- *Excessive Intensity (EI)*: The emotions appealed to by an argument are unnecessarily strong for the discussed issue.
- *Emotional Deception (ED)*: The emotions appealed to are used as deceptive tricks to win, derail, or end the discussion.

**Missing Commitment (MC)**: An argument is missing commitment if the issue is not taken seriously or openness other’s arguments is absent.
- *Missing Seriousness (MS)*: The argument is either trolling others by suggesting (explicitly or implicitly) that the issue is not worthy of being discussed or does not contribute meaningfully to the discussion.
- *Missing Openness (MO)*: The argument displays an unwillingness to consider arguments with opposing viewpoints and does not assess the arguments on their merits but simply rejects them out of hand.

**Missing Intelligibility (MI)**: An argument is not intelligible if its meaning is unclear or irrelevant to the issue or if its reasoning is not understandable.
- *Unclear Meaning (UM)*: The argument’s content is vague, ambiguous, or implicit, such that it remains unclear what is being said about the issue (it could also be an unrelated issue).
- *Missing Relevance (MR)*: The argument does not discuss the issue, but derails the discussion implicitly towards a related issue or shifts completely towards a different issue.
- *Confusing Reasoning (CR)*: The argument’s components (claims and premises) seem not to be connected logically.

**Other Reasons (OR)**: An argument is inappropriate if it contains severe orthographic errors or for reasons not covered by any other dimension.
- *Detrimental Orthography (DO)*: The argument has serious spelling and/or grammatical errors, negatively affecting its readability.
- *Reason Unclassified (RU)*: There are any other reasons than those above for why the argument should be considered inappropriate.




## Reproducibility of the Results
All the code used to obtain the results reported in the paper is available in the folder `src`. This includes the following:
- `src/annotation-interface`: The code for the annotation interface used to collect the annotations for the corpus-correlations
- `src/annotation-study-appropriateness-prediction`: The code for the training and evaluation of the models reported in the paper
- `src/annotator-agreement`: The code for the annotator agreement reported in the paper
- `src/corpus-correlations`: The code for the corpus correlations reported in the paper
- `src/create-corpus`: The code for the creation of the unannotated corpus from its sources
- `src/dagstuhl-argquality-corpus-eda`: The code for the exploratory data analysis of the Dagstuhl-15512 ArgQuality Corpus

The annotations guidelines and previous taxonomy versions which we developed during our work can be found in the folder `annotation-guidelines`.
## Using our Work 
If you are interested in using the models or the corpus, please cite the following paper:

[Modeling Appropriate Language in Argumentation](https://aclanthology.org/2023.acl-long.238) (Ziegenbein et al., ACL 2023)
