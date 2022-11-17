---
title: The Wakong Algorithm and Its Python Implementation
---

# Introduction

Sentence masking is a common procedure in the pre-training of deep learning models. For instance, in the pre-training of the BART model, there is a training objective named text infilling. The aim of text infilling is to let the model acquire the ability to fill in the blanks, which requires the development of a masking algorithm to generate masked sentences for the training. However, existing papers (TODO: who?) provide only vague descriptions of the masking methods in their experiments, which are merely theoretical and cannot really be implemented as algorithms; while most of the implementations only design roughly usable algorithms without in-depth analysis. In response, we propose an appropriate and robust masking algorithm (the wakong algorithm), and publishes a Python library that can be used directly in production.

# Problem Formulation

## Input and output of the algorithm

A sentence masking algorithm is an algorithm for masking a sentence of a fixed length. The input of the algorithm is the length of the sentence, denoted by `seq_len`, and the output is a list of tuples. Each tuple represents a mask in the original sentence. The first element represents the starting position of the mask and the second element represents the length.

For example, when the length of the input sentence 40, the output of the algorithm might be:

```
[(5, 4), (23, 2)]
```

This output indicates that two positions of the sentence are masked. The first one starts at position 5 and has a length of 4, while the second starts at position 23 and has a length of 2. It can also be represented graphically as follows:

```
.....(xxxx)..............(xx)...............
```

## Requirements of the algorithm

Based on the BART paper, we suggest that a reasonable sentence masking algorithm should satisfy the following four requirements:

1. The masking algorithm is a stochastic algorithm, i.e. the algorithm may output different results even if the inputs are the same. This is to allow for a greater variety of masking schemes, so that models can better acquire knowledge of cloze tests;
2. The number of words masked by the masking algorithm should be 15% on average. This number was chosen to be moderate enough not to make the pre-training target too simple, while retaining a certain amount of semantic information that would allow the cloze test to be completed;
3. The length of each blank in the masking algorithm is variable, ranging from 0 to 10, with blanks of length 3 occurring most frequently. The frequency increases from 0 to 3 and decreases from 3 to 10. Blanks of length 0 may occur. This is to allow the model to learn that no words may be required in the cloze test, increasing the difficulty of the training and thus allowing the model to learn more semantic knowledge.
4. Any two masks in the masking algorithm must not overlap or be directly adjacent to each other. They must have a distance of at least 1. This is to ensure that the output of the algorithm is well-formed.

# The Algorithm

1\. Constants

$\mathsf{proposedMaskRate} = 0.188$

$\mathsf{poissonRate} = 4.2$

$\mathsf{maxSpanLen} = 10$

2\. `probsList`

$\mathsf{probsList} = \left[ \mathrm{normalise} \left(  \mathsf{probs} \left[ {:}\,i \right] \right) \mathrm{for} \; i \; \mathrm{in} \left[2, \; .. , \; \mathsf{maxSpanLen} + 1 \right] \right]$

$\mathsf{probs} = \left[ \Pr(X=i) \; \mathrm{for} \; i \; \mathrm{in} \left[ 0, \; .., \; \mathsf{maxSpanLen} + 1 \right] \right]$

$X \sim \mathrm{Pois}(\mathsf{poissonRate})$

3\. `determineShouldMaskLen`

$\mathsf{determineShouldMaskLen} \left( \mathsf{seqLen} \right) =
\begin{cases}
    \lceil x \rceil, & \text{if} \; \omega < p \\
    \lfloor x \rfloor, & \text{otherwise} \\
\end{cases}$

$\omega \sim \mathrm{U} \left( 0, 1 \right)$

$x = \mathsf{seqLen} * \mathsf{proposedMaskRate}$

$p = x - \lfloor x \rfloor$

4\. `generateSpans`

$\mathsf{generateSpans} \left( m \right) = \mathrm{shuffle} \left( \mathrm{anamorphism} \left( f \right) \left( m \right) \right)$

$f \left( \mathsf{remainedLen} \right) =
\begin{cases}
    \mathrm{Nothing}, & \text{if} \; \mathsf{remainedLen} \leq 0 \\
    \left( \mathsf{span}, \; \mathrm{Just} \left( \mathsf{remainedLen} - \mathsf{span} - 1 \right) \right), & \text{otherwise}
\end{cases}$

$\mathsf{span} \sim \mathrm{Categorical} \left( [0, \; .., \; n + 1], \; \mathsf{probsList} \left[ n - 1 \right] \right)$

$n = \min \left( \mathsf{maxSpanLen}, \; \mathsf{remainedLen} \right)$

5\. `distributeInsertPoses`

$\mathsf{distributeInsertPoses} \left( \mathsf{xs} \right) = f \left( \mathsf{xs}, \; 0 \right)$

$f \left( n, \; \mathsf{xs} \right) =
\begin{cases}
    \mathsf{\left[ \, \right]}, & \text{if} \; \mathrm{empty} \left( \mathsf{xs} \right) \\
    \left[ \left( p + n, \; s \right) \right] + f \left(n + s + 1, \; \mathsf{ys} \right), & \text{otherwise} \\
\end{cases}$

$\left[ \left( p, s \right) \right] + \mathsf{ys} \leftarrow \mathsf{xs}$

6\. `randomAddOne`

$\mathsf{randomAddOne} \left( \mathsf{xs} \right) = \begin{cases}
    \mathsf{xs}, & \text{if} \; \omega < 0.5 \\
    \left[ (p + 1, s) \; \mathrm{for} \; (p, s) \; \mathrm{in} \; \mathsf{xs} \right], & \text{otherwise} \\
\end{cases}$

$\omega \sim \mathrm{U} \left( 0, 1 \right)$

7\. `wakong`

$\mathsf{wakong} \left( \mathsf{seqLen} \right) = \mathsf{randomAddOne} \left( \mathsf{distributeInsertPoses} \left( \mathrm{zip} \left( \mathsf{absInsertPoses}, \; \mathsf{spans} \right) \right) \right)$

$\mathsf{absInsertPoses} = \mathrm{sort} \left( X \right)$

$X = X_{1, \; .., \; \mathsf{nSpans}} \sim \mathrm{DiscreteUniform} \left[ 0, \; \mathsf{nPossibleInsertPoses} - 1 \right]$

$\left( \forall \; i, j \in \left\{ 1, \; .., \; \mathsf{nSpans} \right\}, X_i \ne X_j \right)$

$\mathsf{nPossibleInsertPoses} = \mathsf{seqLen} - \mathrm{sum} \left( \mathsf{spans} \right) - \mathsf{nSpans} + 1$

$\mathsf{nSpans} = \mathrm{len} \left( \mathsf{spans} \right)$

$\mathsf{spans} = \mathsf{generateSpans} \left( \mathsf{shouldMaskLen} \right)$

$\mathsf{shouldMaskLen} = \mathsf{determineShouldMaskLen} \left( \mathsf{seqLen} \right)$

# Time Complexity

The step with the highest time complexity in the algorithm is sorting the randomly generated $kn$ blanks, so the overall time complexity is $O \left( n \log n \right)$.

# Difficulties in the Design of the Algorithm

## Determining the number of words to be masked in a sentence

The algorithm requires that an average of 15% of the words in a sentence should be masked, but this calculation sometimes results in fractions. If this occur, we set the number of fractions to be rounded down or up randomly according to the fractional places. For example, if the number of words to be masked is calculated to be 3.3, a random number is randomly generated once with a uniform distribution between 0 and 1, rounded up to 4 if the number is less than 0.3, otherwise rounded down to 3.

## Randomly selecting the length of the mask

Following the BART paper, we sample from a Poisson distribution to randomly generate the length of the mask. Instead of setting the parameter of the Poisson distribution to 4 as in the BART paper, we set it to 3.5 so that masks of length 3 would occur most frequently (however, as will be mentioned in a subsequent step, we revised this parameter to 4.2). For values greater than 10, we set the probability to 0 and normalise the probability of values between 0 and 10 to sum to 1. This produces a distribution with a cumulative distribution function of [0.0151 0.0783 0.2111 0.3970 0.5922 0.7562 0.8710 0.9399 0.9760 0.9929 1.0000].

## Generate a list of the lengths of the masks

The list of mask lengths is generated by repeatedly sampling from the above distribution. The sampling stops when the sum of the lengths of the masks reaches the number of words to be masked.

If the sum of the lengths of the masks does not reach the target number of words, but the sum of the lengths of the sampled results plus the masks is greater than the target number of words (e.g. if the target number of words is 10 and the current sum of lengths is 9, but the sampled result is 5 and 9 plus 5 is greater than 10), the algorithm will discard the result of that sample and resample it until the sum of the lengths of the sampled results plus the masks is within the range of the target number of words. In practice, in order to ensure the efficiency of the algorithm, it should not re-sample when the sampling fails, but should first calculate the range of expected sampling results based on the sum of the lengths of the target words and the mask, then calculate a new distribution based on the above distribution excluding the values outside the expected range of sampling results, normalise the probability to sum to 1 and sample from the new distribution.

The algorithm requires that any two masks cannot be directly adjacent to each other, and a mask of length k actually occupies a position to its right, i.e. the actual length is k+1. Therefore, when calculating the sum of the lengths of the masks, the length of each mask needs to be added by an extra 1, i.e. the sum of the lengths of the masks plus the number of masks. Although this is a good way to avoid the problem of two masks being directly adjacent to each other, it will result in a smaller average number of words masked than the expected 15% (for this reason, the average number of words masked will be adjusted to 18.8% in a subsequent step to make the final result closer to 15%).

An asymmetry arises because samples of length 0 may occur at the start of sampling, while samples of length 0 are unlikely to occur at the end of sampling conditional on the target length being reached. For this reason, the list of mask lengths should be randomly scrambled at the end of sampling so that the lengths of the masks are randomly distributed.

## Distributing the masks evenly across the sentences

Let the length of the sentence be m, the sum of the lengths of the masks be K, and the number of masks be n. There are m-K-n+1 possible starting positions, and n of these starting positions are chosen at random as the starting positions of the masks. The reason for subtracting n is that, as mentioned above, a mask of length k actually occupies a position to its right, so n masks will occupy an additional n positions.

However, this would result in the last word of the sentence never being masked. For this reason, after the above step, a random number between 0 and 1 is randomly generated, and if this number is less than 0.5, all masks are shifted one place to the right, i.e. the empty space is assumed to be on the left, thus ensuring the symmetry of the algorithm.

## Adjusting the parameters

After implementing the algorithm, we found that the average number of words masked was less than 15%. This is due to the fact that, as mentioned above, the sum of the lengths of the masks is calculated by adding an extra 1 to the length of each blank, resulting in the actual number of words masked being less than the target number of words. For this reason, it was found that by adjusting the average number of words masked in the algorithm parameters to 18.8%, the final result was close to 15.17%, which is close to the expected value of 15%.

In addition, the algorithm generated shorter length masks more frequently than expected because the expected sampling results could only occur at smaller values as the sampling neared its end. This is allowed by the algorithm as it is only necessary to ensure that a mask of length 3 occurs most frequently. However, in order to make the algorithm generate masks of longer lengths more frequently in order to make training more difficult, we modified the parameter of the Poisson distribution from 3.5 to 4.2.

# Python Implementation

```python
import jax.numpy as np
import numpyro.distributions as dist
from random import Random

proposed_mask_rate = 0.188  # resulting mask rate would be approximately 0.15
poisson_rate = 4.2  # span length = 3 would be the most frequent in the resulting distribution
max_span_len = 10

def normalise_probs(a: np.ndarray) -> np.ndarray:
    return a / a.sum()

def generate_probs_list() -> list[list[float]]:
    probs_list = []

    poisson = dist.Poisson(rate=poisson_rate)
    probs = np.exp(poisson.log_prob(np.arange(max_span_len + 1)))

    probs_ = normalise_probs(probs)
    probs_list.append(probs_.cumsum().tolist())

    for i in range(max_span_len - 1):
        probs_ = normalise_probs(probs[:-i-1])
        probs_list.append(probs_.cumsum().tolist())

    return probs_list[::-1]

probs_list = generate_probs_list()

MaskScheme = list[tuple[int, int]]

def determine_should_mask_len(rng: Random, seq_len: int) -> int:
    x = seq_len * proposed_mask_rate
    integer_part = int(x)
    fractional_part = x - float(integer_part)
    should_add = rng.random() < fractional_part
    should_mask_len = integer_part + should_add
    return should_mask_len

def generate_spans(rng: Random, should_mask_len: int) -> list[int]:
    spans = []
    while should_mask_len > 0:
        current_max_span_len = min(max_span_len, should_mask_len)
        probs = probs_list[current_max_span_len - 1]
        span_len = rng.choices(range(current_max_span_len + 1), cum_weights=probs)[0]
        spans.append(span_len)
        should_mask_len -= span_len + 1
    rng.shuffle(spans)
    return spans

def distribute_insert_poses(abs_insert_poses: list[int], spans: list[int]) -> MaskScheme:
    offset = 0
    mask_scheme = []
    for abs_insert_pos, span in zip(abs_insert_poses, spans):
        insert_pos = abs_insert_pos + offset
        mask_scheme.append((insert_pos, span))
        offset += span + 1
    return mask_scheme

def random_add_one(rng: Random, mask_scheme: MaskScheme) -> MaskScheme:
    should_add_one = rng.random() < 0.5
    if should_add_one:
        mask_scheme = [(insert_pos + 1, span) for insert_pos, span in mask_scheme]
    return mask_scheme

def wakong(rng: Random, seq_len: int) -> MaskScheme:
    should_mask_len = determine_should_mask_len(rng, seq_len)
    spans = generate_spans(rng, should_mask_len)

    n_spans = len(spans)
    n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
    abs_insert_poses = sorted(rng.sample(range(n_possible_insert_poses), n_spans))

    mask_scheme = distribute_insert_poses(abs_insert_poses, spans)
    mask_scheme = random_add_one(rng, mask_scheme)
    return mask_scheme

def test():
    seed = 42
    rng = Random(seed)
    mask_scheme = wakong(rng, 100)
    print(mask_scheme)

if __name__ == '__main__':
    test()
```
