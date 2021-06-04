import numpy as np
import itertools

k = 2 # number of features
n = 3 # number of images

epsilon = 1e-4 # noise free
alpha = 10
empty = "" # token for empty
prior = 1/n # flat prior over images
beta = 0.5 # tradeoff parameter


# I set of images whereas each image is a set of features
# eg
F_1 = "a red square"
F_2 = "a green circle"
F_3 = "a yellow circle"

I = {
    F_1: ["red", "square"],
    F_2: ["green", "circle"],
    F_3: ["yellow", "circle"]
    }


W = {
    "f1" : ["red", "green", "yellow", empty],
    "f2" : ["square", "circle", "triangle", empty]
    }
# a message is a set of features unified with the empty sequence
print("\n possible messages")
messages=[]
combinations = [list(zip(each_permutation, W["f2"])) for each_permutation in itertools.permutations(W["f1"], len(W["f2"]))]
for cf in combinations:
    for c in cf:
        messages.append(c)
messages = list(set(messages))
print(messages)


def meaning(image, message):
    """
    Args:
        image: feature vector
        message: vector with same lenght as image, each entry is either a feature or empty

    Returns:
^       prob: float, 1 or epsilon
    """
    subset = True
    # 1 if the message is a subset of the image's feature else epsilon
    for i_j, m_j in zip(I[image],message):
        if m_j == empty:
            continue
        elif i_j != m_j:
          subset = False

    if subset:
        return 1.0
    else:
        return epsilon

def make_partiton(feature):
    """
    Args:
        feature: string feature e.g. "cricle"

    Returns:
        issue: set of F which contain feature

    """
    issue = []
    # all images where feature is present
    for i in I.keys():
        if feature in I[i]:
            issue.append(i)

    return issue

def S0(message, image):
    return meaning(image, message)

def L1(message, image):
    return prior * S0(message, image)

def U1(message, issue):
    # log over listener of ?? all images in context (sum) or only specific image
    u1 = 0
    for i in issue:
        u1 += np.log(L1(message, i))
    return u1

def U2(message, issue):
    # entropy over literal speaker
    entropy = - sum([L1(message, i) * np.exp(L1(message, i)) for i in issue])
    return entropy

def cost(image, message):
    return -np.log(meaning(image, message))

def S1(message, image, issue):
    u1 = U1(message, issue)
    u2 = U2(message, issue)
    c = cost(image, message)

    return np.exp(alpha * ((1 - beta) * u1 + beta * u2)) - c

if __name__ == "__main__":
    feature = "circle"
    issue = make_partiton(feature)
    image = F_2
    print("\n image:", image)
    print(f"issue for feature {feature}:")
    print(issue)

    print("\n literal speaker")
    scores_s0 = [S0(m,image) for m in messages]
    #print("scores", scores_s0)
    max_score_s0 = max(scores_s0)
    max_indexes_s0 = [i for i, j in enumerate(scores_s0) if j == max_score_s0]
    print([messages[i] for i in max_indexes_s0])

    print("\n pragmatic listener")
    print("should be equal to base speaker for a flat image prior distribution")

    print("\n issue sensitixe speaker")
    scores_s1 = [S1(m, image, issue) for m in messages]
    #print("scores", scores_s1)
    max_score_s1 = max(scores_s1)
    max_indexes_s1 = [i for i,j in enumerate(scores_s1) if j == max_score_s1]
    print([messages[i] for i in max_indexes_s1])

