import numpy as np
import sklearn.preprocessing
import scipy

k = 2 # number of features
j = 3 # number of possible values per feature
n = 5 # number of images
epsilon = 1e-4
alpha = 10
beta = 0.5 # tradeoff parameter


# vocab: all possible values + empty for each feature
W = {}
for f in range(k):
    W[f] = np.arange(j+1)

# messages: all combinations of the k*j features (unified with the empty feature)
print("\n possible messages")
messages = np.array(np.meshgrid(*[W[k] for k in W.keys()])).T.reshape(-1, k)
print(messages)

# images: set of n feature vectors (but without empty)
image_candidats = np.squeeze(messages[np.argwhere(np.all(messages!=0, -1))])
images = image_candidats[np.random.choice(np.arange(0, len(image_candidats)), n, replace=False)]
print("\nimages:\n",images)

def get_meaning_matrix(images, messages):
    """
    meaning is true if features are equal or the message feature is 0
    thus, the root of images * messages must be equal to messages for all features
    Args:
        images: matrix of size nxk
        messages: matrix of size (j+1^k)xk
    Returns:
^       matrix of size (j+1^k)xn wtih entries 1 or epsilon
    """
    m2 = np.expand_dims(messages, 1)
    im_m = np.asarray([images*m for m in messages])
    b = np.asarray(np.all(np.sqrt(im_m) == m2, -1), dtype=np.int)
    b = np.where(b == 0, epsilon, b)

    return b

def make_partiton(feature):
    """
    Args:
        feature: int vector
    Returns:
        issue: set of F which contain feature
    """
    # all images where feature is present
    issue_idx = np.squeeze(np.argwhere(np.any(images==feature, -1)))
    issue_img = images[issue_idx]

    return issue_img, issue_idx

def get_B_matrix(messages, images):
    return get_meaning_matrix(images, messages)

def get_S0_matrix(B):
    return sklearn.preprocessing.normalize(B, norm="l1")

def get_L1_matrix(S0):
    return sklearn.preprocessing.normalize(np.transpose(S0), norm="l1").T

def get_U1_vector(L1_issue):
    """
    Args:
        L1_issue: matrix of size messages x number of images in issue

    Returns:
        vector of size messages x 1

    """
    # sum over all images in issue
    U1 = np.sum(np.log(L1_issue),-1)

    return U1

def get_U2_vector(L1_issue):
    """
       Args:
           L1_issue: matrix of size messages x number of images in issue

       Returns:
           vector of size messages x 1

       """
    U2 = scipy.stats.entropy(L1_issue, axis=-1)

    return U2

def get_cost_vector(B, img_idx):
    """
    Args:
        B: original meaning matrix (number of messages x n)
        img_idx: int,  index of original image

    Returns:
        vector of size messages x 1

    """
    c = -np.log(B)
    # over original image only
    c = c[:, img_idx]

    return c

def get_S1_vector(image_idx, issue_ids, B=None, S0=None, L1=None):
    """
    Args:
        image_idx: index of image in all images
        issue_ids: list of indexes (referring to images in the images list)
    Returns:
        vector of size number of messages

    """

    if issue_ids.size == 0:
        print("Empty Issue! Try again for a different image-feature combination.")
        raise ValueError

    if B is None: B = get_B_matrix(messages, images)
    if L1 is None:
        if S0 is None:
            S0 = get_S0_matrix(B)
        L1 = get_L1_matrix(S0)

    l1 = L1[:, issue_ids]     # filter for images in issue
    cost = get_cost_vector(B, image_idx)  # at the index of original images
    u1 = get_U1_vector(l1)
    u2 = get_U2_vector(l1)

    return np.exp(alpha * ((1 - beta) * u1 + beta * u2)) - cost

if __name__ == "__main__":

    image_idx = np.random.randint(0, n)
    image = images[image_idx]
    print(f"\noriginal image: {image}")

    feature = image.copy()
    feature[np.random.randint(0, k)]=0
    print(f"\nissue for feature {feature}:")
    issue, issue_ids = make_partiton(feature)
    print(issue)
    B = get_B_matrix(messages, images)

    print(f"\nvalid messages \n")
    b_1 = np.argwhere(B[:, image_idx] == 1)
    print(*messages[b_1])

    print(f"\nliteral speaker says: \n")
    S0 = get_S0_matrix(B)
    s0_max_ids = np.argwhere(S0[:, image_idx] == np.amax(S0[:, image_idx]))
    print(*messages[s0_max_ids])

    print(f"\npragmatic listener says: \n")
    L1 = get_L1_matrix(S0)
    l1_max_ids = np.argwhere(L1[:, image_idx] == np.amax(L1[:, image_idx]))
    print(*messages[l1_max_ids])

    print("\nissue-sensitive speaker says: \n")
    S1 = get_S1_vector(image_idx, issue_ids, B, L1)
    s1_max_ids = np.argwhere(S1 == np.amax(S1))
    print(*messages[s1_max_ids])


