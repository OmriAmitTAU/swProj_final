import sys
from math import sqrt
import numpy as np
import mysymnmf

def to_number(num):
    try:
        return int(float(num))
    except:
        raise Exception()

def init_vector_list(input_data):
    vectors = []
    with open(input_data) as vectors_file:
        for line in vectors_file:
            vector = line.strip()
            if vector:
                vectors.append(tuple(float(point) for point in vector.split(",")))
    return vectors

def parse_input():
    k = to_number(sys.argv[1])
    goal = sys.argv[2]
    input_data = sys.argv[3]

    datapoints = init_vector_list(input_data)
    n = len(datapoints)
    d = len(datapoints[0])
    return datapoints, k, goal, n, d

def init_h(n, k, W):
    np.random.seed(0)
    mean_w = np.mean(W)
    constant_term = 2 * sqrt(mean_w / k)
    H = np.random.uniform(0, high=constant_term, size=(n, k))
    return H.tolist()

def logic(datapoints, k, goal, n, d):
    if goal == "symnmf":
        W = mysymnmf.norm(0, n, d ,datapoints)
        H = init_h(datapoints, k, W)
        mysymnmf.symnmf(k, n, W, H, 0)
    elif goal == "sym":
        mysymnmf.sym(n, d, datapoints)
    elif goal == "ddg":
        mysymnmf.ddg(n, d, datapoints)
    elif goal == "norm":
        mysymnmf.norm(1, n, d, datapoints)
    else:
        raise Exception()

def main():
    try:
        datapoints, k, goal, n, d = parse_input()
        logic(datapoints, k, goal, n, d)
    except:
        print("An Error Has Occurred")

if __name__ == "__main__":
    main()