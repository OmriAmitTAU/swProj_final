import sys


class InputError(Exception):
    def __init__(self, cause):
        self.cause = cause

    def __str__(self):
        if self.cause == "n":
            return "Invalid number of points!"
        elif self.cause == "k":
            return "Invalid number of clusters!"
        elif self.cause == "d":
            return "Invalid dimension of point!"
        elif self.cause == "max_iter":
            return "Invalid maximum iteration!"
        else:
            return "An Error Has Occurred"


def euclidean_distance(vec1, vec2):
    d = 0
    for x1, x2 in zip(vec1, vec2):
        d_single = (x1 - x2) ** 2
        d += d_single
    return d ** 0.5


def to_float(num, arg_name):
    try:
        return float(num)
    except:
        raise InputError(arg_name)


def is_natural(num):
    return num > 0 and num - int(num) == 0


def parse_input():
    if 5 <= len(sys.argv) <= 6:
        k, n, d = [to_float(arg, arg_name) for arg, arg_name in zip(sys.argv[1:4], ["k", "n", "d"])]
        max_iter = to_float(sys.argv[4], "max_iter") if len(sys.argv) == 6 else 200
        if not assert_input(k, n, d, max_iter):
            return
        return k, n, d, max_iter
    else:
        raise Exception()


def assert_input(k, n, d, max_iter):
    if not (n > 1 and is_natural(n)):
        raise InputError("n")
    if not (1 < k < n and is_natural(k)):
        raise InputError("k")
    if not (is_natural(d)):
        raise InputError("d")
    if not (1 < max_iter < 1000 and is_natural(max_iter)):
        raise InputError("max_iter")
    return True


def init_vector_list(input_data):
    vectors = []
    with open(input_data) as vectors_file:
        for line in vectors_file:
            vector = line.strip()
            if vector:
                vectors.append(tuple(float(point) for point in vector.split(",")))
    return vectors


def find_closest_centroid(curr, k_centroids):
    closest_d = float("inf")
    closest_centroid = None

    for centroid in k_centroids.keys():
        distance = euclidean_distance(curr, centroid)
        if distance < closest_d:
            closest_d = distance
            closest_centroid = centroid
    return closest_centroid


def calculate_updated_centroid(centroid_vectors, d):
    num_vectors = len(centroid_vectors)

    updated_centroid = []
    for i in range(d):
        avg_point = (sum(v[i] for v in centroid_vectors)) / num_vectors
        updated_centroid.append(avg_point)
    return tuple(updated_centroid)


def assign_to_closest_centroid(datapoints, k_centroids, vectors_mapping):
    for curr_vect in datapoints:
        closest_centroid = find_closest_centroid(curr_vect, k_centroids)
        prev_centroid = vectors_mapping[curr_vect]  # find prev mapping
        if closest_centroid != prev_centroid:
            if prev_centroid is not None:
                k_centroids[prev_centroid].remove(curr_vect)  # remove the curr from the prev centroid
            vectors_mapping[curr_vect] = closest_centroid  # add to new mapping
            k_centroids[closest_centroid].append(curr_vect)  # add the curr to its closest centroid


def k_means(k, n, d, input_data, max_iter=200):
    e = 0.001
    datapoints = init_vector_list(input_data)
    k_centroids = {datapoints[i]: [datapoints[i]] for i in range(k)}
    vectors_mapping = {vector: vector if vector in k_centroids.keys() else None for vector in datapoints}
    i = 0
    delta_miu = float("inf")
    while delta_miu >= e or i < max_iter:
        assign_to_closest_centroid(datapoints, k_centroids, vectors_mapping)
        new_k_centroids = dict()
        for curr_centroid in k_centroids:
            updated_centroid = calculate_updated_centroid(k_centroids[curr_centroid], d)
            new_k_centroids[updated_centroid] = k_centroids[curr_centroid]  # change to new centroid
            for vector in new_k_centroids[updated_centroid]:
                vectors_mapping[vector] = updated_centroid
            delta_miu = min(euclidean_distance(curr_centroid, updated_centroid), delta_miu)
        k_centroids = new_k_centroids

        i += 1

    for final_centroid in k_centroids.keys():
        centroid_str = ','.join('{:.4f}'.format(coord) for coord in final_centroid)
        print(centroid_str)


def main():
    try:
        k, n, d, max_iter = map(int, parse_input())
        input_file = sys.argv[-1]
        k_means(k, n, d, input_file, max_iter)
    except InputError as e:
        print(e)
    except Exception:
        print("An Error Has Occurred")


if __name__ == "__main__":
    main()
