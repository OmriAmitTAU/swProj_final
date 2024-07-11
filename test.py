import sys

def init_vector_list(input_data):
    vectors = []
    with open(input_data) as vectors_file:
        for line in vectors_file:
            vector = line.strip()
            if vector:
                vectors.append(tuple(float(point) for point in vector.split(",")))
    return vectors

def main():
    print(len(init_vector_list(sys.argv[1])))

if __name__ == "__main__":
    main()