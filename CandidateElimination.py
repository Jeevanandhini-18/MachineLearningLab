# Candidate Elimination Algorithm in Python

def more_general_or_equal(h1, h2):
    """Check if hypothesis h1 is more general than or equal to h2"""
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "∅" and (x == y))
        more_general_parts.append(mg)
    return all(more_general_parts)


def generalize_S(example, S):
    S_new = list(S)
    for i in range(len(S)):
        if S[i] != example[i]:
            S_new[i] = "?"
    return tuple(S_new)


def specialize_G(hypothesis, example):
    specializations = []
    for i in range(len(hypothesis)):
        if hypothesis[i] == "?":
            if example[i] != "∅":
                new_hypothesis = list(hypothesis)
                new_hypothesis[i] = example[i]
                specializations.append(tuple(new_hypothesis))
    return specializations


def candidate_elimination(data):
    # Initialize S and G
    S = list(data[0][0])  # first positive example
    G = [("?",) * len(S)]

    print("Initial S:", S)
    print("Initial G:", G)
    print("----------------------------------")

    for example, label in data:
        print("Training Example:", example, "Label:", label)

        if label == "Yes":  # Positive Example
            # Remove inconsistent hypotheses from G
            G = [g for g in G if more_general_or_equal(g, example)]
            # Generalize S
            S = generalize_S(example, S)

        else:  # Negative Example
            # Specialize G
            new_G = []
            for g in G:
                if more_general_or_equal(g, example):
                    spec = specialize_G(g, example)
                    for h in spec:
                        if more_general_or_equal(h, S):
                            new_G.append(h)
                else:
                    new_G.append(g)
            G = new_G

        print("S =", S)
        print("G =", G)
        print("----------------------------------")

    return S, G


# ----------------- SAMPLE DATASET -----------------
# Attributes: Weather, Temperature, Humidity, Wind
# Class: Yes / No

dataset = [
    (("Sunny", "Hot", "High", "Weak"), "No"),
    (("Sunny", "Hot", "High", "Strong"), "No"),
    (("Overcast", "Hot", "High", "Weak"), "Yes"),
    (("Rain", "Mild", "High", "Weak"), "Yes"),
    (("Rain", "Cool", "Normal", "Weak"), "Yes"),

]

# Run Algorithm
final_S, final_G = candidate_elimination(dataset)

print("Final Specific Hypothesis S =", final_S)
print("Final General Hypothesis G =", final_G)
