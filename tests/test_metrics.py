from eval.metrics import answer_relevance, faithfulness, retrieval_precision_at_k


def test_retrieval_precision_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    relevant = ["doc_a", "doc_c"]
    # 2 hits out of 5 retrieved
    assert retrieval_precision_at_k(retrieved, relevant) == 0.4
    assert retrieval_precision_at_k([], ["doc_a"]) == 0.0
    assert retrieval_precision_at_k(retrieved, []) == 1.0


def test_answer_relevance():
    answer = "Employees traveling domestically can spend up to $75 per day on meals."
    expected = ["$75", "domestically", "meals"]
    assert answer_relevance(answer, expected) == 1.0

    missing_expected = ["$75", "business class", "director"]
    assert answer_relevance(answer, missing_expected) == 1 / 3


def test_faithfulness():
    context = ["Employees receive a $150 monthly wellness reimbursement usable for gym memberships."]
    faithful_answer = "The company offers a $150 monthly wellness reimbursement for gym memberships."
    assert faithfulness(faithful_answer, context) == 1.0

    hallucinated_answer = "Employees receive free flights to Hawaii and luxury sports cars every quarter."
    assert faithfulness(hallucinated_answer, context) == 0.0
