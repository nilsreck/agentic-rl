def build_convlab3_empty_state(dataset_name="multiwoz21"):
    from convlab.util import load_ontology

    ontology = load_ontology(dataset_name)
    return ontology["state"]
