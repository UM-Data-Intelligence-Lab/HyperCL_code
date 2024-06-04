import torch


def build_graph(vocabulary, examples, hyperedge_dropout, device):
    # selected = int((1 - hyperedge_dropout) * len(examples))
    # examples = examples[:selected]
    s, t = [], []
    for hyperedge, example in enumerate(examples):
        L = [example.head, example.tail]
        if example.auxiliary_info:
            for i in example.auxiliary_info.values():
                L += list(i)
        for entity in vocabulary.convert_tokens_to_ids(L):
            s.append(hyperedge)
            t.append(entity)
    forward_edge = torch.tensor([t, s], dtype=torch.long).to(device)
    backward_edge = torch.tensor([s, t], dtype=torch.long).to(device)

    return forward_edge, backward_edge, len(examples)


def build_schema_graph(vocabulary, examples, device):
    s, t = [], []
    for num, example in enumerate(examples):
        L = list(example)
        e_ls = vocabulary.convert_tokens_to_ids(L)
        t.append(e_ls[0])
        s.append(e_ls[1])
    forward_edge = torch.tensor([t, s], dtype=torch.long).to(device)

    return forward_edge
