import os


class ExampleFeature(object):
    """The code and description feature for an example."""
    def __init__(self,
                 func,
                 func_name,
                 cwe_id,
                 source,
                 sink,
                 description,
                 label,
                 idx
                 ):

        self.func = func
        self.func_name = func_name
        self.cwe_id = cwe_id
        self.source = source
        self.sink = sink

        self.description = description
        self.label = label
        self.idx = idx


def generate_description(js):
    func = js["func"]
    func_name = js["name"]
    cwe_id = js["cwe_id"]
    source = js["source"]
    sink = js["sink"]
    label = js["label"]
    idx = js["idx"]
    description = js["reason"]
    return ExampleFeature(func, func_name, cwe_id, source, sink, description, label, idx)
