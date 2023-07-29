import torch
import yaml
import torchvision
import torchtext
from rota.tools import utils
from rota.olla import torch_graph_importer
from multiprocessing import Process

def load_model(model_name, batch_size, device="cpu", opti=False):
    # Initialize model.
    if model_name == "alexnet":
        model = torchvision.models.alexnet()
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
    elif model_name == "vgg":
        model = torchvision.models.vgg11()
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
    elif model_name == "mnasnet":
        model = torchvision.models.mnasnet0_5()
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
    elif model_name == "vit":
        model = torchvision.models.vit_b_16()
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
    elif model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0()
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
    elif model_name == "mobilenet":
        model = torchvision.models.mobilenet_v2()
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
    elif model_name == "bert":
        bert_base = torchtext.models.ROBERTA_BASE_ENCODER
        model = bert_base.get_model()
        transform = bert_base.transform()
        input_batch = ["Hello world"] * batch_size
        inputs = (
            torchtext.functional.to_tensor(transform(input_batch), padding_value=1),
        )
    elif model_name == "xlmr":
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_base.get_model()
        transform = xlmr_base.transform()
        input_batch = ["Hello world"] * batch_size
        inputs = (
            torchtext.functional.to_tensor(transform(input_batch), padding_value=1),
        )
    elif model_name == "gpt2-XL":
        from transformers import GPT2Tokenizer, GPT2Model
        class GPT2Wrapper(torch.nn.Module):
            def __init__(self):
                super(GPT2Wrapper, self).__init__()
                self.model = GPT2Model.from_pretrained(model_name)

            def forward(self, x):
                return self.model(x).last_hidden_state
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        max_seq_len = 1024
        text = "like you"
        text = text * (max_seq_len // len(text.split()))
        tokens = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        inputs = torch.tensor([[indexed_tokens] * batch_size])
        print(inputs.shape)
        model = GPT2Wrapper()

    if device != "cpu":
        model.to(device)
        inputs = list(inputs)
        for idx, input in enumerate(inputs):
            inputs[idx] = input.to(device)
        inputs = tuple(inputs)

    size = utils.get_model_size(model, model_name)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

    # Get model.
    importer = torch_graph_importer.TorchGraphImporter()
    g, pt_node_order = importer.import_via_aotautograd(
        model,
        *inputs,
        optimizer=optimizer,
        loss_fn=criterion,
        mode="train",
        profile=[],
        warm_up_iters=0,
        profile_iters=300,
    )

    g.name = f"{model_name}_{batch_size}_train"
    num_nodes = len(g.nodes.values())

    f = open("./rota/decomposer/config.yaml")
    settings = yaml.load(f, Loader=yaml.Loader)
    
    r = "r" + str(batch_size)
    ratio = settings[model_name][r]
    if not ratio:
        ratio = settings["default"][r]
    
    ratio = int(ratio)

    print("  CANONICALIZD MODEL", flush=True)
    g.canonicalize(True)
    print("  CONSTRAIN TENSOR GENERATORS", flush=True)
    g.constrain_tensor_generators()
    print("  CONSTRAIN ALLOCATIONS", flush=True)
    g.constrain_allocations()
    print("  DELETE SMALL INDEPENDENT GRAPH", flush=True)
    g.delete_independent_graph()            # Small independent graph in Adam.

    if opti:
        print("  ADJUST APPLY POSITION", flush=True)
        g.adjust_apply_position(ratio)
        print("  CONSTRAIN WEIGHT UPDATES EARLY", flush=True)
        g.constrain_weight_updates_early()      # Force Add to execute early.
        print("  CONSTRAIN ADAM ZERO FANIN", flush=True)
        g.constrain_adam_zero_fanin()           # Force zero-fanin node to execute lately.
    g.post_process()


    def dump_model():
        print("  PRINTING MODEL IN THE BACKGROUND", flush=True)
        with open(
            f"./logs/graphs/{model_name}/"
            + model_name
            + "_"
            + str(1)
            + "_raw_"
            + "train"
            + ".txt",
            mode="w",
        ) as f:
            f.write(str(g))

        g.dump(
            f"./logs/graphs/{model_name}/" + model_name + "_" + str(batch_size) + "_raw_" + "train",
            format="svg",
        )

    p = Process(target=dump_model, name="dump_" + model_name, daemon=False)
    p.start()

    return g, g.name, size, num_nodes, pt_node_order