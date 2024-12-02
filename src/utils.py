import torch
from collections import defaultdict
import tqdm.auto as tqdm

def get_hidden_activations(model, hookpoint, tokenized_inputs):

    li = []

    def hook(module, _, outputs):

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        li.append(outputs[0].detach())

    attached_hook = model.get_submodule(hookpoint).register_forward_hook(hook)

    for tokenized_input in tokenized_inputs:
        output = model(**tokenized_input)
        del output

    attached_hook.remove()
    return torch.concat(li, dim=0)


def get_hidden_activations_multiple_hookpoints(model, hookpoints, tokenized_inputs):

    dict_ = defaultdict(list)

    def make_hook(hookpoint):

        def hook(module, _, outputs):

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            dict_[hookpoint].append(outputs[0].detach())

        return hook

    attached_hooks = [
        model.get_submodule(hookpoint).register_forward_hook(make_hook(hookpoint))
        for hookpoint in hookpoints
    ]

    for tokenized_input in tokenized_inputs:
        output = model(**tokenized_input)
        del output

    for attached_hook in attached_hooks:
        attached_hook.remove()

    new_dict = {
        hookpoint: torch.concat(li, dim=0)
        for hookpoint, li in dict_.items()
    }

    return new_dict


def get_input_size(model, hookpoint):

    activation_output = None

    def hook_fn(module, input, output):
        nonlocal activation_output
        activation_output = output

    submodule = model.get_submodule(hookpoint)
    hook = submodule.register_forward_hook(hook_fn)

    sample_input = torch.randint(0, model.config.vocab_size, (1, 10)).to(model.device)
    model(sample_input)

    hook.remove()

    input_size = activation_output.shape[-1]

    return input_size


# def get_activations(text_batch, model, hookpoints, tokenizer, device):
#     inputs = [
#         tokenizer(text, return_tensors='pt').to(device)
#         for text in text_batch
#     ]

#     activations = get_hidden_activations_multiple_hookpoints(model, hookpoints, inputs)

#     return activations


# def activation_generator(dataloader, model, hookpoints, tokenizer, device):
#     for text_batch in tqdm.tqdm(dataloader):
#         yield get_activations(text_batch, model, hookpoints, tokenizer, device)


def get_activations_and_labels(text_batch, label_batch, model, hookpoints, tokenizer, device):

    inputs = [
        tokenizer(text, return_tensors='pt').to(device)
        for text in text_batch
    ]

    activations = get_hidden_activations_multiple_hookpoints(model, hookpoints, inputs)

    labels = torch.concat(
        [
            torch.Tensor([label] * len(input_['input_ids'][0]))
            for label, input_ in zip(label_batch, inputs)
        ]
    ).unsqueeze(-1).to(device)

    return activations, labels


def activation_label_generator(dataloader, model, hookpoints, tokenizer, device):
    for text_batch, label_batch in tqdm.tqdm(dataloader):
        yield get_activations_and_labels(text_batch, label_batch, model, hookpoints, tokenizer, device)

