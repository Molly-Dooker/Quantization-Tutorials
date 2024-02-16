import torch
import torch.quantization as tq
import torch.ao.quantization as taq
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.nn.quantized as nnq
import torch.fx as fx

from evaluate import evaluate
from qconfigs import learnable_act, learnable_weights, fixed_act
from ipdb_hook import ipdb_sys_excepthook

from model.resnet import resnet18

ipdb_sys_excepthook()

model = resnet18(pretrained=True)

# from torch.ao.quantization import get_default_qconfig_mapping
# qconfig_mapping = get_default_qconfig_mapping("fbgemm")

#qconfig_layer = tq.QConfig(
    #activation=learnable_act(range=2),
    #weight=learnable_weights
#)

qconfig_FF = tq.QConfig(
    activation=learnable_act(range=2),
    weight=tq.default_fused_per_channel_wt_fake_quant
)

qconfig_QS_DQ = tq.QConfig(
    activation=fixed_act(min=0, max=1),
    weight=tq.default_fused_per_channel_wt_fake_quant
)


qconfig_mapping = QConfigMapping().set_object_type((taq.QuantStub, taq.DeQuantStub), qconfig_QS_DQ) \
                                .set_object_type(nnq.FloatFunctional, qconfig_FF)

# Awkward we have to do this manually, just for the sake of accessing the `out_channels` attribute
for name, module in model.named_modules():
    if hasattr(module, 'out_channels'):
        qconfig = tq.QConfig(
            activation=learnable_act(range=2),
            weight=learnable_weights(channels=module.out_channels)
        )
        qconfig_mapping.set_module_name(name, qconfig)
        module.qconfig = qconfig


example_inputs = (torch.randn(1, 3, 224, 224),)

model.eval()
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

model.train()
fake_quant_model = tq.prepare_qat(model, inplace=False)

# NOTE: need to figure out how to place the fixed qparams qconfig correctly
# at beginning and end. Also, mention that PTQ is on in both cases, so we are cheating
# by doing dynamic quant to some degree.
# `activation_post_process` is also moved outside the module, so each module has
# a `weight_fake_quant` attribute, but the `activation_post_process` is a seperate requantization
# step, and it isn't attached.
print('\n Original')
evaluate(model, 'cpu')

print('\n FX prepared')
evaluate(fx_model, 'cpu')

print('\n Eager mode prepared')
evaluate(fake_quant_model, 'cpu')
# Can experiment with visualize the graph, e.g.
# >> fx_model
# >> print(fx_model.graph)  # prints the DAG

# Prints the graph as a table
print("\nGraph as a Table:\n")
fx_model.graph.print_tabular()

# Plots the graph
# Need to install GraphViz and have it on PATH (or as a local PATH variable)
#from torch.fx import passes
#g = passes.graph_drawer.FxGraphDrawer(fx_model, 'fx-model')
#with open("graph.svg", "wb") as f:
    #f.write(g.get_dot_graph().create_svg())

# We will correct the graph to have fixedqparams on the input quantstub
# NOTE: taken from https://pytorch.org/docs/stable/fx.html#graph-manipulation
def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    """
    This function takes a PyTorch module `m` and returns a new PyTorch module that
    transforms the original module by applying a specific transformation to any
    function calls found within the original module's graph. The transformation
    is specified by a custom tracer class that iterates through the graph nodes
    and checks if the current node is a call function node. If it is a call function
    node and the target function is `torch.add`, the tracer replaces the target
    attribute with `torch.mul`. The transformed graph is then wrapped into a
    `fx.GraphModule` to produce the final transformed module.

    Args:
        m (torch.nn.Module): The `m` parameter is the torch.nn module to be transformed.
        tracer_class (fx.Tracer): The tracer class parameter is an optional argument
            that can be set to any instance of the FX Tracer class; if left unset
            ("FX.Tracer"), then it uses the default FX Tracer instance. When you
            use a custom tracer object other than the standard one by specifying
            a type other than FX.Tracer as the argument value., It's an optional
            argument and can be specified only if needed

    Returns:
        torch.nn.Module: The output of the given function is a Torch graph module
        object (`fx.GraphModule`) wrapped around the original input PyTorch model
        instance (`m`).

    """
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)

# Experiment with iterator pattern:
# NOTE: taken from https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
from torch.fx.node import Node
from typing import Dict

class GraphIteratorStorage:
    """
    A general Iterator over the graph. This class takes a `GraphModule`,
    and a callable `storage` representing a function that will store some
    attribute for each node when the `propagate` method is called.

    Its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments, e.g. an example input tensor.
    As each operation executes, the GraphIteratorStorage class stores
    away the result of the callable for the output values of each operation on
    the attributes of the operation's `Node`. For exmaple,
    one could use a callable `store_shaped_dtype()` where:

    ```
    def store_shape_dtype(result):
        if isinstance(result, torch.Tensor):
            node.shape = result.shape
            node.dtype = result.dtype
    ```
    This would store the `shape` and `dtype` of each operation on
    its respective `Node`, for the given input to `propagate`.
    """
    def __init__(self, mod, storage):
        """
        This function is initializing an object for the class `MyModule` with the
        arguments `mod` and `storage`. It sets the object's attributes `mod`,
        `graph`, and `modules` to the corresponding values from the argument `mod`,
        and sets the attribute `storage` to the value of the argument `storage`.

        Args:
            mod (): The input `mod` parameter is a module object that the class
                instance is being created for. It holds the modules attributes
                like graph and named modules
            storage (dict): Here's the definition:
                
                def __init__(self (...)storage):
                     ...
                
                It means that during instance creation of an object based on this
                class - which might be refered to as "module" by convention here
                due to graph and mod attributes created within the body of its
                class scope using the values passed as parameters upon initialization
                of said module - the function accepts one parameter 'storage';
                which is also then saved under the attribute called 'storage'.

        """
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.storage = storage

    def propagate(self, *args):
        """
        This is a Python function `propagate` defined inside another object-oriented
        language construct such as a class or module. Its purpose is to interpret
        a Graph Module format description of computations written using the Torch
        Fusion API and propagate the computation by executing each node and storage
        operation within the graph. It also takes an optional arg `*args`, which
        it doesn't seem to use (although that could have been changed somewhere
        higher up and broken here).

        """
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            """
            This function maps each argument `a` of the function to an environment
            variable `env[n.name]`

            Args:
                a (): The `a` input parameter is a tensor that gets passed into
                    the function `load_arg()`. This tensor gets used to look up
                    entries within the `env` dictionary using its string values
                    as keys to extract these entries. The goal of doing so is to
                    turn these key-value pairs into PyTorch TensorArguments for
                    efficient handling later on downstream.

            Returns:
                : The output returned by the function "load_arg" is a Python dictionary.

            """
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            """
            This function fetches an attribute from a given target string by
            iterating through the dot-separated components of the target and
            retrieving the attribute from each component's object using getattr().
            If any component is non-existent. it raises a RuntimeError. The last
            component's attribute value is returned.

            Args:
                target (str): The target input parameter is the name of an attribute
                    to retrieve within a module using dotted notation (e.g., '
                    Foo.bar'). The function breaks apart the given target into its
                    atoms and iterates through the attributes within each atom
                    using the object's module until it successfully retrieves the
                    desired attribute value. In simpler words : target parameter
                    is a string that we are going to look up

            Returns:
                : The output returned by this function is nonexistent node. Because
                if any of the target nodes do not exist then it raises a RuntimeError
                exception.

            """
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to the `storage` function.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter
            self.storage(node, result)

            env[node.name] = result
        #return load_arg(self.graph.result)

def store_shape_dtype(node, result):
    """
    Function that takes in the current node, and the tensor it is operating on (`result`)
    and stores the shape and dtype of `result` on the node as attributes.
    """
    if isinstance(result, torch.Tensor):
        node.shape = result.shape
        node.dtype = result.dtype
# NOTE: I just discovered that they have the `Interpreter` class, which accomplishes the same thing:
# https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter
GraphIteratorStorage(fx_model, store_shape_dtype).propagate(example_inputs[0])
for node in fx_model.graph.nodes:
    print(node.name, node.shape, node.dtype)
xxx

