### LIBRARIES ###
# Global libraries
import os
import torch
import copy
import logging

logger = logging.getLogger(__name__)

### FUNCTION DEFINITION ###
def load_conceptBert(model, path_pretrained):
    serialization_dir = "/".join(path_pretrained.split("/")[:-1])
    WEIGHTS_NAME = path_pretrained.split("/")[-1]    
        
    # Load config
    weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
    )
    if "state_dict" in dir(state_dict):
        state_dict = state_dict.state_dict()
        
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
        
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # Copy `state_dict`, so that `_load_from_state_dict` can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")
                
    start_prefix = ""
    # Use the old name for old model
    # if not hasattr(model, "Kilbert") and any(s.startswith("Kilbert.") for s in state_dict.keys()):
    #     start_prefix = "Kilbert."
    if not hasattr(model, "ConceptBert") and any(s.startswith("ConceptBert.") for s in state_dict.keys()):
        start_prefix = "ConceptBert."
    load(model, prefix=start_prefix)
    
    if len(missing_keys) > 0:
        logger.info(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        logger.info(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
        
    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}: \n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)
            )
        )
    return model
