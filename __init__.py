from .nodo import RenaultDegradeNode

NODE_CLASS_MAPPINGS = {
    "RenaultDegradeNode": RenaultDegradeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RenaultDegradeNode": "Degradación Dashcam (Renault)"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
